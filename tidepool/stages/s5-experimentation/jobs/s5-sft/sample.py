"""Role-balanced sampling for the supervised mix, and the calibration behind it.

The rendered corpus puts 79% of its 201.5M tokens on SQL and code, 20% on tool calling
and 0.7% on structured output. Training on it as it stands spends the budget inversely to
the project's own priority order, so the reference arm samples it instead. The weights are
not hand-picked: they fall out of three stated constraints, and the numbers they produce
are written into the run's score dict so the choice is auditable rather than asserted.

  1. Tool calling is the first priority and is never downsampled: it gets `tool_epochs`
     full passes.
  2. Structured output is tiny (2,542 rows). Upsampling is the only way to give it weight
     at all, and repetition buys memorization past a point, so it is capped at
     `struct_max_epochs` passes and no further.
  3. Tool plus structured output together take at least `min_priority_share` of the epoch's
     token budget. SQL and code fill the remainder, downsampled in proportion to their own
     sizes so neither is favoured over the other.

The epoch budget is whatever those three constraints imply. It is not a free parameter,
which is the point: nothing here can be tuned to make a mix look better after the fact.
"""

import gzip
import hashlib
import json

ROLES = ("tool", "struct", "sql", "code")
FILLER = ("sql", "code")


def calibrate(avail, tool_epochs=1.0, struct_max_epochs=3.0, min_priority_share=0.5):
    """avail: {role: available tokens}. Returns the plan, in tokens, per role."""
    priority = avail.get("tool", 0) * tool_epochs + avail.get("struct", 0) * struct_max_epochs
    filler_avail = sum(avail.get(r, 0) for r in FILLER)
    want_filler = priority / min_priority_share - priority
    # If the corpus does not hold enough filler to dilute the priority roles down to the
    # floor, that is not a failure: the priority share simply lands above it.
    filler = min(want_filler, filler_avail)
    plan = {"tool": avail.get("tool", 0) * tool_epochs,
            "struct": avail.get("struct", 0) * struct_max_epochs}
    for r in FILLER:
        plan[r] = filler * (avail.get(r, 0) / filler_avail) if filler_avail else 0.0
    total = sum(plan.values()) or 1.0
    return {
        "budget_tokens": int(total),
        "per_role_tokens": {r: int(plan.get(r, 0)) for r in ROLES},
        "per_role_share": {r: round(plan.get(r, 0) / total, 4) for r in ROLES},
        "per_role_repeats": {r: round(plan.get(r, 0) / avail[r], 4) if avail.get(r) else 0.0
                             for r in ROLES},
        "priority_share_achieved": round((plan["tool"] + plan["struct"]) / total, 4),
        "priority_share_floor": min_priority_share,
        "filler_capped_by_corpus": want_filler > filler_avail + 1,
        "available_tokens": {r: int(avail.get(r, 0)) for r in ROLES},
    }


def _rank(row, salt):
    """A deterministic per-row order that does not depend on file order or dict iteration."""
    key = "%s|%s|%s" % (row.get("c"), row.get("i"), salt)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def index_split(path, log=None):
    """One streaming pass: per-role token totals and a per-role list of (rank, line_no)."""
    per_role, order = {}, {}
    with gzip.open(path, "rt") as fh:
        for ln, line in enumerate(fh):
            r = json.loads(line)
            role = r.get("role") or "unknown"
            per_role[role] = per_role.get(role, 0) + int(r.get("n_tok") or 0)
            order.setdefault(role, []).append((_rank(r, "mix"), ln))
    for role in order:
        order[role].sort()
    if log:
        log("indexed %d roles: %s" % (len(per_role), json.dumps(per_role)))
    return per_role, order


def choose(order, per_role, plan, log=None):
    """Line numbers to train on, with repeats, honouring the calibrated token plan.

    Rows are taken in a hash order that is stable across runs, so two arms that differ in
    one hyperparameter see the same rows in the same order and the comparison is paired.
    """
    lines, detail = [], {}
    for role, want in plan["per_role_tokens"].items():
        rows = order.get(role) or []
        if not rows or want <= 0:
            detail[role] = {"rows_taken": 0, "tokens": 0}
            continue
        avail = per_role.get(role, 0)
        mean_tok = avail / len(rows)
        n_want = int(round(want / mean_tok)) if mean_tok else 0
        taken = []
        while len(taken) < n_want:
            need = n_want - len(taken)
            taken.extend(ln for _, ln in rows[:need])
        lines.extend(taken)
        detail[role] = {"rows_taken": len(taken), "rows_available": len(rows),
                        "tokens": int(len(taken) * mean_tok),
                        "mean_tokens_per_row": round(mean_tok, 1)}
    # Interleave by hash so the roles are mixed through the epoch rather than blocked,
    # which matters for a single-epoch run where a blocked role trains last and dominates.
    lines.sort(key=lambda ln: hashlib.blake2b(str(ln).encode(), digest_size=8).digest())
    if log:
        log("selected %d training rows: %s" % (len(lines), json.dumps(detail)))
    return lines, detail


def read_lines(path, wanted):
    """Materialize the selected rows, preserving the requested order and repeats."""
    need = {}
    for pos, ln in enumerate(wanted):
        need.setdefault(ln, []).append(pos)
    out = [None] * len(wanted)
    with gzip.open(path, "rt") as fh:
        for ln, line in enumerate(fh):
            if ln in need:
                r = json.loads(line)
                for pos in need[ln]:
                    out[pos] = r
    return [r for r in out if r is not None]
