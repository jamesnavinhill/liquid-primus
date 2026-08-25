"""Role-balanced sampling for the supervised sweep, under a fixed per-arm token budget.

This is `s5-sft-l40s/sample.py` with two things added, and the reasoning behind both is
worth keeping next to the code.

**A fixed budget.** At `s5.1` the epoch budget was whatever the three mix constraints
implied, which was the right shape for a calibration run and the wrong one for a sweep.
Eight arms that differ in a hyperparameter must cost the same, or the comparison measures
compute as much as it measures the hyperparameter. Every arm therefore trains on exactly
`budget_tokens`, and the calibrated shares are rescaled to hit it rather than recomputed.
The measured L40S rate of 3,994.2 tok/s turns 64.0M tokens into 4.45 GPU-hours per arm.

**A guardrail block.** `s5.3` built 7,988 rows of defective tool returns paired with their
intact counterparts, and the amended stop/go gate asks the probe flag rate to move from
0.0074 to 0.35. The block is dosed in epochs like structured output is, not sampled by
share, because its size is fixed and small and what matters is how many times the model
sees it. It is taken **out of** the budget, not added to it, so the `C2'` ablation that
removes it trains on the same number of tokens and the comparison is guardrail-block
against an equal weight of the base mix.

The three original constraints are unchanged:

  1. Tool calling is never downsampled: `tool_epochs` full passes.
  2. Structured output is tiny and repetition buys memorization past a point, so it is
     capped at `struct_max_epochs` passes.
  3. Tool plus structured output take at least `min_priority_share` of the pre-scaling
     budget. SQL and code fill the remainder in proportion to their own sizes.

Nothing here can be tuned to make a mix look better after the fact: every derived number
lands in the run's `mix_calibration.json` artifact.
"""

import gzip
import hashlib
import json

BASE_ROLES = ("tool", "struct", "sql", "code")
GUARDRAIL = "tool_guardrail"
REPLAY = "replay"
ROLES = BASE_ROLES + (GUARDRAIL, REPLAY)
FILLER = ("sql", "code")
PRIORITY = ("tool", "struct", GUARDRAIL)


def calibrate(avail, tool_epochs=1.0, struct_max_epochs=3.0, min_priority_share=0.5,
              guardrail_epochs=2.0, replay_frac=0.0, budget_tokens=None):
    """avail: {role: available tokens}. Returns the plan, in tokens, per role."""
    # --- the s5.1 calibration, over the four base roles only -----------------
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
    natural = sum(plan.values()) or 1.0

    # --- the fixed doses that come out of the budget before it is shared -----
    guard = avail.get(GUARDRAIL, 0) * guardrail_epochs
    notes = {}
    if budget_tokens:
        budget = float(budget_tokens)
        replay = budget * max(0.0, replay_frac)
        if avail.get(REPLAY):
            replay = min(replay, avail[REPLAY])
        else:
            if replay_frac > 0:
                notes["replay_unavailable"] = True
            replay = 0.0
        room = budget - guard - replay
        if room <= 0:
            # The doses alone exceed the budget. Scale everything, and say so loudly:
            # an arm that silently trained on a different amount is an arm that cannot
            # be compared to the others.
            k = budget / max(1.0, guard + replay)
            guard, replay, room = guard * k, replay * k, 0.0
            notes["doses_exceeded_budget"] = True
        k = room / natural
        for r in BASE_ROLES:
            plan[r] = plan.get(r, 0.0) * k
        notes["scale_applied"] = round(k, 6)
    else:
        replay = 0.0
    plan[GUARDRAIL] = guard
    plan[REPLAY] = replay

    total = sum(plan.values()) or 1.0
    got_priority = sum(plan.get(r, 0.0) for r in PRIORITY)
    return {
        "budget_tokens": int(total),
        "budget_requested": int(budget_tokens) if budget_tokens else None,
        "per_role_tokens": {r: int(plan.get(r, 0)) for r in ROLES},
        "per_role_share": {r: round(plan.get(r, 0) / total, 4) for r in ROLES},
        "per_role_repeats": {r: round(plan.get(r, 0) / avail[r], 4) if avail.get(r) else 0.0
                             for r in ROLES},
        "priority_share_achieved": round(got_priority / total, 4),
        "priority_share_floor": min_priority_share,
        "natural_budget_tokens": int(natural),
        "filler_capped_by_corpus": want_filler > filler_avail + 1,
        "available_tokens": {r: int(avail.get(r, 0)) for r in ROLES},
        "doses": {"tool_epochs": tool_epochs, "struct_max_epochs": struct_max_epochs,
                  "guardrail_epochs": guardrail_epochs, "replay_frac": replay_frac},
        "notes": notes,
    }


def _rank(row, salt):
    """A deterministic per-row order that does not depend on file order or dict iteration."""
    key = "%s|%s|%s" % (row.get("c"), row.get("i"), salt)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def index_split(paths, log=None):
    """One streaming pass per source: per-role token totals and per-role (rank, id) lists.

    `paths` is an ordered list. An id is the pair (source index, line number), so the
    guardrail block and any replay set are addressed in the same space as the main split
    and the sampler needs no special case for them.
    """
    if isinstance(paths, str):
        paths = [paths]
    per_role, order = {}, {}
    for src, path in enumerate(paths):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            for ln, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                role = r.get("role") or "unknown"
                per_role[role] = per_role.get(role, 0) + int(r.get("n_tok") or 0)
                order.setdefault(role, []).append((_rank(r, "mix"), (src, ln)))
    for role in order:
        order[role].sort()
    if log:
        log("indexed %d source(s), %d roles: %s"
            % (len(paths), len(per_role), json.dumps(per_role)))
    return per_role, order


def choose(order, per_role, plan, log=None):
    """Row ids to train on, with repeats, honouring the calibrated token plan.

    Rows are taken in a hash order that is stable across runs, so two arms that differ in
    one hyperparameter see the same rows in the same order and the comparison is paired.
    """
    ids, detail = [], {}
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
            taken.extend(i for _, i in rows[:need])
        ids.extend(taken)
        detail[role] = {"rows_taken": len(taken), "rows_available": len(rows),
                        "tokens": int(len(taken) * mean_tok),
                        "mean_tokens_per_row": round(mean_tok, 1)}
    # Interleave by hash so the roles are mixed through the epoch rather than blocked,
    # which matters for a sub-epoch run where a blocked role trains last and dominates.
    ids.sort(key=lambda i: hashlib.blake2b(str(i).encode(), digest_size=8).digest())
    if log:
        log("selected %d training rows: %s" % (len(ids), json.dumps(detail)))
    return ids, detail


def read_lines(paths, wanted):
    """Materialize the selected rows, preserving the requested order and repeats."""
    if isinstance(paths, str):
        paths = [paths]
    need = {}
    for pos, i in enumerate(wanted):
        need.setdefault(i, []).append(pos)
    out = [None] * len(wanted)
    for src, path in enumerate(paths):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            for ln, line in enumerate(fh):
                key = (src, ln)
                if key in need:
                    r = json.loads(line)
                    for pos in need[key]:
                        out[pos] = r
    return [r for r in out if r is not None]
