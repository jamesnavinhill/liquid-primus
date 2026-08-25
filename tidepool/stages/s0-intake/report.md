# Stage 0: Intake & scoping

_Status: complete — s0.1 and s0.2 done._

## Headline

Tidepool is scoped to its first tier: make LiquidAI's 1.2B on-device model call tools
reliably enough for the operator's agent gateway, and keep that gain after 4-bit
compression. The pre-registered bar is BFCLv3 at or above 52.43, the tier's current
best, and at least 3 points above the 49.12 starting point, with instruction-following
and general knowledge held within 2 points and the shipped 4-bit build at 1.5 GB or less.

## Initial prompt (verbatim)

> hello my friend. Ive put together a thorough and thoughtful plan. It is detailed in the [readme.md](http://readme.md) --the other docs are supporting. I tried not to get into the details too much on how we get to our goals, but tried to provide some general guidance and starting points. I have also included some creds for github, huggingface, w&b, theyre attached thru the settings panel. id love for all the work to get sent to the proper repos as it lands but i totally understand if that's not possible and ill do that manually. but here they are in case :) Id also love any form of insights on the run and metrics, logs etc. whatever ui i can access is wonderful. im open to your thoughts and suggestions.
>
> <https://github.com/jamesnavinhill/liquid-primus>
>
> <https://huggingface.co/jamesnavinhill/liquid-primus>

## Work log

- 2026-08-25 · s0.1 · Captured the verbatim prompt → `initial-prompt.md`; scaffolded the
  project skeleton and `tasks.md`; named the project `tidepool`.
- 2026-08-25 · s0.2 · Read all ten planning documents and both attached papers' framing,
  then wrote the provisional `overview.md`: 20 assumption rows, 5 open questions, and a
  pre-registered claim on the tier-1 tool-calling axis.

## Attached material

`documents/liquid-primus/` (listed in `attached-folders.json`) carries the operator's
plan and its supporting docs, all dated 2026-08-25:

| File | What it carries |
| --- | --- |
| `readme.md` | The goal and directive: the target family, the baseline to beat, hard constraints, and the priority order when they conflict. |
| `models.md` | The LFM2.5 family with verified Hugging Face IDs, per-model architecture from `config.json`, first-party recipes, and the license terms. |
| `benchmarks.md` | The published baseline table per size tier — vendor and community incumbents — with the named weakness to close in each tier. |
| `datasets.md` | Verified training and eval datasets with licenses, sizes, and the capability axis each one teaches. |
| `moe.md` | The hybrid mixture-of-experts goal, three verified findings about what is already shipped, and four ranked approach options. |
| `fine-tune.md` | What "specialized to our stack" has to mean: six task classes, the outcome spec, and the fine-tuning menu. |
| `quants.md` | Quantization options and the quality bar for shipped 4-bit builds. |
| `sources.md` | Every load-bearing claim traced to a primary source, fetched 2026-08-25. |
| `outline.md` | The raw working-notes record. Superseded by the topic docs; kept as history, and explicitly not to be cited. |
| `2607.15232v1.pdf`, `2608.00017v1.pdf` | Primary papers: in-place tokenizer expansion, and memory reward inflation in self-improving agents. |

Two things in that folder are read as instructions to us, and both are followed:
`outline.md` says its own claims are superseded by the topic docs wherever they conflict,
and `models.md` scopes the improvement work to LFM2.5 checkpoints, with the competitor
models used as baselines only.

## Outputs

- `initial-prompt.md`
- `tasks.md`
- `stages/s0-intake/report.md` (this file)

## Provisional scope (s0.2)

`overview.md` is written and carries `Status: provisional (pre-literature)`. The reasoning
behind the calls that were not simply read off the operator's documents:

**The one big call: one tier, not four.** The operator's directive asks for improved
checkpoints across four device tiers, a vision-language lane, a mixture-of-experts family,
and a quantization lane. Executing all of that at once would produce a shallow result on
every axis. The directive itself resolves this: it says to start in the smallest tier where
a win is cheapest to demonstrate, prove it holds, then scale up, and to name a specific
weakness. So the project's primary lane is tier 1 (`LFM2.5-1.2B-Instruct`, under 1.5 GB at
4-bit) and the weakness is tool-calling reliability, which is also first in the operator's
stated priority order. Sequencing rather than dropping: the boundary and the promotion
candidates are written into "Out of scope", so an operator who wants a wider project can
push back at the scope sign-off with the cost visible.

**Why the mixture-of-experts work sits behind rather than beside.** The operator's own
architecture note establishes that the composable core is the shared 128K-vocabulary,
2048-hidden fabric of the 2.6B, the 8B-A1B, and the VL-3B, and that the 1.2B is misaligned
with it (65K vocabulary, different rotary base, different special-token ids). A tier-1 result
therefore cannot be an input to the mixture-of-experts lane without embedding surgery that is
itself out of scope. Running both at once would mean two unrelated projects sharing a budget.

**Where the numbers came from, and which are soft.** The baseline figures are the operator's
verified transcriptions, and the claim is pinned to the two that matter: the starting point's
49.12 and the tier leader's 52.43. Every comparison is specified as a matched rerun in this
project's own harness rather than a card quote, because the operator's benchmark document
says so explicitly and because the vendor harnesses are unstated. The tolerances underneath
the claim are this project's defaults and are the low-confidence part of the scope: the
reliability probe's 0.80 flag rate with a 0.10 false-flag ceiling rests on nothing external
yet, and so does the 5% throughput floor. Both are flagged `low` in the assumptions table and
both are targets for the literature review to either ground or replace before anything is
frozen.

**Two metrics had to be invented because no public benchmark covers them.** The operator's
top-priority behaviour is that a model flags rather than asserts when a tool returns garbage,
and no published suite scores that. It becomes a held-out probe of roughly 300 items, scored
as a flag-rate and false-flag-rate pair so that neither number can be gamed alone. The second
is the stack-idiom probe, anchored by HumanEval and MBPP so a claim about the operator's stack
does not rest only on a yardstick built here.

**Standing preferences that shaped the file.** Specialization aims at underlying technologies
(SQL dialects, JSON Schema, MCP, inter-process communication, containers, tracing semantics)
rather than one vendor's SDK, which is a stated owner decision and governs both the training
mix and the in-house probe. The compute budget is deliberately left open here, because the
operator asked for the first sweep to be sized to what is actually available with the cost of
a larger version stated before overspending; that number is derived and put in front of them
with the experiment matrix attached. Artifact mirroring to the operator's own repositories is
recorded as a deliverable and acted on once artifacts exist.

**What was deliberately not escalated.** Scope breadth is the one genuine ambiguity in the
ask, and the directive settles it, so it is recorded as an assumption rather than raised as a
question. The in-house corpus and the gated dataset are real external dependencies, and both
are recorded as open questions with a working answer, to be raised at data preparation where
the answer actually changes what runs.

## Outputs

- `initial-prompt.md`
- `overview.md` (provisional)
- `tasks.md`
- `stages/s0-intake/report.md` (this file)

## Next steps

The literature review runs next with no stop in between. Its queries cover the named weakness
(tool-calling reliability in small models), the recipes on the table (supervised fine-tuning,
preference optimization, verifiable-reward reinforcement learning, distillation), the
quantization question (whether 4-bit damage lands disproportionately on structured output),
and every low-confidence assumption above. The first checkpoint is the scope sign-off once the
papers are in.
