# tidepool

Working directory for the Primus research project behind the LFM2.5 work in this
repository. Everything here is mirrored out of the project as it lands, so the
state below is the state of the run, not a summary written afterwards.

Stage 4 (data preparation) is complete: the corpus is rendered and measured at 518,330
supervised examples and 201.5M trainable tokens, plus a 447,053-prompt pool, and the two
hand-written probe sets are built and proven held out. Stage 5 (experimentation) is next.
Nothing has drawn on the GPU allowance yet, since every job so far ran on CPU.

| Path | What it is |
| --- | --- |
| `overview.md` | Problem statement, datasets, splits, evaluation, success criteria, constraints, risks, out of scope. |
| `tasks.md` | The checklist the run works from. Ticks here are the record of what is actually done. |
| `initial-prompt.md` | The verbatim prompt the project was created from. |
| `budget.json` | Compute ledger: planned total in GPU-hours, one entry per job with what it actually cost. |
| `experiments.json` | The Transformer Lab experiments this project created. |
| `stages/s0-intake/` | Scoping. |
| `stages/s1-literature-review/` | Literature review report. |
| `stages/s2-research-summary/` | Cited synthesis and the reader brief. |
| `stages/s3-research-plan/` | `plan.md` carries the hypotheses, the experiment matrix and the compute budget. |
| `stages/s4-data-preparation/` | Access and licence findings, exploratory analysis, split design. |
| `stages/s4-data-preparation/eda/diagnostics-<job>/` | Raw per-corpus statistics from the diagnostics job, one file per corpus, plus its figures. |
| `stages/s4-data-preparation/jobs/` | The code each queued job ran, exactly as submitted. |
| `stages/s4-data-preparation/preprocess-4674a2ec/` | Per-corpus rendering counts, the unrenderable-reason breakdown, the purity check and the figure from the preprocessing job. |
| `stages/s4-data-preparation/probes-05bbcd49/` | `probes.jsonl` is the 434 hand-written evaluation items in full, with the build's own summary and score beside it. |

## Reading the numbers

Every figure in a stage report comes from a job that has an id, and the artifacts
that produced it sit under `eda/`. Where a number was corrected after a full-corpus
scan replaced a sample, both versions are kept: the s4 report carries a corrections
table rather than a silently rewritten paragraph.

## The probe sets

`stages/s4-data-preparation/probes-05bbcd49/probes.jsonl` is the one dataset in this
project written here rather than downloaded, so it ships verbatim. 434 items: 290 that
hand a model a broken or quietly-wrong tool return and grade whether it says so instead
of quoting a value the return does not contain, and 144 on this stack's own idioms across
six families and four question framings each. Each item carries the specific wrong answer
it was written to catch.

They are evaluation-only. Every item was checked against the rendered training split on
the same 13-gram rule the corpora were decontaminated with, and none of the 434 shares a
single 13-gram with any of the 494,341 training rows.

## Not mirrored here

The converted full texts of the reviewed papers (roughly 230 of them) and the
per-paper reading notes stay in the project rather than being republished here.
The synthesis that cites them is in `stages/s2-research-summary/`.
