# tidepool

Working directory for the Primus research project behind the LFM2.5 work in this
repository. Everything here is mirrored out of the project as it lands, so the
state below is the state of the run, not a summary written afterwards.

Stage 4 (data preparation) is in progress. Stages 0 through 3 are complete.

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

## Reading the numbers

Every figure in a stage report comes from a job that has an id, and the artifacts
that produced it sit under `eda/`. Where a number was corrected after a full-corpus
scan replaced a sample, both versions are kept: the s4 report carries a corrections
table rather than a silently rewritten paragraph.

## Not mirrored here

The converted full texts of the reviewed papers (roughly 230 of them) and the
per-paper reading notes stay in the project rather than being republished here.
The synthesis that cites them is in `stages/s2-research-summary/`.
