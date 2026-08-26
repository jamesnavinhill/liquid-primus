# Operator notes

Standing preferences from the people running this project. They are not
suggestions: follow them unless an explicit instruction in this run says
otherwise, and when one shaped a decision, say so in the stage report.

This file is generated before every run and is not writable — editing it
changes nothing. To record a NEW preference an operator states, append to
`.notes` instead; see the primus skill, "Operator notes".

Each note says where it came from. "Written by" means an operator typed it
into the project themselves; "recorded by Primus from" means it was taken
from something they said while working. That is provenance and nothing more
— the two carry EQUAL weight, and a typed note does not outrank a recorded
one. Use only the rule below to decide between them.

Where two notes pull in different directions, the LATER one wins, and a
"This project" note outranks an "All projects" note regardless of date.

## All projects

- Wants visible insight into runs: metrics, logs, and any dashboard or UI they can open themselves. Surface where to look rather than only summarizing in prose.  _(recorded by Primus from james@jami.studio, 2026-08-25)_
- A concurrency cap on GPU instances is not a cap on parallel work. Pack each GPU with as many runs as fit and engineer the isolation so one run failing, stopping or being cancelled does not disturb the others; do not treat an instance limit as a reason to run one job per card or to serialize work.  _(recorded by Primus from james@jami.studio, 2026-08-25)_
- Prefers being told when a constraint is being worked around rather than accepted. If a limit looks binding, check whether it is actually binding before planning around it, and say plainly when an earlier framing was wrong.  _(recorded by Primus from james@jami.studio, 2026-08-25)_

## This project

- Mirror project artifacts to the operator's own repos as they land: code and docs to github.com/jamesnavinhill/liquid-primus, weights and quantized builds to huggingface.co/jamesnavinhill/liquid-primus. GitHub, Hugging Face and Weights & Biases credentials are provided for this.  _(recorded by Primus from james@jami.studio, 2026-08-25)_
- Size the first sweep to the compute that is actually available, and say what a larger version would cost before spending past what has been approved.  _(recorded by Primus from james@jami.studio, 2026-08-25)_
- Specialize models to the underlying technology (SQL dialects, JSON schemas, MCP, IPC, containers, tracing semantics) rather than to one vendor's SDK idiosyncrasies, so the fleet stays useful as the stack shuffles. Stated as a deliberate owner decision, not a suggestion.  _(recorded by Primus from james@jami.studio, 2026-08-25)_
