# Third-party and copied files in this task

The graders for the two public benchmarks are the benchmarks' own code, vendored rather than
reimplemented. A reimplementation would make our numbers incomparable to the published ones for
reasons no reader could audit.

| File | Origin | Licence | Change |
| --- | --- | --- | --- |
| `ifstruct_validator.py` | `Liquid4All/ifstruct`, `validator.py` | as published in that repository | none; verbatim |
| `ife_instructions.py` | `google-research/google-research`, `instruction_following_eval/instructions.py` | Apache 2.0 (header retained) | imports flattened to this directory |
| `ife_instructions_registry.py` | same, `instructions_registry.py` | Apache 2.0 (header retained) | imports flattened |
| `ife_instructions_util.py` | same, `instructions_util.py` | Apache 2.0 (header retained) | imports flattened |

`bank_tools.py`, `bank_stack.py` and `build.py` are copied unchanged from this project's own
`s4-probes` task, so the clean control arm is constructed by byte-identical code to the graded
probe items it is compared against. They are ours, not third-party.

`bfcl.py` is our own abstract-syntax checker over the public BFCLv3 data. The upstream checker
is entangled with a harness that executes calls against live APIs, so it is not runnable here;
ours is verified instead by scoring a ground-truth oracle, which it returns 1.0000 on across 858
items, with three deliberate degradations at 0.000. The verification is in
`../../runs/s5.2-baselines.md`.
