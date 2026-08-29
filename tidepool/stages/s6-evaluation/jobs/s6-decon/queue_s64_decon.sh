#!/usr/bin/env bash
# s6.4 -- decontamination re-check against the SHIPPED splits. CPU only; no GPU-hours.
#
# `s4.3` did decontaminate. It indexed eleven benchmark corpora at 13-gram, found 799 rows of
# 1,047,820 overlapping, closed its own coverage gap by adding both `allenai/IFBench_multi-turn`
# configs, and dropped every flagged row from every split. None of that is in doubt and this
# job does not re-litigate it. What it measures is the statement a paper actually makes, which
# `s4.3` was not in a position to check, on three counts.
#
# The side. `s4.3` matched benchmark n-grams against training PROMPTS. The shipped rows are
# whole conversations and the assistant turns have never been read. A benchmark answer sitting
# in a training target is the overlap that would actually teach the benchmark, and a prompt-only
# rule cannot see it. Every count here is split by side for that reason.
#
# The items. `s4.3` indexed the benchmark corpora whole. The reported numbers come from 3,489
# specific BFCL ids, 2,000 IFStruct items pulled from a pinned raw URL, and a 434-item probe
# bank we wrote ourselves. BFCL v3 is far larger than the slice this project scores, and an
# overlap against an item no number came from is not contamination of any result. So BFCL is
# restricted to the ids in a scored file, and the job asserts it found them.
#
# The file. `s4.3` measured the PRE-DROP mix and then dropped. Whether the drop worked is a
# property of `train.jsonl.gz`, and nothing has read that file for it. `val` and `test` ride
# along, where a hit would mean something different again: our own held-out slice overlapping
# a public benchmark.
#
# A hit is not automatically a leak, so the job prints the matched span next to the eval item
# it matched. A shared instruction preamble and a shared JSON schema both produce long exact
# overlaps and teach nothing about any answer; that is a judgement a reader makes from the
# text, not one a count can make. 30 fixture checks green.

# SECOND PASS. The first (job 154c72a4) did every scan correctly and wrote both artifacts, then
# died on the line after them: the SDK on that worker has no `job_complete`, so a complete set
# of numbers ended up under a FAILED status. A job's status is its citation, so those numbers
# are not quotable however right they look. The closing call is now guarded and the fixture's
# stub SDK deliberately lacks the method, which fails the tests if it is ever left bare again.
# Six minutes of CPU to buy back a citable id.

# THIRD PASS. da77f60a reproduced the first pass byte for byte under a COMPLETE status, so the
# overlap numbers are settled. What it could not do is hand the next test a sample: it reported
# how many training rows overlap each eval set and not WHICH eval items they overlap. The
# peakedness check that closes this substage is a stratified design -- eval items whose text
# appears in a training target against matched items with no overlap -- and without a named
# item list it degenerates into an unstratified main effect that cannot separate contamination
# from ordinary entropy collapse after fine-tuning. Every scan now names the items it hit, on
# both sides, capped on the input side only. Same six minutes of CPU. 33 checks green.

# FOURTH PASS. Job 95136740 landed clean and its own item lists exposed the reading problem in
# the three passes before it. The headline it produced -- 56,707 of 494,341 shipped training
# rows sharing a 13-gram with a scored eval item, 11.47% -- is not a contamination rate. Of the
# 1.42 million matching windows, 1,417,225 were credited to the probe bank and 219 to BFCL, and
# the probe matches are all one string: the tool-calling instruction template written at s4,
# which sits in the system prompt of every tool-calling training row and in every probe. Two
# copies of our own boilerplate matching each other teach nothing.
#
# Two defects made that hard to see, and both are fixed here.
#
# `Index.freeze` deduplicates grams and keeps ONE owner per gram, so a span two eval sets share
# is credited to whichever was indexed first and every other set's figure is a remainder after
# that. `per_set_scan` re-scans the sets whose numbers this project reports -- BFCL and the
# probes -- against an index containing only themselves, which removes the competition and
# makes those two rows exact. The shared pass stays as the cheap all-sets sweep it was built
# to be.
#
# And the target side was counted without being quoted. Three BFCL items appear in a training
# TARGET, which is the entire contamination finding, and the artifact recorded their ids and
# nothing about what they shared -- the one thing that separates a leaked answer from a shared
# phrase. Every target-side item now carries the matched span and the training row it was found
# in. Three extra CPU passes over the train split, about ten more minutes. 43 checks green.
# This artifact SUPERSEDES 95136740's.

set -e
# New task id: `task upload` refuses task.yaml, and pass 4 adds two parameters that
# have to be DECLARED or the queue line cannot set them and the per-set passes would
# silently default to off -- a pass 4 that looked like pass 4 and was pass 3.
lab task queue c529cf4b-7728-4d93-9e04-c5bc2edd1b93 -e tidepool --no-interactive \
  --provider "${PROVIDER:-aws}" \
  -p train_object=tidepool/s4.4/train.jsonl.gz \
  -p val_object=tidepool/s4.4/val.jsonl.gz \
  -p test_object=tidepool/s4.4/test.jsonl.gz \
  -p bfcl_ids_from=tidepool/s5.3/arms/B1/scored_bfcl_native_tools.jsonl \
  -p probes_object=tidepool/s4.4/probes/probes.jsonl \
  -p eval_sets=ifeval,ifbench_test,ifbench_mt_if,ifbench_mt_ie \
  -p ngram_n=13 \
  -p worst_rows=25 \
  -p examples=15 \
  -p item_report_cap=200 \
  -p per_set_scan=bfcl_scored,probes \
  -m 'Decontamination re-check for s6.4, against the shipped splits rather than the pre-drop mix.
13-gram overlap between train/val/test and the exact eval items behind the reported numbers,
counted separately for the model input and the training target, per corpus and per benchmark,
with the matched span printed beside the item it matched so a boilerplate collision can be told
from a leak. Includes both IFBench multi-turn configs, the coverage gap s4.2 recorded.'
