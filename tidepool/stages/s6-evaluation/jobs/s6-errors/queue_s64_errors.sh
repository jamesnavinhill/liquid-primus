#!/usr/bin/env bash
# s6.4 -- error analysis. CPU only; no GPU-hours.
#
# Three questions s6 has been carrying as readings rather than as measurements, each with a
# recorded number attached to it that nobody has yet been able to point at an item for.
#
# 1. THE GRADER EDIT. `runs/s6.1-baseline-replay.md` records B1 and B3 both moving by -0.0051
#    of native tool-calling composite when the grader was corrected, and explains the
#    coincidence as one item flipping in a small category: the composite is a macro average
#    over 11 categories, so a single flip in a category of n items moves it by exactly
#    1/(11n), and 1/(11*18) = 0.00505. B3's pre-edit verdicts survive as artifacts and are
#    staged; B1's were lost to the typed-save bug. So B3 is an OBSERVED result -- name the
#    item, show the reason string change, and check the recomputed delta against the flip
#    arithmetic to nine decimals -- and B1 is a PREDICTION carried across from B3's flips and
#    checked against its recorded 0.6700. The residual is the whole content of the B1 row.
#
# 2. THE JSON-ONLY STRUCTURED-OUTPUT REGRESSION. R3 loses JSON validity against B1 (0.1310
#    against 0.2210) while YAML barely moves. Reading the grader messages already in the
#    scored files, R3's JSON failures are dominated by "Unclosed code block" (432 against
#    B1's 160) and "Response must use a code block but none was found" (332 against 120)
#    while "required field missing" FALLS, 400 to 144. Two failure modes wear the same
#    number: a model that stopped understanding the schema, and one that understands it and
#    cannot close a fence. `hit_max_new_tokens = 55` over R3's whole run rules out the
#    768-token cap as the cause. The job splits every failure into envelope-only and
#    schema-any -- envelope-only meaning EVERY error on that item is a fence error -- reads
#    the raw completion back for the fence structure, and reports the envelope share of
#    failures per format per arm. Nothing above is citable until this runs.
#
# 3. WHAT THE LOST CALLS LOOK LIKE. s6.2 established that the quantized arms' flat native
#    cell is a cancellation: -0.086 over the eight categories where a call is warranted,
#    +0.159 over the three where abstaining is right. The pooled number cannot say whether a
#    lost item emitted the wrong call, emitted prose, or emitted nothing at all, and the
#    three have different fixes. Restricted to the call-warranted half, this tabulates every
#    item the arm lost against the reference by grader reason family and by what the model
#    actually put on the wire, and quotes twelve.
#
# Seven arms with staged completions, each verified against the run whose verdicts are in
# storage by byte-comparing that job's eval_summary.json to the promoted one -- the packed
# llama.cpp jobs wrote empty completions and were discarded in favour of the solo runs.
# B3 rides along for the grader-delta component only and has no completions staged; the
# structured-output and loss components skip an arm whose completions are absent rather
# than failing the pass.

# SECOND PASS. The first (job 61131f2a) ran the whole tabulation and its own arithmetic check
# caught a defect in it: keyed on the bare item id, keeping the first of any repeat. BFCL v3
# ships exactly one repeat, two different `live_relevance` questions both carrying the id
# `live_relevance_3-3-0`, and the single item the grader edit reclassified is the SECOND
# occurrence. So the first pass reported zero flips against a composite that had moved by
# -0.005051, and said so as an assertion failure rather than reconciling it. The comparison
# job at `s5.3` has paired by occurrence since it was written and its numbers were never at
# risk; this job now uses the same `<id>#k` convention, asserts both files list their ids in
# the same order, and carries the parsed-call count on either side of each flip. 46 checks green.
# This artifact SUPERSEDES 61131f2a's, which stays in the ledger as the pass that found the bug.

# THIRD PASS. Job 47cbb480 landed clean -- zero assertion failures, the B3 flip named and its
# arithmetic reconciling to -0.005051, the B1 prediction leaving a residual of -4.6e-05 inside
# the rounding of a four-decimal recorded figure -- and reading its own tables exposed two
# places where the job stopped one step short of the claim it was built to support.
#
# The reason table degenerated. The grader writes one clause per ground-truth call, prefixed
# `gt<i>/call<j>:`, so `wrong function: total_revenue` and `wrong function: list_orders` became
# two different families and the call-loss table came back as roughly forty singleton rows that
# name functions and say nothing about failure modes. The prefix is now stripped and the first
# clause decides the family, which is what makes the table answer the question in component 3.
#
# The structured-output claim was still mine rather than the job's. Raw validity mixes a model
# that lost the schema with a model that kept it and did not close a fence, and separating them
# was the entire point of component 2 -- but the job reported the envelope share of failures and
# left the conditional to be worked out by hand from two other columns. Each format bucket now
# carries `clean_envelope_items` and `validity_given_clean_envelope` directly, so the sentence
# "an arm whose raw validity falls while the conditional holds lost a wrapper, not a schema"
# has a job id under it. 48 checks green. This artifact SUPERSEDES 47cbb480's.

set -e
lab task queue 140e3fab-172c-456b-975f-82f80bfbf496 -e tidepool --no-interactive \
  --provider "${PROVIDER:-aws}" \
  -p arms=B1,R3,C7,R3-F16,R3-Q4_K_M,C7-F16,C7-Q4_K_M,B3 \
  -p reference=B1 \
  -p scored_prefix=tidepool/s5.3/arms \
  -p arm_prefixes='{"R3-F16":"tidepool/s5.6/arms","R3-Q4_K_M":"tidepool/s5.6/arms","C7-F16":"tidepool/s5.6/arms","C7-Q4_K_M":"tidepool/s5.6/arms","B3":"tidepool/s6.1/baselines"}' \
  -p completions_prefix=tidepool/s6.4/completions \
  -p pre_edit_objects='{"B3":"tidepool/s6.4/pre_edit/B3/scored_bfcl_native_tools.jsonl"}' \
  -p recorded_pre_edit_composite='{"B1":0.6700}' \
  -p top_errors=20 \
  -p examples=12 \
  -m 'Error analysis for s6.4: name the items the grader correction reclassified and check the
flip arithmetic against both the recomputed and the recorded composite; split every structured-
output failure into a formatting-envelope failure and a schema failure, per format per arm, and
read the raw completions back for the fence structure behind them; and tabulate what the model
emitted on the call-warranted items each arm lost against the base, by grader reason family.'
