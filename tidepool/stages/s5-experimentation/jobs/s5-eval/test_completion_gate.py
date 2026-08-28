"""A component's item-count assertions (`ifstruct_count == 2000`, etc.) check how many
prompts went in, which says nothing about what came back. Attempt 2 of s5.6's GGUF quality
pack (job 46bd54cb, 2026-08-28) showed why that gap matters: three llama-servers packed onto
one L4 collapsed generation throughput under compute contention, individual requests to the
two busiest arms timed out, and `gen_gguf.py` turned every timeout into a silent empty-string
completion. Two of three arms finished with `completion_status: success`, correct item
counts, and zero real completions on three of four components -- a corrupted run that looked,
on every assertion that existed, like a clean one.

`check_completions` lives in `asserts.py`, split out of `main.py` precisely so it can be
tested without pulling in `gen.py`'s hard `import torch` (main.py -> gen -> torch, which
this shell does not have). The test needs no job API and no network.
"""

import unittest

import asserts


class CheckCompletions(unittest.TestCase):

    def setUp(self):
        asserts.ASSERTS[:] = []

    def test_all_real_completions_pass(self):
        ok = asserts.check_completions("ifstruct", ["a real answer"] * 2000)
        self.assertTrue(ok)
        self.assertTrue(asserts.ASSERTS[-1]["ok"])

    def test_a_few_genuine_empties_stay_under_the_line(self):
        # 1% empty, under the 2% ceiling -- a couple of genuinely empty model outputs should
        # not fail a healthy run.
        outs = ["x"] * 990 + [""] * 10
        self.assertTrue(asserts.check_completions("probes", outs))

    def test_the_attempt_2_pattern_fails_loudly(self):
        # R3-F16's actual shape: 100% empty across ifstruct/probes/bfcl_native_tools.
        ok = asserts.check_completions("ifstruct", [""] * 2000)
        self.assertFalse(ok)
        detail = asserts.ASSERTS[-1]["detail"]
        self.assertIn("2000/2000", detail)

    def test_a_partial_stall_still_fails(self):
        # R3-F16's bfcl_tools_text shape: 2543/3490 empty (~73%), not 100% -- the gate must
        # not be tuned to only catch total failure.
        outs = ["x"] * 947 + [""] * 2543
        self.assertFalse(asserts.check_completions("bfcl_tools_text", outs))

    def test_whitespace_only_counts_as_empty(self):
        # gen_gguf.py's _one() already .strip()s, but a completion that is all whitespace
        # from some other path should not be miscounted as content.
        self.assertFalse(asserts.check_completions("ifeval", ["   \n"] * 541))

    def test_empty_output_list_does_not_divide_by_zero(self):
        self.assertTrue(asserts.check_completions("probes", []))

    def test_records_into_asserts_like_a_count_check(self):
        asserts.check_completions("ifeval", [""] * 541)
        names = [a["name"] for a in asserts.ASSERTS]
        self.assertIn("ifeval_completions_nonempty", names)


if __name__ == "__main__":
    unittest.main()
