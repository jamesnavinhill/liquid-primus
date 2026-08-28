"""The 4-bit path can serve a published quantization or one this project made. Not both.

s5.6 exports our own GGUF into shared storage, so `gen_gguf.load_gguf` had to grow a second
source. The risk the tests below are written against is not a crash: a row that resolves the
wrong file still serves, still scores, and reports a retention number attributed to weights it
never loaded. So the mutually-exclusive case is an error rather than a precedence rule, and an
ambiguous directory is an error rather than a first match.
"""

import os
import tempfile
import unittest

import gen_gguf


def quiet(*a, **k):
    pass


class ResolveGguf(unittest.TestCase):

    def test_storage_object_resolving_to_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "R3-Q4_K_M.gguf")
            open(f, "wb").write(b"\0" * 2048)
            path, repo, name, src = gen_gguf.resolve_gguf(
                {"gguf_object": "tidepool/s5.6/R3/R3-Q4_K_M.gguf"},
                storage=lambda o: f, out_dir=d, log=quiet)
        self.assertEqual(path, f)
        self.assertEqual(src, "storage")
        self.assertEqual(name, "R3-Q4_K_M.gguf")
        self.assertEqual(repo, "tidepool/s5.6/R3/R3-Q4_K_M.gguf")

    def test_storage_object_resolving_to_a_directory_with_one_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            inner = os.path.join(d, "held")
            os.makedirs(inner)
            f = os.path.join(inner, "R3-Q4_0.gguf")
            open(f, "wb").write(b"\0" * 16)
            path, _, name, src = gen_gguf.resolve_gguf(
                {"gguf_object": "tidepool/s5.6/R3/"}, storage=lambda o: d, out_dir=d, log=quiet)
        self.assertEqual(path, f)
        self.assertEqual(name, "R3-Q4_0.gguf")
        self.assertEqual(src, "storage")

    def test_a_directory_holding_two_ggufs_is_an_error_not_a_first_match(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("a.gguf", "b.gguf"):
                open(os.path.join(d, n), "wb").write(b"\0")
            with self.assertRaises(RuntimeError) as e:
                gen_gguf.resolve_gguf({"gguf_object": "tidepool/s5.6/R3/"},
                                      storage=lambda o: d, out_dir=d, log=quiet)
        self.assertIn("name the file", str(e.exception))

    def test_both_sources_set_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            gen_gguf.resolve_gguf(
                {"gguf_object": "tidepool/s5.6/R3/R3-Q4_0.gguf",
                 "gguf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                 "gguf_file": "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"},
                storage=lambda o: o, out_dir=".", log=quiet)
        self.assertIn("both set", str(e.exception))

    def test_neither_source_set_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            gen_gguf.resolve_gguf({}, storage=lambda o: o, out_dir=".", log=quiet)
        self.assertIn("gguf_object", str(e.exception))

    def test_the_hub_path_is_unchanged_and_still_names_the_file_it_wants(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "pub.gguf")
            open(f, "wb").write(b"\0" * 8)
            calls = []

            def hub(repo_id, filename, local_dir):
                calls.append((repo_id, filename, local_dir))
                return f

            path, repo, name, src = gen_gguf.resolve_gguf(
                {"gguf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                 "gguf_file": "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"},
                storage=lambda o: o, out_dir=d, log=quiet, hub=hub)
        self.assertEqual(src, "hub")
        self.assertEqual(path, f)
        self.assertEqual(repo, "LiquidAI/LFM2.5-1.2B-Instruct-GGUF")
        self.assertEqual(len(calls), 1)

    def test_a_repo_without_a_named_file_lists_what_it_publishes(self):
        with self.assertRaises(RuntimeError) as e:
            gen_gguf.resolve_gguf(
                {"gguf_repo": "some/repo"}, storage=lambda o: o, out_dir=".", log=quiet,
                list_files=lambda r: ["a.gguf", "readme.md", "b.gguf"])
        self.assertIn("a.gguf", str(e.exception))
        self.assertIn("b.gguf", str(e.exception))
        self.assertNotIn("readme.md", str(e.exception))


if __name__ == "__main__":
    unittest.main()
