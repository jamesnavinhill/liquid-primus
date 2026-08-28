"""The serving path has to load before an arm is allowed to serve anything.

`s5-llama-build` copies `llama-server` out of its build tree without the CUDA runtime it links
against. Whether that matters is a property of the *host*, not of the build: the L40S image that
scored the s5.2 4-bit rows carried `libnccl.so.2` on the default loader path and the L4 image
that ran the first s5.6 quality pass did not, so all six arms of `af80ef62` and `f7ebf0a3` died
at startup with the weights already staged.

Two things are tested here, and the second one is why this file exists rather than a comment.
The repair itself is worth a test because its three moves have to happen in order and the last
one has to *raise* -- a repair that quietly gave up would leave an arm serving nothing while the
pack summary counted it. And `load_gguf` is scanned for unbound names because the same first
pass that lost six arms to the loader was also carrying a `NameError` in the facts dict, one
line past the point where the server becomes healthy: fixing only the visible fault would have
bought a second dead card. A whole-function scan is the check that catches that class, not the
instance.
"""

import ast
import os
import unittest

import gen_gguf


def quiet(*a, **k):
    pass


NCCL = "llama-server: error while loading shared libraries: libnccl.so.2: cannot open shared object file"


class RepairLoaderPath(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.get("LD_LIBRARY_PATH")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = self._saved

    def test_a_binary_that_already_loads_is_left_alone(self):
        calls = []

        def run(cmd):
            calls.append(cmd)
            return 0, "version: 10622"

        self.assertEqual(gen_gguf.repair_loader_path("/bin/llama-server", log=quiet, run=run),
                         "none needed")
        self.assertEqual(len(calls), 1, "a loadable binary should be probed once and not repaired")

    def test_the_wheel_directories_are_tried_before_installing_anything(self):
        seen = {"n": 0}

        def run(cmd):
            seen["n"] += 1
            return (127, NCCL) if seen["n"] == 1 else (0, "version: 10622")

        def pip():
            raise AssertionError("pip must not run when the environment already has the libraries")

        self.assertEqual(
            gen_gguf.repair_loader_path("/bin/llama-server", log=quiet, run=run, pip=pip),
            "site-packages CUDA wheels")

    def test_the_wheel_is_the_second_move_and_is_reported(self):
        seen = {"n": 0}

        def run(cmd):
            seen["n"] += 1
            return (0, "version") if seen["n"] == 3 else (127, NCCL)

        self.assertEqual(
            gen_gguf.repair_loader_path("/bin/llama-server", log=quiet, run=run, pip=lambda: 0),
            "nvidia-nccl-cu12 wheel (pip rc=0)")

    def test_an_unrepairable_host_raises_before_anything_is_served(self):
        with self.assertRaises(RuntimeError) as e:
            gen_gguf.repair_loader_path("/bin/llama-server", log=quiet,
                                        run=lambda cmd: (127, NCCL), pip=lambda: 1)
        self.assertIn("libnccl.so.2", str(e.exception))
        self.assertIn("Nothing has been served", str(e.exception))

    def test_a_non_loader_failure_is_not_read_as_a_missing_library(self):
        # `--version` exiting non-zero because the tool dislikes its arguments says nothing about
        # linking. Treating it as a loader fault would install a wheel on every healthy host.
        self.assertEqual(
            gen_gguf.repair_loader_path("/bin/llama-server", log=quiet,
                                        run=lambda cmd: (1, "unknown argument: --version")),
            "none needed")

    def test_the_repair_prefers_directories_that_travel_with_the_archive(self):
        dirs = gen_gguf._nvidia_lib_dirs(root="/nonexistent-serving-path")
        self.assertNotIn("/nonexistent-serving-path/lib", dirs,
                         "a directory that does not exist must not go on the loader path")
        self.assertEqual(len(dirs), len(set(dirs)), "the loader path must not repeat directories")


class LoadGgufHasNoUnboundNames(unittest.TestCase):
    """The `size_mb` regression: bound in `resolve_gguf`, read in `load_gguf`'s facts dict."""

    def test_every_name_load_gguf_reads_is_bound_in_it(self):
        import builtins

        with open(gen_gguf.__file__) as fh:
            tree = ast.parse(fh.read())
        module_level = {n.name for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                module_level |= {(a.asname or a.name).split(".")[0] for a in n.names}

        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "load_gguf")
        bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)

        read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)}
        unbound = sorted(read - bound - module_level - set(dir(builtins)))
        self.assertEqual(unbound, [], "load_gguf reads names nothing binds: %s" % unbound)


if __name__ == "__main__":
    unittest.main()
