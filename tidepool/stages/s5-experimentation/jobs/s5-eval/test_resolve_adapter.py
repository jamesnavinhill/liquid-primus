"""The adapter an arm produces must survive the trip to the evaluator.

The sweep writes `adapter.zip`; shared storage hands back a directory holding it; PEFT wants
the directory the config is in. Three hops, each of which is fine on its own, and the failure
mode when they do not line up is not a crash: `PeftModel.from_pretrained` on the wrong path
raises, but a harness that caught and continued would score the base model under the arm's
name. So the unwrapping is a function with a test rather than three lines in main().

Pure CPU over a fixture of a few files; no model, no GPU.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

fails = []


def adapter_dir(root, name="a"):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    json.dump({"peft_type": "LORA", "r": 16}, open(os.path.join(d, "adapter_config.json"), "w"))
    open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"\0" * 64)
    return d


def check(name, ok, detail=""):
    if not ok:
        fails.append("%s: %s" % (name, detail))


def main():
    import adapters
    R = adapters.resolve
    root = tempfile.mkdtemp()
    try:
        # 1. the plain case: a directory that is already the adapter
        d = adapter_dir(root, "plain")
        check("plain", R(d, os.path.join(root, "d1")) == d, "an adapter directory should pass through")

        # 2. what storage actually hands back: a directory holding adapter.zip
        src = adapter_dir(root, "src")
        holder = os.path.join(root, "holder")
        os.makedirs(holder)
        shutil.make_archive(os.path.join(holder, "adapter"), "zip", src)
        out = R(holder, os.path.join(root, "d2"))
        check("zip_in_dir", os.path.exists(os.path.join(out, "adapter_config.json")),
              "unpacked %s without a config" % holder)

        # 3. the archive itself
        out = R(os.path.join(holder, "adapter.zip"), os.path.join(root, "d3"))
        check("zip_file", os.path.exists(os.path.join(out, "adapter_config.json")),
              "unpacking the archive directly should work")

        # 4. one level of nesting, which is what a download of a prefix looks like
        nest = os.path.join(root, "nest")
        os.makedirs(nest)
        shutil.copytree(src, os.path.join(nest, "inner"))
        out = R(nest, os.path.join(root, "d4"))
        check("nested", os.path.exists(os.path.join(out, "adapter_config.json")),
              "a single nested directory should be followed")

        # 5. and the case that matters: something that is not an adapter must raise, not
        #    quietly return a path the loader will ignore.
        empty = os.path.join(root, "empty")
        os.makedirs(empty)
        open(os.path.join(empty, "readme.txt"), "w").write("nothing here")
        try:
            R(empty, os.path.join(root, "d5"))
            fails.append("empty: a directory with no adapter returned a path instead of raising")
        except RuntimeError:
            pass

        # 6. an archive of the wrong thing, likewise
        junk = os.path.join(root, "junk")
        os.makedirs(junk)
        with zipfile.ZipFile(os.path.join(junk, "adapter.zip"), "w") as zf:
            zf.writestr("notes.txt", "not weights")
        try:
            R(junk, os.path.join(root, "d6"))
            fails.append("junk_zip: an archive with no adapter_config.json should raise")
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)


main()
if fails:
    print("FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("adapter resolution holds across 6 cases: directory, zip-in-directory, bare archive, "
      "one level of nesting, and two shapes that must raise")
