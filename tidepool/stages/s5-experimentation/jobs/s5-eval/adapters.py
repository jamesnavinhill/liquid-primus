"""Turning what shared storage hands back into a directory PEFT will load.

Its own module because it needs a test and `main.py` cannot be imported without a GPU stack.
The three hops are each reasonable alone -- the sweep writes `adapter.zip`, storage returns a
directory holding the archive, PEFT wants the directory the config is in -- and lining them up
wrongly does not crash in a useful place: a harness that caught the load error and carried on
would report a full set of plausible scores for the base model under an arm's name.
"""

import os
import zipfile


def resolve(path, dest):
    if os.path.isfile(path) and path.endswith(".zip"):
        arc = path
    elif os.path.isdir(path):
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            return path
        zips = [f for f in sorted(os.listdir(path)) if f.endswith(".zip")]
        subs = [f for f in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, f))]
        if len(zips) == 1:
            arc = os.path.join(path, zips[0])
        elif not zips and len(subs) == 1:
            return resolve(os.path.join(path, subs[0]), dest)
        else:
            raise RuntimeError("%s holds no adapter_config.json and %d archive(s); an adapter "
                               "that does not load would score the base model under an arm's "
                               "name" % (path, len(zips)))
    else:
        raise RuntimeError("the adapter path %s is neither an archive nor a directory" % path)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(arc) as zf:
        zf.extractall(dest)
    if not os.path.exists(os.path.join(dest, "adapter_config.json")):
        inner = [os.path.join(dest, f) for f in sorted(os.listdir(dest))
                 if os.path.isdir(os.path.join(dest, f))]
        if len(inner) == 1 and os.path.exists(os.path.join(inner[0], "adapter_config.json")):
            return inner[0]
        raise RuntimeError("%s unpacked without an adapter_config.json, so nothing in it is "
                           "a LoRA adapter" % arc)
    return dest
