"""Turning what shared storage hands back into a directory the harness can load.

Two shapes arrive. A LoRA arm writes an `adapter_config.json` next to its `adapter_model`
tensors and is merged onto the base model. A full-parameter arm writes an ordinary
`config.json` and `model.safetensors`, because tuning every weight leaves nothing to adapt --
so it is not an adapter at all, it *is* the model, and the caller has to load it as one. The
distinction is made here, by looking, rather than by a parameter the queue command has to get
right: an arm mislabelled either way loads without complaint and scores the wrong weights.

Its own module because it needs a test and `main.py` cannot be imported without a GPU stack.
The three hops are each reasonable alone -- the sweep writes `adapter.zip`, storage returns a
directory holding the archive, PEFT wants the directory the config is in -- and lining them up
wrongly does not crash in a useful place: a harness that caught the load error and carried on
would report a full set of plausible scores for the base model under an arm's name.
"""

import os
import zipfile

ADAPTER = "adapter"
FULL = "full"


def kind(d):
    """Which of the two shapes this directory is, or None if it is neither."""
    if os.path.exists(os.path.join(d, "adapter_config.json")):
        return ADAPTER
    if os.path.exists(os.path.join(d, "config.json")):
        try:
            names = os.listdir(d)
        except OSError:
            return None
        if any(n.startswith(("model", "pytorch_model"))
               and n.endswith((".safetensors", ".bin")) for n in names):
            return FULL
    return None


def resolve(path, dest):
    if os.path.isfile(path) and path.endswith(".zip"):
        arc = path
    elif os.path.isdir(path):
        if kind(path):
            return path
        zips = [f for f in sorted(os.listdir(path)) if f.endswith(".zip")]
        subs = [f for f in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, f))]
        if len(zips) == 1:
            arc = os.path.join(path, zips[0])
        elif not zips and len(subs) == 1:
            return resolve(os.path.join(path, subs[0]), dest)
        else:
            raise RuntimeError("%s holds no loadable checkpoint and %d archive(s); weights "
                               "that do not load would score the base model under an arm's "
                               "name" % (path, len(zips)))
    else:
        raise RuntimeError("the adapter path %s is neither an archive nor a directory" % path)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(arc) as zf:
        zf.extractall(dest)
    if not kind(dest):
        inner = [os.path.join(dest, f) for f in sorted(os.listdir(dest))
                 if os.path.isdir(os.path.join(dest, f))]
        if len(inner) == 1 and kind(inner[0]):
            return inner[0]
        raise RuntimeError("%s unpacked without an adapter_config.json or a model config, so "
                           "nothing in it is loadable weights" % arc)
    return dest
