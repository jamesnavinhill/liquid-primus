import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")).read()
start = src.index("def _ids(enc):"); end = src.index("def row_tokens(msgs):")
ns = {}
exec(src[start:end], ns)
_ids = ns["_ids"]

class BatchEncoding(dict):
    # mirrors HF: a dict subclass that also exposes .input_ids
    def __init__(self, d): super().__init__(d)
    @property
    def input_ids(self): return self["input_ids"]

IDS = [1, 2, 3, 4, 5, 6, 7]
cases = [
    ("plain list of ids",            IDS,                                              7),
    ("BatchEncoding (2 keys)",       BatchEncoding({"input_ids": IDS, "attention_mask": [1]*7}), 7),
    ("plain dict",                   {"input_ids": IDS, "attention_mask": [1]*7},       7),
    ("batched list-of-lists",        [IDS],                                            7),
    ("BatchEncoding, batched",       BatchEncoding({"input_ids": [IDS], "attention_mask": [[1]*7]}), 7),
    ("empty list",                   [],                                               0),
    ("tuple of ids",                 tuple(IDS),                                       7),
]
bad = 0
for name, enc, want in cases:
    got = len(_ids(enc))
    ok = got == want
    bad += not ok
    print(("  ok  " if ok else "  FAIL"), f"{name:28s} -> {got} (want {want})")

# the exact regression: len() straight on a BatchEncoding is the old, wrong answer
old = len(BatchEncoding({"input_ids": IDS, "attention_mask": [1]*7}))
print(("  ok  " if old == 2 else "  FAIL"), f"{'regression: bare len() == 2':28s} -> {old}")
bad += old != 2
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
