"""Chạy chính thức: 8 seed × các cấu hình. Tự tiếp tục nếu đã có tệp kết quả."""
import json, os, sys, time, numpy as np
from train import *
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "run_001_main"); os.makedirs(OUT, exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D"))
W = match_widths(tgt)
mh_ptr = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_ptr")) - tgt))
JOBS = [("A", "A", W["A"][0]), ("B", "B", W["B"][0]), ("C", "C", W["C"][0]), ("D", "D", 32), ("D", "D_ptr", mh_ptr),
        ("D", "D_oracle", 32), ("D", "D_fixed", 32), ("D", "D_shuffle", 32), ("D", "D_T1", 32),
        ("D", "D_ring", 32), ("D", "D_rand50", 32), ("D", "D_unshared", 32)]
LR = 1e-2
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
meta = dict(target_params=tgt, widths=W, mh_ptr=mh_ptr, lr=LR, epochs=40, batch=200)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
for seed in range(8):
    data = TK.build(seed)
    for kind, name, w in JOBS:
        f = os.path.join(OUT, f"seed{seed}_{name}.json")
        if os.path.exists(f): continue
        R = run(kind, name, seed, w, LR, epochs=40, bs=200, data=data); R["name"] = name
        r = evaluate(R, full=(name in ("D", "D_ptr"))); r["curve"] = R["curve"]; r["name"] = name
        json.dump(conv(r), open(f, "w"))
        print(seed, name, "ood %.4f test %.4f" % (r["mse_ood"], r["mse_test"]), flush=True)
print("XONG", flush=True)
