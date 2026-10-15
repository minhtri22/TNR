"""Lượt chính thức v2 theo PREREG_v2: seed 8–19, 100 epoch. Tự tiếp tục nếu đã có tệp."""
import json, os, numpy as np
from train import *
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "run_002_main"); os.makedirs(OUT, exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D")); W = match_widths(tgt)
mh_ptr = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_ptr")) - tgt))
# (kind, cấu hình, bề rộng, lịch, tên tệp)
JOBS = [("A", "A", W["A"][0], None, "A"), ("B", "B", W["B"][0], None, "B"), ("C", "C", W["C"][0], None, "C"),
        ("D", "D", 32, None, "D"), ("D", "D", 32, "anneal", "D_anneal"),
        ("D", "D_ptr", mh_ptr, None, "D_ptr"), ("D", "D_ptr", mh_ptr, "anneal", "D_ptr_anneal"),
        ("D", "D_oracle", 32, None, "D_oracle"), ("D", "D_fixed", 32, None, "D_fixed")]
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
json.dump(dict(target_params=tgt, widths=W, mh_ptr=mh_ptr, lr=1e-2, epochs=100, batch=200), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
for seed in range(8, 20):
    data = TK.build(seed)
    for kind, cname, w, sc, fname in JOBS:
        f = os.path.join(OUT, f"seed{seed}_{fname}.json")
        if os.path.exists(f): continue
        R = run(kind, cname, seed, w, 1e-2, epochs=100, bs=200, data=data, sched=sc); R["name"] = cname
        r = evaluate(R, full=(fname.startswith("D") and fname not in ("D_oracle", "D_fixed"))); r["curve"] = R["curve"]; r["name"] = fname
        json.dump(conv(r), open(f, "w"))
        print(seed, fname, "ood %.4f test %.4f" % (r["mse_ood"], r["mse_test"]), flush=True)
print("XONG", flush=True)
