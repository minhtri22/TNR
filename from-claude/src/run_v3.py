"""Lượt chính thức v3 theo PREREG_v3: sàng lọc seed bằng oracle (20–49) rồi chạy các nhánh trên 10 seed hợp lệ."""
import json, os, numpy as np
from train import *
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "run_003_main"); os.makedirs(OUT, exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D")); W = match_widths(tgt)
wE = min(range(8, 300), key=lambda w: abs(count_shape("E", w, make_cfg("E")) - tgt))
wA = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_attn")) - tgt))
json.dump(dict(target_params=tgt, widths=W, wE=wE, w_attn=wA, lr=1e-2, epochs=100, batch=200), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
def do(seed, kind, cname, w, fname, data, full=False):
    f = os.path.join(OUT, f"seed{seed}_{fname}.json")
    if os.path.exists(f): return json.load(open(f))
    R = run(kind, cname, seed, w, 1e-2, epochs=100, bs=200, data=data); R["name"] = cname
    r = evaluate(R, full=full); r["curve"] = R["curve"]; r["name"] = fname
    json.dump(conv(r), open(f, "w")); print(seed, fname, "ood %.4f test %.4f train %.5f" % (r["mse_ood"], r["mse_test"], R["curve"][-1]), flush=True)
    return r
valid, screened = [], []
for s in range(20, 50):
    r = do(s, "D", "D_oracle", 32, "D_oracle", TK.build(s))
    ok = (r["final_train_loss"] < 1e-3) and (r["mse_ood"] < 0.5 * r["trivial_ood"])
    screened.append(dict(seed=s, valid=bool(ok), ood=r["mse_ood"], train=r["final_train_loss"]))
    if ok: valid.append(s)
    json.dump(dict(screened=screened, valid=valid), open(os.path.join(OUT, "valid_seeds.json"), "w"), indent=1)
    if len(valid) >= 10: break
print("SÀNG LỌC XONG: hợp lệ", valid, flush=True)
if len(valid) < 8:
    print("XONG (không đủ seed hợp lệ)", flush=True); raise SystemExit
ARMS = [("A", "A", W["A"][0], "A"), ("B", "B", W["B"][0], "B"), ("C", "C", W["C"][0], "C"), ("E", "E", wE, "E"),
        ("D", "D_attn", wA, "D_attn"), ("D", "D_sup", wA, "D_sup"), ("D", "D_soft", wA, "D_soft")]
for s in valid:
    data = TK.build(s)
    for kind, cname, w, fname in ARMS:
        do(s, kind, cname, w, fname, data, full=fname in ("D_attn", "D_sup"))
print("XONG", flush=True)
