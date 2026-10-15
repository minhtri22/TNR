"""Lượt chính thức v5 theo PREREG_v5: khởi động lại nhiều lần (3) cho mọi nhánh, 5 thế giới (seed 80–84)."""
import json, os, numpy as np
from train import *
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "run_005_main"); os.makedirs(OUT, exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D")); W = match_widths(tgt)
wA = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_attn")) - tgt))
json.dump(dict(target_params=tgt, widths=W, w_attn=wA, lr=1e-2, epochs=100, batch=200, restarts=3, worlds=[80, 81, 82, 83, 84]), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
ARMS = [("A", "A", W["A"][0], "A")] + [("D", n, 32 if n == "D_oracle" else wA, n) for n in
        ("D_oracle", "D_trueG", "D_trueG_oh", "D_trueG_ent", "D_trueG_id", "D_sup_trueo", "D_sup_trueo_sg", "D_sup_sg_oh")]
for world in (80, 81, 82, 83, 84):
    data = TK.build(world)
    for r in range(3):
        for kind, cname, w, fname in ARMS:
            f = os.path.join(OUT, f"world{world}_r{r}_{fname}.json")
            if os.path.exists(f): continue
            R = run(kind, cname, 500 + world * 10 + r, w, 1e-2, epochs=100, bs=200, data=data); R["name"] = cname
            x = evaluate(R); x["curve"] = R["curve"]; x["name"] = fname
            json.dump(conv(x), open(f, "w")); print(world, r, fname, "train %.5f test %.4f ood %.4f" % (x["mse_train"], x["mse_test"], x["mse_ood"]), flush=True)
print("XONG", flush=True)
