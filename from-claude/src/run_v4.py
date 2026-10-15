"""Lượt chính thức v4 theo PREREG_v4: sàng lọc seed 50–79 bằng oracle (cần 8 hợp lệ), 8 nhánh, rồi chẩn đoán D0."""
import json, os, numpy as np
from train import *
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "run_004_main"); os.makedirs(OUT, exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D")); W = match_widths(tgt)
wA = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_attn")) - tgt))
json.dump(dict(target_params=tgt, widths=W, w_attn=wA, lr=1e-2, epochs=100, batch=200), open(os.path.join(OUT, "meta.json"), "w"), indent=1)
def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
def do(seed, kind, cname, w, fname, data, full=False, init_seed=None, tag=None):
    f = os.path.join(OUT, f"{tag or 'seed%d' % seed}_{fname}.json")
    if os.path.exists(f): return json.load(open(f))
    R = run(kind, cname, init_seed if init_seed is not None else seed, w, 1e-2, epochs=100, bs=200, data=data); R["name"] = cname
    r = evaluate(R, full=full); r["curve"] = R["curve"]; r["name"] = fname
    json.dump(conv(r), open(f, "w")); print(tag or seed, fname, "ood %.4f test %.4f train %.5f" % (r["mse_ood"], r["mse_test"], R["curve"][-1]), flush=True)
    return r
valid, screened = [], []
for s in range(50, 80):
    r = do(s, "D", "D_oracle", 32, "D_oracle", TK.build(s))
    ok = (r["final_train_loss"] < 1e-3) and (r["mse_ood"] < 0.5 * r["trivial_ood"])
    screened.append(dict(seed=s, valid=bool(ok), ood=r["mse_ood"], train=r["final_train_loss"])); 
    if ok: valid.append(s)
    json.dump(dict(screened=screened, valid=valid), open(os.path.join(OUT, "valid_seeds.json"), "w"), indent=1)
    if len(valid) >= 8: break
print("SÀNG LỌC XONG: hợp lệ", valid, flush=True)
ARMS = [("A", "A", W["A"][0], "A", False)] + [("D", n, wA, n, n in ("D_sup",)) for n in ("D_sup", "D_sup_trueo", "D_trueG", "D_cat_st", "D_cat_gs", "D_cat_rm", "D_cat_rf")]
if len(valid) >= 6:
    for s in valid:
        data = TK.build(s)
        for kind, cname, w, fname, full in ARMS:
            do(s, kind, cname, w, fname, data, full=full)
else:
    print("không đủ seed hợp lệ (<6): bỏ qua các nhánh", flush=True)
# D0: plateau thuộc về thế giới/dữ liệu hay khởi tạo? (mô tả) — dùng dữ liệu của seed v3, khởi tạo mới
for world in (20, 21, 22, 25, 27, 29):
    data = TK.build(world)
    for init in (100, 101, 102):
        do(world, "D", "D_oracle", 32, "D_oracle", data, init_seed=init, tag=f"d0_w{world}_i{init}")
print("XONG", flush=True)
