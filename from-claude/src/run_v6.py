"""Lượt chính thức v6 theo PREREG_v6 — thiết kế để chạy trên Colab (tự tiếp tục, lưu kết quả + checkpoint ra thư mục --out).
Ví dụ:  python src/run_v6.py --worlds 90 91 --out /content/drive/MyDrive/tnr_v6"""
import argparse, hashlib, json, os, sys, numpy as np, jax
sys.path.insert(0, os.path.dirname(__file__))
from train import *
ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, nargs="+", default=[90, 91, 92, 93, 94, 95])
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs", "run_006_main"))
ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--epochs", type=int, default=100)      # giữ 100 cho lượt chính thức (PREREG_v6); giá trị khác chỉ để chạy thử
ap.add_argument("--arms", default="")                    # danh sách nhánh cách nhau dấu phẩy, rỗng = tất cả
a = ap.parse_args()
OUT = a.out; os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
tgt = count_shape("D", 32, make_cfg("D")); W = match_widths(tgt)
wA = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_attn")) - tgt))
ALL = [("A", "A", W["A"][0])] + [("D", n, 32 if n == "D_oracle" else wA) for n in
       ("D_oracle", "D_trueG", "D_trueG_id", "D_trueG_bal", "D_trueG_tau", "D_trueG_rm",
        "D_e2e_id", "D_e2e_bal", "D_e2e_tau", "D_e2e_rm", "D_e2e_id_g50", "D_e2e_id_g25", "D_e2e_id_g10")]
ARMS = [x for x in ALL if not a.arms or x[1] in a.arms.split(",")]
meta = dict(target_params=tgt, widths=W, w_attn=wA, lr=1e-2, epochs=a.epochs, batch=200, restarts=a.restarts, worlds=a.worlds, arms=[x[1] for x in ARMS], jax=jax.__version__)
json.dump(meta, open(os.path.join(OUT, f"meta_{'_'.join(map(str, a.worlds))}.json"), "w"), indent=1)


def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def save_ckpt(R, path):
    flat, _ = jax.tree_util.tree_flatten_with_path(R["p"])
    np.savez_compressed(path, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})


for world in a.worlds:
    data = TK.build(world)
    for r in range(a.restarts):
        for kind, cname, w in ARMS:
            f = os.path.join(OUT, f"world{world}_r{r}_{cname}.json")
            if os.path.exists(f): continue
            R = run(kind, cname, 500 + world * 10 + r, w, 1e-2, epochs=a.epochs, bs=200, data=data); R["name"] = cname
            x = evaluate(R); x["curve"] = R["curve"]; x["name"] = cname
            save_ckpt(R, os.path.join(OUT, "ckpt", f"world{world}_r{r}_{cname}.npz"))
            json.dump(conv(x), open(f, "w")); print(world, r, cname, "train %.5f test %.4f ood %.4f" % (x["mse_train"], x["mse_test"], x["mse_ood"]), flush=True)
with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as m:
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p) and fn != "MANIFEST.sha256": m.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {fn}\n")
print("XONG", flush=True)
