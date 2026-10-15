import sys, json, time
from train import *
seed = 1000
tgt = count_shape("D", 32, make_cfg("D"))
mh_ptr = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_ptr")) - tgt))
W = match_widths(tgt); print("target", tgt, W, "mh_ptr", mh_ptr, flush=True)
jobs = [("D", "D", 32), ("D", "D_ptr", mh_ptr), ("A", "A", W["A"][0]), ("B", "B", W["B"][0]), ("C", "C", W["C"][0])]
for lr in (1e-2,):
    for kind, name, w in jobs:
        t = time.time()
        R = run(kind, name, seed, w, lr, epochs=40, bs=200); R["name"] = name
        r = evaluate(R)
        print(name, lr, "val %.4f test %.4f ood %.4f triv %.3f train %.4f secs %.0f macs %.0f" % (r["mse_val"], r["mse_test"], r["mse_ood"], r["trivial_mse"], r["mse_train"], time.time() - t, r["macs_test"]),
              "route_ood", {k: round(v, 3) for k, v in r.get("route_ood", {}).items()}, flush=True)
