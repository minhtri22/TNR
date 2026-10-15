import sys, time, json
from train import *
seed = 1000
tgt = count_shape("D", 32, make_cfg("D"))
mh_ptr = min(range(8, 300), key=lambda w: abs(count_shape("D", w, make_cfg("D_ptr")) - tgt))
jobs = [("D_oracle", 32, None, 100), ("D_oracle", 32, None, 160), ("D", 32, None, 100), ("D", 32, "anneal", 100),
        ("D_ptr", mh_ptr, None, 100), ("D_ptr", mh_ptr, "anneal", 100)]
for name, w, sc, ep in jobs:
    t = time.time()
    R = run("D", name, seed, w, 1e-2, epochs=ep, bs=200, sched=sc); R["name"] = name
    r = evaluate(R)
    print(name, sc, ep, "val %.4f test %.4f ood %.4f trivood %.3f macs %.0f secs %.0f" % (r["mse_val"], r["mse_test"], r["mse_ood"], r["trivial_ood"], r["macs_test"], time.time() - t),
          {k: round(v, 3) for k, v in r.get("route_ood", {}).items() if k in ("precision", "recall", "gate_entropy", "edges_per_sample")}, flush=True)
