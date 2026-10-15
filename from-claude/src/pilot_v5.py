import time
from train import *
for nm in ("D_trueG_oh", "D_trueG_ent", "D_sup_trueo_sg", "D_sup_sg_oh"):
    t = time.time(); R = run("D", nm, 1000, 55, 1e-2, epochs=100, bs=200); R["name"] = nm
    r = evaluate(R)
    print(nm, "train_mse %.4f test %.4f ood %.4f trivood %.3f" % (r["mse_train"], r["mse_test"], r["mse_ood"], r["trivial_ood"]),
          {k: round(v, 3) for k, v in r.get("route_ood", {}).items() if k in ("precision", "recall")}, "secs %.0f" % (time.time() - t), flush=True)
print("XONG", flush=True)
