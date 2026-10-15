import time
from train import *
for nm in ("D_cat_st", "D_cat_gs", "D_cat_rm", "D_cat_rf", "D_trueG", "D_sup_trueo", "D_sup"):
    t = time.time(); R = run("D", nm, 1000, 55, 1e-2, epochs=100, bs=200); R["name"] = nm
    r = evaluate(R)
    print(nm, "val %.4f test %.4f ood %.4f trivood %.3f train %.5f secs %.0f" % (r["mse_val"], r["mse_test"], r["mse_ood"], r["trivial_ood"], R["curve"][-1], time.time() - t),
          {k: round(v, 3) for k, v in r.get("route_ood", {}).items() if k in ("precision", "recall", "node_recall")}, flush=True)
print("XONG", flush=True)
