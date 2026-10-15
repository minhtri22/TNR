import time
from train import *
R = run("D", "D_oracle", 1000, 32, 1e-2, epochs=60)
for nm in ("test", "ood"):
    d = {k: jnp.asarray(v) for k, v in R["data"][nm].items()}
    y, _ = predict("D", R["p"], d, R["cfg"], 32)
    fam = np.asarray(d["fam"]); e = np.asarray(((y - d["y"]) ** 2).mean(-1))
    print("triv", float(np.mean((R["data"][nm]["y"] - R["data"]["train"]["y"].mean(0))**2)), nm, [round(float(e[fam == f].mean()), 3) for f in np.unique(fam)], flush=True)
