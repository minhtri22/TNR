import sys, time
from train import *
seed = 1000
cfgs = [(1e-2, 60), (3e-3, 150), (1e-2, 150)]
for lr, ep in cfgs:
    t = time.time()
    R = run("D", "D_oracle", seed, 32, lr, epochs=ep); R["name"] = "D_oracle"
    r = evaluate(R)
    print("oracle lr", lr, "ep", ep, "test %.4f ood %.4f train %.4f triv_id %.3f triv_ood %.3f secs %.0f" % (r["mse_test"], r["mse_ood"], r["mse_train"], r["trivial_mse"], r["trivial_ood"], time.time() - t), flush=True)
