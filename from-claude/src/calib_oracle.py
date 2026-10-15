import time, json
from train import *
out = []
for lr, ep in [(3e-3, 100), (1e-3, 100), (3e-3, 200)]:
    for seed in (1000, 1001, 1002, 1003):
        t = time.time()
        R = run("D", "D_oracle", seed, 32, lr, epochs=ep, bs=200); R["name"] = "D_oracle"
        r = evaluate(R)
        print(lr, ep, seed, "test %.4f ood %.4f trivood %.3f train_end %.4f secs %.0f" % (r["mse_test"], r["mse_ood"], r["trivial_ood"], R["curve"][-1], time.time() - t), flush=True)
print("XONG", flush=True)
