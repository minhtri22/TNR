import sys, json, time
from train import *
print("khớp tham số...", flush=True)
cfgD = make_cfg("D"); target = count_shape("D", 32, cfgD)
print("D params", target, match_widths(target))
seed = 1000
t=time.time()
R = run("D", "D_oracle", seed, 32, 3e-3, epochs=5)
R["name"]="D_oracle"; r = evaluate(R); print("oracle 5ep", r["mse_test"], r["mse_ood"], r["trivial_mse"], "secs", time.time()-t)
