"""Phân tích theo đúng quy tắc của PREREG_v1. Xuất results/results.json, results/bang.md và figures/*.png."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RUN = os.path.join(ROOT, "runs", "run_001_main")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
NAMES = ["A", "B", "C", "D", "D_ptr", "D_oracle", "D_fixed", "D_shuffle", "D_T1", "D_ring", "D_rand50", "D_unshared"]
R = {n: {} for n in NAMES}
for f in glob.glob(os.path.join(RUN, "seed*_*.json")):
    b = os.path.basename(f)[4:-5]; s, n = b.split("_", 1); R[n][int(s)] = json.load(open(f))
meta = json.load(open(os.path.join(RUN, "meta.json")))
seeds = sorted(set.intersection(*[set(R[n]) for n in NAMES]))
get = lambda n, k: np.array([R[n][s][k] for s in seeds])
getr = lambda n, blk, k: np.array([R[n][s][blk][k] for s in seeds])
DELTA = -0.10


def boot(rel, B=10000):
    rng = np.random.default_rng(0); idx = rng.integers(0, len(rel), (B, len(rel)))
    m = rel[idx].mean(1); return float(rel.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decide(mean, lo, hi, valid=True):
    if not valid: return "CHƯA KẾT LUẬN (UNRESOLVED)"
    if hi < 0 and mean <= DELTA: return "ĐẠT (PASS)"
    if lo > DELTA: return "KHÔNG ĐẠT (FAIL)"
    return "CHƯA KẾT LUẬN (UNRESOLVED)"


def rel(x, ref): return (x - ref) / ref


out = dict(seeds=seeds, meta=meta)
if len(seeds) < 2: raise SystemExit("chưa đủ seed")
tab = {}
for n in NAMES:
    t = dict(params=get(n, "params").mean(), mse_test=(get(n, "mse_test").mean(), get(n, "mse_test").std()),
             mse_ood=(get(n, "mse_ood").mean(), get(n, "mse_ood").std()), trivial=get(n, "trivial_mse").mean(),
             trivial_ood=get(n, "trivial_ood").mean(), macs=get(n, "macs_test").mean(), finite=int(get(n, "finite").sum()))
    if "route_ood" in R[n][seeds[0]]:
        for k in ("precision", "recall", "exact", "unnecessary", "edges_per_sample", "gate_entropy"):
            t["ood_" + k] = getr(n, "route_ood", k).mean(); t["id_" + k] = getr(n, "route_test", k).mean()
    tab[n] = t
out["table"] = tab
# điều kiện hợp lệ
base = ["A", "B", "C"]
V = {}
V["V1"] = all(abs(tab[b]["params"] - tab["D"]["params"]) / tab["D"]["params"] <= 0.05 for b in base)
bid = min(base, key=lambda b: tab[b]["mse_test"][0]); bood = min(base, key=lambda b: tab[b]["mse_ood"][0])
V["V2"] = tab[bid]["mse_test"][0] < 0.5 * tab[bid]["trivial"]
V["V3"] = tab["D"]["macs"] <= 1.5 * tab["A"]["macs"]
V["V4"] = tab["D_oracle"]["mse_ood"][0] < 0.5 * tab["D_oracle"]["trivial_ood"]
V["V5"] = all(tab[n]["finite"] >= min(7, len(seeds)) for n in NAMES)
V["V1_ptr"] = abs(tab["D_ptr"]["params"] - tab["D"]["params"]) / tab["D"]["params"] <= 0.05
out["valid"] = V; out["best_baseline_id"] = bid; out["best_baseline_ood"] = bood
H = {}
r = rel(get("D", "mse_ood"), get(bood, "mse_ood")); m, lo, hi = boot(r)
H["H1"] = dict(ref=bood, mean=m, lo=lo, hi=hi, valid=all(V[k] for k in ("V1", "V2", "V3", "V4", "V5")))
H["H1"]["verdict"] = decide(m, lo, hi, H["H1"]["valid"])
r2 = rel(get("D", "mse_test"), get(bid, "mse_test")); m2, lo2, hi2 = boot(r2)
macr = tab["D"]["macs"] / tab[bid]["macs"]
v2 = "ĐẠT (PASS)" if (hi2 <= 0.10 and macr <= 0.5) else ("KHÔNG ĐẠT (FAIL)" if (lo2 > 0.10 or macr > 0.5) else "CHƯA KẾT LUẬN (UNRESOLVED)")
H["H2"] = dict(ref=bid, mean=m2, lo=lo2, hi=hi2, mac_ratio=macr, verdict=v2)
pr, rc = tab["D"]["ood_precision"], tab["D"]["ood_recall"]
v3 = "KHÔNG ĐẠT (FAIL)" if (pr < 0.8 or rc < 0.8) else ("ĐẠT (PASS)" if hi2 <= 0.10 else "CHƯA KẾT LUẬN (UNRESOLVED)")
H["H3"] = dict(precision=pr, recall=rc, acc_hi=hi2, verdict=v3)
r4 = rel(get("D", "mse_ood"), get("D_fixed", "mse_ood")); m4, lo4, hi4 = boot(r4)
H["H4"] = dict(ref="D_fixed", mean=m4, lo=lo4, hi=hi4, verdict=decide(m4, lo4, hi4, V["V5"]))
r5 = rel(get("D_ptr", "mse_ood"), get(bood, "mse_ood")); m5, lo5, hi5 = boot(r5)
v5ok = all(V[k] for k in ("V1_ptr", "V2", "V4", "V5"))
H["H5"] = dict(ref=bood, mean=m5, lo=lo5, hi=hi5, valid=v5ok, verdict=decide(m5, lo5, hi5, v5ok))
out["H"] = H
# mô tả ablation (so với D, MSE OOD)
abl = {}
for n in NAMES:
    if n in ("D",): continue
    a = boot(rel(get(n, "mse_ood"), get("D", "mse_ood"))); abl[n] = dict(mean=a[0], lo=a[1], hi=a[2])
out["ablation_vs_D_ood"] = abl
# cơ chế
mech = {}
for n in ("D", "D_ptr"):
    for blk in ("mech_test", "mech_ood"):
        mm = [R[n][s][blk] for s in seeds]
        mech[f"{n}_{blk}"] = dict(base=float(np.mean([x["base"] for x in mm])),
                                  route_swap=float(np.mean([x["route_swap_mse"] for x in mm])),
                                  module_ablate_mean=float(np.mean([np.mean(x["ablate_module_delta"]) for x in mm])),
                                  module_ablate_max=float(np.mean([np.max(x["ablate_module_delta"]) for x in mm])),
                                  iter_ablate=[float(np.mean([x["ablate_iter_delta"][t] for x in mm])) for t in range(4)])
out["mechanism"] = mech
json.dump(out, open(os.path.join(ROOT, "results", "results.json"), "w"), indent=1, default=float)
# hình
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
main = ["A", "B", "C", "D", "D_ptr", "D_oracle"]
for a, k, ttl in ((ax[0], "mse_test", "Trong phân phối (test)"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)")):
    mu = [tab[n][k][0] for n in main]; sd = [tab[n][k][1] for n in main]
    a.bar(main, mu, yerr=sd, color=["#888", "#888", "#888", "#c33", "#e90", "#39c"]); a.set_title(ttl); a.set_ylabel("MSE")
    a.axhline(tab["D"]["trivial" if k == "mse_test" else "trivial_ood"], ls="--", c="k", lw=1)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "mse_theo_mo_hinh.png"), dpi=130); plt.close()
plt.figure(figsize=(5.5, 4))
for n in main:
    plt.scatter(tab[n]["macs"], tab[n]["mse_ood"][0], s=60); plt.annotate(n, (tab[n]["macs"], tab[n]["mse_ood"][0]))
plt.xscale("log"); plt.xlabel("MAC trung bình / mẫu"); plt.ylabel("MSE OOD"); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "mac_va_ood.png"), dpi=130); plt.close()
plt.figure(figsize=(5.5, 4))
for n in ("D", "D_ptr"):
    plt.plot(np.mean([R[n][s]["curve"] for s in seeds], 0), label=n)
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss huấn luyện"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "duong_cong_huan_luyen.png"), dpi=130); plt.close()
print(json.dumps(dict(seeds=seeds, valid=V, H=H), indent=1, default=float))
