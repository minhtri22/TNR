"""Phân tích v2 theo PREREG_v2. Xuất results/results_v2.json và figures/v2_*.png."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RUN = os.path.join(ROOT, "runs", "run_002_main")
NAMES = ["A", "B", "C", "D", "D_anneal", "D_ptr", "D_ptr_anneal", "D_oracle", "D_fixed"]
R = {n: {} for n in NAMES}
for f in glob.glob(os.path.join(RUN, "seed*_*.json")):
    b = os.path.basename(f)[4:-5]; s, n = b.split("_", 1); R[n][int(s)] = json.load(open(f))
seeds = sorted(set.intersection(*[set(R[n]) for n in NAMES]))
get = lambda n, k: np.array([R[n][s][k] for s in seeds]); getr = lambda n, b, k: np.array([R[n][s][b][k] for s in seeds])
DELTA = -0.10


def boot(rel, lo_p=2.5, hi_p=97.5, B=10000):
    rng = np.random.default_rng(0); idx = rng.integers(0, len(rel), (B, len(rel))); m = rel[idx].mean(1)
    return float(rel.mean()), float(np.percentile(m, lo_p)), float(np.percentile(m, hi_p))


def decide(mean, lo, hi, valid=True):
    if not valid: return "CHƯA KẾT LUẬN (UNRESOLVED)"
    if hi < 0 and mean <= DELTA: return "ĐẠT (PASS)"
    if lo > DELTA: return "KHÔNG ĐẠT (FAIL)"
    return "CHƯA KẾT LUẬN (UNRESOLVED)"


rel = lambda x, r: (x - r) / r
BON = (0.625, 99.375)  # khoảng 98,75%
tab = {}
for n in NAMES:
    t = dict(params=float(get(n, "params").mean()), mse_test=(float(get(n, "mse_test").mean()), float(get(n, "mse_test").std())),
             mse_ood=(float(get(n, "mse_ood").mean()), float(get(n, "mse_ood").std())), trivial=float(get(n, "trivial_mse").mean()),
             trivial_ood=float(get(n, "trivial_ood").mean()), macs=float(get(n, "macs_test").mean()), finite=int(get(n, "finite").sum()))
    if "route_ood" in R[n][seeds[0]]:
        for k in ("precision", "recall", "exact", "unnecessary", "edges_per_sample", "gate_entropy"):
            t["ood_" + k] = float(getr(n, "route_ood", k).mean()); t["id_" + k] = float(getr(n, "route_test", k).mean())
    tab[n] = t
base = ["A", "B", "C"]; bid = min(base, key=lambda b: tab[b]["mse_test"][0]); bood = min(base, key=lambda b: tab[b]["mse_ood"][0])
pm = lambda n: abs(tab[n]["params"] - tab["D"]["params"]) / tab["D"]["params"] <= 0.05
V = dict(V1=all(pm(b) for b in base), V1_ptr=pm("D_ptr") and pm("D_ptr_anneal"), V2=tab[bid]["mse_test"][0] < 0.5 * tab[bid]["trivial"],
         V3={n: tab[n]["macs"] <= 2.0 * tab["A"]["macs"] for n in ("D", "D_anneal", "D_ptr", "D_ptr_anneal")},
         V4=tab["D_oracle"]["mse_ood"][0] < 0.5 * tab["D_oracle"]["trivial_ood"], V5=all(tab[n]["finite"] >= min(11, len(seeds)) for n in NAMES))
H = {}
def ood_test(arm, ref, extra_valid):
    m, lo, hi = boot(rel(get(arm, "mse_ood"), get(ref, "mse_ood")), *BON)
    return dict(arm=arm, ref=ref, mean=m, lo=lo, hi=hi, valid=bool(extra_valid), verdict=decide(m, lo, hi, extra_valid))
core = V["V1"] and V["V2"] and V["V4"] and V["V5"]
H["H1"] = ood_test("D", bood, core and V["V3"]["D"]); H["H1b"] = ood_test("D_anneal", bood, core and V["V3"]["D_anneal"])
H["H5"] = ood_test("D_ptr", bood, V["V1_ptr"] and V["V2"] and V["V4"] and V["V5"] and V["V3"]["D_ptr"])
H["H5b"] = ood_test("D_ptr_anneal", bood, V["V1_ptr"] and V["V2"] and V["V4"] and V["V5"] and V["V3"]["D_ptr_anneal"])
m, lo, hi = boot(rel(get("D", "mse_ood"), get("D_fixed", "mse_ood"))); H["H4"] = dict(mean=m, lo=lo, hi=hi, verdict=decide(m, lo, hi, V["V5"]))
for arm in ("D", "D_anneal"):
    m2, lo2, hi2 = boot(rel(get(arm, "mse_test"), get(bid, "mse_test"))); macr = tab[arm]["macs"] / tab[bid]["macs"]
    v = "ĐẠT (PASS)" if (hi2 <= 0.10 and macr <= 0.5) else ("KHÔNG ĐẠT (FAIL)" if (lo2 > 0.10 or macr > 0.5) else "CHƯA KẾT LUẬN (UNRESOLVED)")
    H["H2_" + arm] = dict(ref=bid, mean=m2, lo=lo2, hi=hi2, mac_ratio=macr, verdict=v)
for arm in ("D", "D_anneal", "D_ptr", "D_ptr_anneal"):
    m2, lo2, hi2 = boot(rel(get(arm, "mse_test"), get(bid, "mse_test")))
    pr, rc = tab[arm]["ood_precision"], tab[arm]["ood_recall"]
    H["H3_" + arm] = dict(precision=pr, recall=rc, acc_hi=hi2, verdict="KHÔNG ĐẠT (FAIL)" if (pr < 0.8 or rc < 0.8) else ("ĐẠT (PASS)" if hi2 <= 0.10 else "CHƯA KẾT LUẬN (UNRESOLVED)"))
out = dict(seeds=seeds, table=tab, valid=V, H=H, best_id=bid, best_ood=bood)
json.dump(out, open(os.path.join(ROOT, "results", "results_v2.json"), "w"), indent=1, default=float)
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); main = NAMES
for a, k, ttl in ((ax[0], "mse_test", "Trong phân phối"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)")):
    a.bar(range(len(main)), [tab[n][k][0] for n in main], yerr=[tab[n][k][1] for n in main]); a.set_xticks(range(len(main))); a.set_xticklabels(main, rotation=60, fontsize=8)
    a.axhline(tab["D"]["trivial" if k == "mse_test" else "trivial_ood"], ls="--", c="k", lw=1); a.set_title(ttl); a.set_ylabel("MSE")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "v2_mse_theo_mo_hinh.png"), dpi=130)
print(len(seeds), "seed"); print(json.dumps(dict(valid=V, H=H), indent=1, default=float))
