"""Phân tích v5 theo PREREG_v5. Xuất results/results_v5.json và figures/v5_ket_qua.png."""
import json, glob, os, numpy as np
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RUN = os.path.join(ROOT, "runs", "run_005_main")
meta = json.load(open(os.path.join(RUN, "meta.json"))); tgt = meta["target_params"]
NAMES = ["A", "D_oracle", "D_trueG", "D_trueG_oh", "D_trueG_ent", "D_trueG_id", "D_sup_trueo", "D_sup_trueo_sg", "D_sup_sg_oh"]
ld = lambda w, r, n: (json.load(open(os.path.join(RUN, f"world{w}_r{r}_{n}.json"))) if os.path.exists(os.path.join(RUN, f"world{w}_r{r}_{n}.json")) else None)
worlds = [w for w in meta["worlds"] if all(ld(w, r, n) is not None for r in range(3) for n in NAMES)]
RR = {n: {w: [ld(w, r, n) for r in range(3)] for w in worlds} for n in NAMES}
best = {n: {w: min(RR[n][w], key=lambda x: x["mse_train"]) for w in worlds} for n in NAMES}   # chọn theo loss huấn luyện, không theo OOD
get = lambda n, k: np.array([best[n][w][k] for w in worlds]); getr = lambda n, b, k: np.array([best[n][w][b][k] for w in worlds])
DELTA = -0.10


def boot(x, lo_p, hi_p, B=10000):
    rng = np.random.default_rng(0); idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, lo_p)), float(np.percentile(m, hi_p))


def decide(mean, lo, hi, valid, c0):
    if not valid: return "CHƯA KẾT LUẬN (UNRESOLVED)"
    if hi < 0 and mean <= DELTA: return "ĐẠT (PASS)" if c0 else "KHÔNG ĐẠT (FAIL) [quy tắc δ thỏa nhưng không thỏa C0]"
    if lo > DELTA: return "KHÔNG ĐẠT (FAIL)"
    return "CHƯA KẾT LUẬN (UNRESOLVED)"


rel = lambda x, r: (x - r) / r
tab = {}
for n in NAMES:
    allr = [x for w in worlds for x in RR[n][w]]
    t = dict(params=float(get(n, "params").mean()), mse_test=(float(get(n, "mse_test").mean()), float(get(n, "mse_test").std())),
             mse_ood=(float(get(n, "mse_ood").mean()), float(get(n, "mse_ood").std())), trivial=float(get(n, "trivial_mse").mean()),
             trivial_ood=float(get(n, "trivial_ood").mean()), macs=float(get(n, "macs_test").mean()),
             conv_runs=int(sum(x["mse_train"] < 1e-3 for x in allr)), n_runs=len(allr),
             conv_worlds=int(sum(best[n][w]["mse_train"] < 1e-3 for w in worlds)),
             restart_test=[[x["mse_test"] for x in RR[n][w]] for w in worlds])
    t["C0"] = t["mse_ood"][0] <= 0.5 * t["trivial_ood"]; t["C0_ID"] = t["mse_test"][0] <= 0.5 * t["trivial"]
    if "route_ood" in best[n][worlds[0]]:
        for k in ("precision", "recall"): t["ood_" + k] = float(getr(n, "route_ood", k).mean())
    tab[n] = t
V = dict(n_worlds=len(worlds), V0=len(worlds) >= 4, V1=all(abs(tab[n]["params"] - tgt) / tgt <= 0.06 for n in NAMES if n != "D_oracle"),
         V4p=sum((best["D_oracle"][w]["mse_train"] < 1e-3 and best["D_oracle"][w]["mse_ood"] < 0.5 * best["D_oracle"][w]["trivial_ood"]) for w in worlds) >= 4)
ok = V["V0"] and V["V1"]; okood = ok and V["V4p"]
H = {}
for key, arm in (("H1", "D_trueG_oh"), ("H2", "D_trueG_ent"), ("H3", "D_trueG_id")):
    m, lo, hi = boot(rel(get(arm, "mse_test"), get("D_trueG", "mse_test")), 0.833, 99.167)
    H[key] = dict(arm=arm, ref="D_trueG", mean=m, lo=lo, hi=hi, c0=bool(tab[arm]["C0_ID"]), verdict=decide(m, lo, hi, ok, tab[arm]["C0_ID"]))
d = getr("D_sup_trueo_sg", "route_ood", "recall") - getr("D_sup_trueo", "route_ood", "recall"); m, lo, hi = boot(d, 2.5, 97.5)
H["H4"] = dict(mean_recall_diff=m, lo=lo, hi=hi, recall_sg=float(getr("D_sup_trueo_sg", "route_ood", "recall").mean()), recall_base=float(getr("D_sup_trueo", "route_ood", "recall").mean()),
               verdict=("CHƯA KẾT LUẬN (UNRESOLVED)" if not ok else "ĐẠT (PASS)" if (lo > 0 and m >= 0.2) else "KHÔNG ĐẠT (FAIL)" if hi <= 0.05 else "CHƯA KẾT LUẬN (UNRESOLVED)"))
for key, arm in (("H5a", "D_trueG_oh"), ("H5b", "D_trueG_ent"), ("H5c", "D_trueG_id")):
    m, lo, hi = boot(rel(get(arm, "mse_ood"), get("A", "mse_ood")), 0.833, 99.167)
    H[key] = dict(arm=arm, ref="A", mean=m, lo=lo, hi=hi, c0=bool(tab[arm]["C0"]), verdict=decide(m, lo, hi, okood, tab[arm]["C0"]))
m, lo, hi = boot(rel(get("D_sup_sg_oh", "mse_ood"), get("A", "mse_ood")), 2.5, 97.5)
H["H6"] = dict(arm="D_sup_sg_oh", ref="A", mean=m, lo=lo, hi=hi, c0=bool(tab["D_sup_sg_oh"]["C0"]), verdict=decide(m, lo, hi, okood, tab["D_sup_sg_oh"]["C0"]))
H["H7_sg_oh_route"] = dict(precision=tab["D_sup_sg_oh"]["ood_precision"], recall=tab["D_sup_sg_oh"]["ood_recall"],
                            verdict=("ĐẠT (PASS)" if (tab["D_sup_sg_oh"]["ood_precision"] >= 0.8 and tab["D_sup_sg_oh"]["ood_recall"] >= 0.8) else "KHÔNG ĐẠT (FAIL)"))
fi = {}
for n in NAMES:
    if n == "D_trueG": continue
    a, b = tab[n]["conv_runs"], tab[n]["n_runs"] - tab[n]["conv_runs"]; c, dd = tab["D_trueG"]["conv_runs"], tab["D_trueG"]["n_runs"] - tab["D_trueG"]["conv_runs"]
    fi[n] = dict(conv=f"{a}/{a+b}", p_fisher_vs_D_trueG=float(fisher_exact([[a, b], [c, dd]])[1]))
out = dict(worlds=worlds, table=tab, valid=V, H=H, fisher_hoi_tu=fi)
json.dump(out, open(os.path.join(ROOT, "results", "results_v5.json"), "w"), indent=1, default=float)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, k, ttl in ((ax[0], "mse_test", "Trong phân phối (tốt nhất trong 3 lần)"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)")):
    a.bar(range(len(NAMES)), [tab[n][k][0] for n in NAMES], yerr=[tab[n][k][1] for n in NAMES]); a.set_xticks(range(len(NAMES))); a.set_xticklabels(NAMES, rotation=65, fontsize=7)
    a.set_yscale("symlog", linthresh=1e-3); a.set_title(ttl)
ax[2].bar(range(len(NAMES)), [tab[n]["conv_runs"] / tab[n]["n_runs"] for n in NAMES]); ax[2].set_xticks(range(len(NAMES))); ax[2].set_xticklabels(NAMES, rotation=65, fontsize=7); ax[2].set_title("Tỷ lệ lần chạy hội tụ (loss huấn luyện < 1e-3)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "v5_ket_qua.png"), dpi=130)
print(len(worlds), "thế giới"); print(json.dumps(dict(valid=V, H=H, fisher=fi), indent=1, default=float, ensure_ascii=False))
