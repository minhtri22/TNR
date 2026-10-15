"""Phân tích v4 theo PREREG_v4. Xuất results/results_v4.json và figures/v4_ket_qua.png."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RUN = os.path.join(ROOT, "runs", "run_004_main")
meta = json.load(open(os.path.join(RUN, "meta.json"))); vs = json.load(open(os.path.join(RUN, "valid_seeds.json")))
NAMES = ["A", "D_sup", "D_sup_trueo", "D_trueG", "D_cat_st", "D_cat_gs", "D_cat_rm", "D_cat_rf", "D_oracle"]
ld = lambda s, n: json.load(open(os.path.join(RUN, f"seed{s}_{n}.json"))) if os.path.exists(os.path.join(RUN, f"seed{s}_{n}.json")) else None
seeds = [s for s in vs["valid"] if all(ld(s, n) is not None for n in NAMES)]
R = {n: {s: ld(s, n) for s in seeds} for n in NAMES}
get = lambda n, k: np.array([R[n][s][k] for s in seeds]); getr = lambda n, b, k: np.array([R[n][s][b][k] for s in seeds])
DELTA = -0.10; tgt = meta["target_params"]


def boot(rel, lo_p, hi_p, B=10000):
    rng = np.random.default_rng(0); idx = rng.integers(0, len(rel), (B, len(rel))); m = rel[idx].mean(1)
    return float(rel.mean()), float(np.percentile(m, lo_p)), float(np.percentile(m, hi_p))


def decide(mean, lo, hi, valid, c0):
    if not valid: return "CHƯA KẾT LUẬN (UNRESOLVED)"
    if hi < 0 and mean <= DELTA: return "ĐẠT (PASS)" if c0 else "KHÔNG ĐẠT (FAIL) [quy tắc δ thỏa nhưng không thỏa C0]"
    if lo > DELTA: return "KHÔNG ĐẠT (FAIL)"
    return "CHƯA KẾT LUẬN (UNRESOLVED)"


rel = lambda x, r: (x - r) / r
tab = {}
for n in NAMES:
    t = dict(params=float(get(n, "params").mean()), mse_test=(float(get(n, "mse_test").mean()), float(get(n, "mse_test").std())),
             mse_ood=(float(get(n, "mse_ood").mean()), float(get(n, "mse_ood").std())), trivial=float(get(n, "trivial_mse").mean()),
             trivial_ood=float(get(n, "trivial_ood").mean()), macs=float(get(n, "macs_test").mean()), finite=int(get(n, "finite").sum()))
    t["C0"] = t["mse_ood"][0] <= 0.5 * t["trivial_ood"]; t["C0_ID"] = t["mse_test"][0] <= 0.5 * t["trivial"]
    if "route_ood" in R[n][seeds[0]]:
        for k in ("precision", "recall", "node_recall", "unnecessary", "edges_per_sample"):
            t["ood_" + k] = float(getr(n, "route_ood", k).mean()); t["id_" + k] = float(getr(n, "route_test", k).mean())
    tab[n] = t
V = dict(n_valid=len(seeds), V0=len(seeds) >= 6, V1=all(abs(tab[n]["params"] - tgt) / tgt <= 0.05 for n in NAMES if n not in ("D_oracle", "A")) and abs(tab["A"]["params"] - tgt) / tgt <= 0.05,
         V2=tab["A"]["C0_ID"], V5=all(tab[n]["finite"] >= len(seeds) - 1 for n in NAMES))
ok = V["V0"] and V["V1"] and V["V2"] and V["V5"]
H = {}
def cmp(arm, ref, key, lo_p, hi_p, c0key):
    m, lo, hi = boot(rel(get(arm, key), get(ref, key)), lo_p, hi_p)
    return dict(arm=arm, ref=ref, mean=m, lo=lo, hi=hi, c0=bool(tab[arm][c0key]), verdict=decide(m, lo, hi, ok, tab[arm][c0key]))
H["H1"] = cmp("D_sup_trueo", "D_sup", "mse_ood", 1.25, 98.75, "C0"); H["H2"] = cmp("D_trueG", "D_sup", "mse_ood", 1.25, 98.75, "C0")
p1, p2 = H["H1"]["verdict"].startswith("ĐẠT"), H["H2"]["verdict"].startswith("ĐẠT")
H["H1H2_dien_giai"] = ("cả đầu đọc và cổng" if p1 and p2 else "đầu đọc là nguyên nhân" if p1 else "cổng học được là nguyên nhân" if p2 else "nguyên nhân ở nơi khác (module/dữ liệu), hoặc chưa kết luận")
for e in ("gs", "rm", "rf"): H["H3_" + e] = cmp("D_cat_" + e, "D_cat_st", "mse_test", 0.833, 99.167, "C0_ID")
for e in ("st", "gs", "rm", "rf"):
    a = "D_cat_" + e; pr, rc = tab[a]["ood_precision"], tab[a]["ood_recall"]
    H["H4_" + e] = dict(precision=pr, recall=rc, verdict="CHƯA KẾT LUẬN (UNRESOLVED)" if not V["V0"] else ("ĐẠT (PASS)" if pr >= 0.8 and rc >= 0.8 else "KHÔNG ĐẠT (FAIL)"))
    H["H5_" + e] = cmp(a, "A", "mse_ood", 0.625, 99.375, "C0")
H["H6"] = cmp("D_sup", "A", "mse_ood", 2.5, 97.5, "C0")
# can thiệp lúc đánh giá trên D_sup
iv = {k: float(np.mean([R["D_sup"][s]["interv"][k] for s in seeds])) for k in R["D_sup"][seeds[0]]["interv"]}
# D0
d0 = {}
for f in sorted(glob.glob(os.path.join(RUN, "d0_w*_i*_D_oracle.json"))):
    b = os.path.basename(f); w = int(b.split("_w")[1].split("_")[0]); r = json.load(open(f))
    d0.setdefault(w, []).append(dict(train=r["final_train_loss"], ood=r["mse_ood"], conv=bool(r["final_train_loss"] < 1e-3 and r["mse_ood"] < 0.5 * r["trivial_ood"])))
out = dict(seeds=seeds, screened=vs["screened"], table=tab, valid=V, H=H, intervention_D_sup=iv, D0=d0)
json.dump(out, open(os.path.join(ROOT, "results", "results_v4.json"), "w"), indent=1, default=float)
fig, ax = plt.subplots(1, 3, figsize=(14, 4)); nm = NAMES
for a, k, ttl, tr in ((ax[0], "mse_test", "Trong phân phối", "trivial"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)", "trivial_ood")):
    a.bar(range(len(nm)), [tab[n][k][0] for n in nm], yerr=[tab[n][k][1] for n in nm]); a.set_xticks(range(len(nm))); a.set_xticklabels(nm, rotation=60, fontsize=8)
    a.axhline(tab["A"][tr], ls="--", c="k", lw=1); a.axhline(0.5 * tab["A"][tr], ls=":", c="r", lw=1); a.set_title(ttl)
cats = ["D_sup", "D_cat_st", "D_cat_gs", "D_cat_rm", "D_cat_rf"]; x = np.arange(len(cats))
ax[2].bar(x - 0.2, [tab[a]["ood_precision"] for a in cats], 0.4, label="precision"); ax[2].bar(x + 0.2, [tab[a]["ood_recall"] for a in cats], 0.4, label="recall")
ax[2].axhline(0.8, ls="--", c="k"); ax[2].set_xticks(x); ax[2].set_xticklabels(cats, rotation=30, fontsize=8); ax[2].legend(); ax[2].set_title("Khôi phục cạnh (OOD)")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "v4_ket_qua.png"), dpi=130)
print(len(seeds), "seed hợp lệ"); print(json.dumps(dict(valid=V, H=H, interv=iv, D0=d0), indent=1, default=float, ensure_ascii=False))
