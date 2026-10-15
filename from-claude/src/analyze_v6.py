"""Phân tích v6 theo PREREG_v6. Gộp kết quả từ nhiều thư mục (nhiều phiên Colab).
Dùng:  python src/analyze_v6.py --dirs runs/run_006_main [thư_mục_khác ...]"""
import argparse, glob, json, os, sys, numpy as np
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser(); ap.add_argument("--dirs", nargs="+", default=[os.path.join(ROOT, "runs", "run_006_main")]); a = ap.parse_args()
NAMES = ["A", "D_oracle", "D_trueG", "D_trueG_id", "D_trueG_bal", "D_trueG_tau", "D_trueG_rm", "D_e2e_id", "D_e2e_bal", "D_e2e_tau", "D_e2e_rm", "D_e2e_id_g50", "D_e2e_id_g25", "D_e2e_id_g10"]
files = {}
for d in a.dirs:
    for f in glob.glob(os.path.join(d, "world*_r*_*.json")):
        b = os.path.basename(f)[5:-5]; w, rest = b.split("_r", 1); r, n = rest.split("_", 1); files[(int(w), int(r), n)] = f
tgt = None
for d in a.dirs:
    for m in glob.glob(os.path.join(d, "meta*.json")): tgt = json.load(open(m))["target_params"]
worlds = sorted({w for (w, r, n) in files if all((w, rr, nn) in files for rr in range(3) for nn in NAMES)})
RR = {n: {w: [json.load(open(files[(w, r, n)])) for r in range(3)] for w in worlds} for n in NAMES}
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
             conv_runs=int(sum(x["mse_train"] < 1e-3 for x in allr)), n_runs=len(allr), conv_worlds=int(sum(best[n][w]["mse_train"] < 1e-3 for w in worlds)))
    t["C0"] = t["mse_ood"][0] <= 0.5 * t["trivial_ood"]; t["C0_ID"] = t["mse_test"][0] <= 0.5 * t["trivial"]
    if "route_ood" in best[n][worlds[0]]:
        for k in ("precision", "recall"): t["ood_" + k] = float(getr(n, "route_ood", k).mean())
    tab[n] = t
nw = len(worlds)
V = dict(n_worlds=nw, V0=nw >= 4, V1=all(abs(tab[n]["params"] - tgt) / tgt <= 0.06 for n in NAMES if n != "D_oracle"),
         V4p=sum((best["D_oracle"][w]["mse_train"] < 1e-3 and best["D_oracle"][w]["mse_ood"] < 0.5 * best["D_oracle"][w]["trivial_ood"]) for w in worlds) >= max(1, int(np.ceil(2 * nw / 3))))
ok = V["V0"] and V["V1"]; okood = ok and V["V4p"]
H = {}
for key, arm in (("H1a", "D_trueG_bal"), ("H1b", "D_trueG_tau"), ("H1c", "D_trueG_rm")):
    m, lo, hi = boot(rel(get(arm, "mse_test"), get("D_trueG", "mse_test")), 0.833, 99.167)
    H[key] = dict(arm=arm, ref="D_trueG", mean=m, lo=lo, hi=hi, c0=bool(tab[arm]["C0_ID"]), verdict=decide(m, lo, hi, ok, tab[arm]["C0_ID"]))
for key, arm in (("H2a", "D_e2e_bal"), ("H2b", "D_e2e_tau"), ("H2c", "D_e2e_rm"), ("H2d", "D_e2e_id")):
    m, lo, hi = boot(rel(get(arm, "mse_ood"), get("A", "mse_ood")), 0.625, 99.375)
    H[key] = dict(arm=arm, ref="A", mean=m, lo=lo, hi=hi, c0=bool(tab[arm]["C0"]), verdict=decide(m, lo, hi, okood, tab[arm]["C0"]))
for arm in ("D_e2e_bal", "D_e2e_tau", "D_e2e_rm", "D_e2e_id"):
    pr, rc = tab[arm]["ood_precision"], tab[arm]["ood_recall"]
    H["H3_" + arm] = dict(precision=pr, recall=rc, verdict="CHƯA KẾT LUẬN (UNRESOLVED)" if not ok else ("ĐẠT (PASS)" if pr >= 0.8 and rc >= 0.8 else "KHÔNG ĐẠT (FAIL)"))
for key, arm in (("H4a", "D_e2e_id_g50"), ("H4b", "D_e2e_id_g25"), ("H4c", "D_e2e_id_g10")):
    m, lo, hi = boot(rel(get(arm, "mse_ood"), get("D_e2e_id", "mse_ood")), 0.833, 99.167)
    v = "CHƯA KẾT LUẬN (UNRESOLVED)" if not okood else ("ĐẠT (PASS)" if (hi <= 0.10 and tab[arm]["C0"]) else "KHÔNG ĐẠT (FAIL)" if (lo > 0.10 or (hi <= 0.10 and not tab[arm]["C0"])) else "CHƯA KẾT LUẬN (UNRESOLVED)")
    H[key] = dict(arm=arm, ref="D_e2e_id", mean=m, lo=lo, hi=hi, c0=bool(tab[arm]["C0"]), recall_ood=tab[arm]["ood_recall"], verdict=v)
fi = {}
for n in NAMES:
    if n == "D_trueG": continue
    x, y = tab[n]["conv_runs"], tab[n]["n_runs"] - tab[n]["conv_runs"]; c, d2 = tab["D_trueG"]["conv_runs"], tab["D_trueG"]["n_runs"] - tab["D_trueG"]["conv_runs"]
    fi[n] = dict(conv=f"{x}/{x+y}", p_fisher_vs_D_trueG=float(fisher_exact([[x, y], [c, d2]])[1]))
out = dict(worlds=worlds, table=tab, valid=V, H=H, fisher_hoi_tu=fi)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "results_v6.json"), "w"), indent=1, default=float)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for ax_, k, ttl in ((ax[0], "mse_test", "Trong phân phối (tốt nhất trong 3 lần)"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)")):
    ax_.bar(range(len(NAMES)), [tab[n][k][0] for n in NAMES], yerr=[tab[n][k][1] for n in NAMES]); ax_.set_xticks(range(len(NAMES))); ax_.set_xticklabels(NAMES, rotation=70, fontsize=6)
    ax_.set_yscale("symlog", linthresh=1e-3); ax_.set_title(ttl)
ax[2].bar(range(len(NAMES)), [tab[n]["conv_runs"] / tab[n]["n_runs"] for n in NAMES]); ax[2].set_xticks(range(len(NAMES))); ax[2].set_xticklabels(NAMES, rotation=70, fontsize=6); ax[2].set_title("Tỷ lệ lần chạy hội tụ")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "v6_ket_qua.png"), dpi=130)
print(nw, "thế giới"); print(json.dumps(dict(valid=V, H=H, fisher=fi), indent=1, default=float, ensure_ascii=False))
