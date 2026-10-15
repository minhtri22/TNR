"""Phân tích v3 theo PREREG_v3. Xuất results/results_v3.json và figures/v3_*.png."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); RUN = os.path.join(ROOT, "runs", "run_003_main")
meta = json.load(open(os.path.join(RUN, "meta.json"))); vs = json.load(open(os.path.join(RUN, "valid_seeds.json")))
seeds = vs["valid"]; NAMES = ["A", "B", "C", "E", "D_attn", "D_sup", "D_soft", "D_oracle"]
R = {n: {s: json.load(open(os.path.join(RUN, f"seed{s}_{n}.json"))) for s in seeds if os.path.exists(os.path.join(RUN, f"seed{s}_{n}.json"))} for n in NAMES}
seeds = [s for s in seeds if all(s in R[n] for n in NAMES)]
get = lambda n, k: np.array([R[n][s][k] for s in seeds]); getr = lambda n, b, k: np.array([R[n][s][b][k] for s in seeds])
DELTA = -0.10; tgt = meta["target_params"]


def boot(rel, lo_p=2.5, hi_p=97.5, B=10000):
    rng = np.random.default_rng(0); idx = rng.integers(0, len(rel), (B, len(rel))); m = rel[idx].mean(1)
    return float(rel.mean()), float(np.percentile(m, lo_p)), float(np.percentile(m, hi_p))


def decide(mean, lo, hi, valid=True):
    if not valid: return "CHƯA KẾT LUẬN (UNRESOLVED)"
    if hi < 0 and mean <= DELTA: return "ĐẠT (PASS)"
    if lo > DELTA: return "KHÔNG ĐẠT (FAIL)"
    return "CHƯA KẾT LUẬN (UNRESOLVED)"


rel = lambda x, r: (x - r) / r
BON = (1.25, 98.75)
tab = {}
for n in NAMES:
    t = dict(params=float(get(n, "params").mean()), mse_test=(float(get(n, "mse_test").mean()), float(get(n, "mse_test").std())),
             mse_ood=(float(get(n, "mse_ood").mean()), float(get(n, "mse_ood").std())), trivial=float(get(n, "trivial_mse").mean()),
             trivial_ood=float(get(n, "trivial_ood").mean()), macs=float(get(n, "macs_test").mean()), finite=int(get(n, "finite").sum()))
    if "route_ood" in R[n][seeds[0]]:
        for k in ("precision", "recall", "node_recall", "node_precision", "usage_entropy", "unnecessary", "edges_per_sample", "gate_entropy"):
            t["ood_" + k] = float(getr(n, "route_ood", k).mean()); t["id_" + k] = float(getr(n, "route_test", k).mean())
    tab[n] = t
base = ["A", "B", "C", "E"]; bid = min(base, key=lambda b: tab[b]["mse_test"][0]); bood = min(base, key=lambda b: tab[b]["mse_ood"][0])
pm = lambda n: abs(tab[n]["params"] - tgt) / tgt <= 0.05
V = dict(n_valid_seeds=len(seeds), V0_enough_seeds=len(seeds) >= 8, V1=all(pm(n) for n in ("A", "B", "C", "E", "D_attn", "D_sup", "D_soft")),
         V2=tab[bid]["mse_test"][0] < 0.5 * tab[bid]["trivial"], V3={n: tab[n]["macs"] <= 2.0 * tab["A"]["macs"] for n in ("D_attn", "D_sup")},
         V5=all(tab[n]["finite"] >= len(seeds) - 1 for n in NAMES))
H = {}
def ood(arm):
    m, lo, hi = boot(rel(get(arm, "mse_ood"), get(bood, "mse_ood")), *BON)
    ok = V["V0_enough_seeds"] and V["V1"] and V["V2"] and V["V3"][arm] and V["V5"]
    return dict(arm=arm, ref=bood, mean=m, lo=lo, hi=hi, valid=bool(ok), verdict=decide(m, lo, hi, ok))
H["H1"] = ood("D_attn"); H["H2"] = ood("D_sup")
for arm, key in (("D_sup", "H3a"), ("D_attn", "H3b")):
    pr, rc = tab[arm]["ood_precision"], tab[arm]["ood_recall"]
    H[key] = dict(precision=pr, recall=rc, verdict=("CHƯA KẾT LUẬN (UNRESOLVED)" if not V["V0_enough_seeds"] else ("ĐẠT (PASS)" if (pr >= 0.8 and rc >= 0.8) else "KHÔNG ĐẠT (FAIL)")))
m, lo, hi = boot(rel(get("D_soft", "mse_test"), get("D_attn", "mse_test"))); H["H4"] = dict(mean=m, lo=lo, hi=hi, verdict=decide(m, lo, hi, V["V1"] and V["V5"]))
m, lo, hi = boot(rel(get("E", "mse_test"), get("A", "mse_test")))
H["H5"] = dict(mean=m, lo=lo, hi=hi, verdict="ĐẠT (PASS)" if hi <= 0.10 else ("KHÔNG ĐẠT (FAIL)" if lo > 0.10 else "CHƯA KẾT LUẬN (UNRESOLVED)"))
for arm in ("D_attn", "D_sup"):
    ue, nr = tab[arm]["id_usage_entropy"], tab[arm]["id_node_recall"]
    H["H6_" + arm] = dict(usage_entropy=ue, node_recall=nr, verdict="ĐẠT (PASS)" if (ue >= 0.8 and nr >= 0.8) else "KHÔNG ĐẠT (FAIL)")
m, lo, hi = boot(rel(get("D_attn", "mse_test"), get(bid, "mse_test"))); mr = tab["D_attn"]["macs"] / tab[bid]["macs"]
H["H7"] = dict(ref=bid, mean=m, lo=lo, hi=hi, mac_ratio=mr, verdict="ĐẠT (PASS)" if (hi <= 0.10 and mr <= 0.5) else ("KHÔNG ĐẠT (FAIL)" if (lo > 0.10 or mr > 0.5) else "CHƯA KẾT LUẬN (UNRESOLVED)"))
mech = {}
for n in ("D_attn", "D_sup"):
    for blk in ("mech_test", "mech_ood"):
        mm = [R[n][s][blk] for s in seeds]
        mech[f"{n}_{blk}"] = dict(base=float(np.mean([x["base"] for x in mm])), route_swap=float(np.mean([x["route_swap_mse"] for x in mm])),
                                  module_ablate_mean=float(np.mean([np.mean(x["ablate_module_delta"]) for x in mm])))
out = dict(seeds=seeds, screened=vs["screened"], table=tab, valid=V, H=H, best_id=bid, best_ood=bood, mechanism=mech)
json.dump(out, open(os.path.join(ROOT, "results", "results_v3.json"), "w"), indent=1, default=float)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, k, ttl in ((ax[0], "mse_test", "Trong phân phối"), (ax[1], "mse_ood", "Tuyến chưa thấy (OOD)")):
    a.bar(range(len(NAMES)), [tab[n][k][0] for n in NAMES], yerr=[tab[n][k][1] for n in NAMES]); a.set_xticks(range(len(NAMES))); a.set_xticklabels(NAMES, rotation=60, fontsize=8)
    a.axhline(tab["A"]["trivial" if k == "mse_test" else "trivial_ood"], ls="--", c="k", lw=1); a.set_title(ttl); a.set_ylabel("MSE")
arms = ["D_attn", "D_sup"]; x = np.arange(2)
ax[2].bar(x - 0.2, [tab[a]["ood_precision"] for a in arms], 0.4, label="precision"); ax[2].bar(x + 0.2, [tab[a]["ood_recall"] for a in arms], 0.4, label="recall")
ax[2].axhline(0.8, ls="--", c="k"); ax[2].set_xticks(x); ax[2].set_xticklabels(arms); ax[2].set_title("Khôi phục cạnh (OOD)"); ax[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "v3_ket_qua.png"), dpi=130)
print(len(seeds), "seed hợp lệ"); print(json.dumps(dict(valid=V, H=H), indent=1, default=float))
