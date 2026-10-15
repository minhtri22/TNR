"""Huấn luyện + đánh giá. Optimizer Adam tự cài, lịch cosine, lặp theo epoch bằng lax.scan."""
import json, time, numpy as np, jax, jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment
import tasks as TK
from models import *
from models_v3 import *
from models_v4 import *
ROUTED = ("mlp", "fixed", "ptr", "attn", "sup", "soft", "cat")
GAMMA = 0.1   # trọng số tổn thất giám sát bố cục (chỉ D_sup)
BAL = 0.01    # trọng số cân bằng tải MoE
BETA = 0.05   # trọng số phạt entropy đầu đọc (D_trueG_ent)

LAM = 2e-4  # hệ số chi phí định tuyến (preregistered)


def make_cfg(name, T=4, K=2, seed=0):
    base = dict(T=T, K=K, mode="mlp", valid=valid_mask("full"), shared=True, router="mlp", shuffle=False, cond_in=False, est=None)
    v = {"C": dict(mode="static", router="none", cond_in=True),
         "D": {},
         "D_fixed": dict(mode="fixed", router="fixed"),
         "D_ptr": dict(mode="ptr", router="ptr"),
         "D_ring": dict(valid=valid_mask("ring")),
         "D_rand50": dict(valid=valid_mask("random50", seed)),
         "D_unshared": dict(shared=False),
         "D_T1": dict(T=1),
         "D_shuffle": dict(shuffle=True),
         "D_oracle": dict(mode="oracle", router="none"),
         "D_attn": dict(mode="attn", router="attn"), "D_sup": dict(mode="sup", router="attn"),
         "D_soft": dict(mode="soft", router="attn"),
         "D_sup_trueo": dict(mode="sup", router="attn", true_o=True), "D_trueG": dict(mode="attn", router="attn", true_g=True),
         "D_trueG_oh": dict(mode="attn", router="attn", true_g=True, ro="oh"), "D_trueG_ent": dict(mode="attn", router="attn", true_g=True, ent=True),
         "D_sup_trueo_sg": dict(mode="sup", router="attn", true_o=True, sg=True), "D_sup_sg_oh": dict(mode="sup", router="attn", sg=True, ro="oh"),
         "D_trueG_id": dict(mode="attn", router="attn", true_g=True, ro_init="id"),
         "D_trueG_bal": dict(mode="attn", router="attn", true_g=True, ro_init="bal"), "D_trueG_tau": dict(mode="attn", router="attn", true_g=True, ro="tau"),
         "D_trueG_rm": dict(mode="attn", router="attn", true_g=True, ro="rm"),
         "D_e2e_id": dict(mode="sup", router="attn", sg=True, ro_init="id"), "D_e2e_bal": dict(mode="sup", router="attn", sg=True, ro_init="bal"),
         "D_e2e_tau": dict(mode="sup", router="attn", sg=True, ro="tau"), "D_e2e_rm": dict(mode="sup", router="attn", sg=True, ro="rm"),
         "D_e2e_id_g50": dict(mode="sup", router="attn", sg=True, ro_init="id", gfrac=0.5), "D_e2e_id_g25": dict(mode="sup", router="attn", sg=True, ro_init="id", gfrac=0.25),
         "D_e2e_id_g10": dict(mode="sup", router="attn", sg=True, ro_init="id", gfrac=0.1),
         "D_cat_st": dict(mode="cat", router="cat", est="st"), "D_cat_gs": dict(mode="cat", router="cat", est="gs"),
         "D_cat_rm": dict(mode="cat", router="cat", est="rm"), "D_cat_rf": dict(mode="cat", router="cat", est="rf"), "E": dict(mode="moe", router="none")}[name]
    base.update(v)
    return base


def init_model(kind, key, width, cfg=None):
    if kind == "E": return init_moe(key, mh=width, T=cfg["T"])
    if cfg is not None and cfg.get("router") == "cat": return init_cat(key, mh=width, T=cfg["T"])
    if cfg is not None and cfg.get("router") == "attn":
        p = init_attn(key, mh=width, T=cfg["T"])
        if cfg.get("ro_init") == "id":   # đầu đọc học khởi tạo đúng (đơn vị nhọn) — chẩn đoán v5
            p["pWo"] = 6.0 * jnp.eye(M); p["pbo"] = jnp.zeros(M)
        if cfg.get("ro_init") == "bal":  # v6: khởi tạo cân bằng (đồng đều, không thiên vị module nào)
            p["pWo"] = jnp.zeros((M, M)); p["pbo"] = jnp.zeros(M)
        return p
    if kind == "A": return init_mlp(key, width)
    if kind == "B": return init_res(key, width)
    return init_graph(key, mh=width, T=cfg["T"], shared=cfg["shared"], router=cfg["router"], cond_in=cfg["cond_in"])


def count_shape(kind, w, cfg):
    """Đếm tham số bằng eval_shape (không cấp phát thật) để khớp ngân sách nhanh."""
    f = lambda: init_model(kind, random.PRNGKey(0), w, cfg)
    return nparams(jax.eval_shape(f))


def match_widths(target):
    """Chọn bề rộng để số tham số khớp mô hình D (mục tiêu)."""
    out = {}; cC = make_cfg("C")
    for kind in ("A", "B", "C"):
        best = min(range(8, 400), key=lambda w: abs(count_shape(kind, w, cC) - target))
        out[kind] = (best, count_shape(kind, best, cC))
    return out


def predict(kind, p, d, cfg, width, idx=None, **kw):
    x, c = d["x"], d["c"]
    if kind == "A": return fwd_mlp(p, x, c), None
    if kind == "B": return fwd_res(p, x, c), None
    if kind == "E": return fwd_moe(p, x, c, cfg["K"], cfg["T"])
    if cfg["mode"] == "cat":
        return fwd_cat(p, x, c, cfg, gt_g=d["g"], gt_o=d["o"], **{k: v for k, v in kw.items() if k in ("g_override", "h_mask", "c_router", "rkey", "o_override")})
    if cfg["mode"] in ("attn", "sup", "soft"):
        return fwd_attn(p, x, c, cfg, gt_g=d["g"], gt_o=d["o"], **{k: v for k, v in kw.items() if k in ("g_override", "h_mask", "c_router", "o_override", "tau", "rkey")})
    return fwd_graph(p, x, c, cfg, gt_g=d["g"], gt_o=d["o"], **kw)


def run(kind, cfgname, seed, width, lr, epochs=60, bs=150, data=None, verbose=False, sched=None):
    cfg = make_cfg(cfgname if kind in ("C", "D", "E") else "D", seed=seed)
    data = data or TK.build(seed)
    tr = {k: jnp.asarray(v) for k, v in data["train"].items()}
    if cfg["shuffle"]:  # đối chứng: phá quan hệ giữa bộ mô tả và đích
        perm = np.random.default_rng(seed).permutation(len(tr["c"]))
        tr["c"] = tr["c"][perm]; tr["g"] = tr["g"][perm]; tr["o"] = tr["o"][perm]
    if cfg.get("gfrac") is not None:   # v6: chỉ một phần TUYẾN huấn luyện có nhãn cổng
        fams = np.unique(np.asarray(tr["fam"])); rs = np.random.default_rng(seed + 99)
        chosen = rs.permutation(fams)[: max(1, int(round(cfg["gfrac"] * len(fams))))]
        tr["gm"] = jnp.asarray(np.isin(np.asarray(tr["fam"]), chosen).astype(np.float32))
    else:
        tr["gm"] = jnp.ones(tr["x"].shape[0])
    key = random.PRNGKey(seed)
    p = init_model(kind, key, width, cfg)
    n = tr["x"].shape[0]; nb = n // bs; total = epochs * nb

    def loss_fn(p, b, t):
        kw = {}
        lam = LAM
        if sched == "anneal" and kind in ("C", "D") and cfg["mode"] in ("mlp", "fixed", "ptr"):
            frac = t / total
            mix = jnp.clip((0.6 - frac) / 0.3, 0.0, 1.0)        # mềm hoàn toàn đến 30%, giảm dần đến 60%
            lam = LAM * jnp.clip((frac - 0.3) / 0.3, 0.0, 1.0)   # λ tăng từ 0 trong giai đoạn chuyển
            kw = dict(mix=mix, sigma=mix, rkey=jax.random.fold_in(random.PRNGKey(seed + 7), t.astype(jnp.int32)))
        if cfg["mode"] == "cat":
            kw = dict(rkey=jax.random.fold_in(random.PRNGKey(seed + 11), t.astype(jnp.int32)))
        if cfg.get("ro") == "tau":   # nhiệt độ giảm tuyến tính 1,0 -> 0,1 trong 70% đầu huấn luyện (v6)
            kw = dict(tau=jnp.maximum(0.1, 1.0 - (t / total) / 0.7 * 0.9))
        if cfg.get("ro") == "rm":
            kw = dict(rkey=jax.random.fold_in(random.PRNGKey(seed + 23), t.astype(jnp.int32)))
        y, aux = predict(kind, p, b, cfg, width, **kw)
        l = jnp.mean((y - b["y"]) ** 2)
        if cfg["mode"] == "cat" and cfg["est"] == "rf":   # REINFORCE: hạng ưu thế = tổn thất mẫu − trung bình lô
            pe = ((y - b["y"]) ** 2).mean(-1); adv = jax.lax.stop_gradient(pe - pe.mean())
            l = l + (adv * aux["logp"]).mean()
        if aux is not None and cfg["mode"] in ROUTED:
            l = l + lam * aux["soft"].sum((1, 2, 3)).mean()
        if cfg["mode"] == "sup" and kind == "D":
            lg = aux["lg"]; l = l + GAMMA * (((jax.nn.softplus(lg) - lg * b["g"]).sum((1, 2, 3))) * b["gm"]).mean()
        if cfg.get("ent") and aux is not None:
            o_ = aux["o"]; l = l + BETA * (-(o_ * jnp.log(o_ + 1e-9)).sum(-1)).mean()
        if kind == "E":
            l = l + BAL * (M * (aux["probs"].mean(0) ** 2).sum(-1)).mean()
        return l

    m0 = jax.tree_util.tree_map(jnp.zeros_like, p)

    def step(carry, inp):
        p, m, v, t = carry
        b = inp
        l, g = jax.value_and_grad(loss_fn)(p, b, t)
        t = t + 1
        cur = lr * 0.5 * (1 + jnp.cos(jnp.pi * t / total))
        m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
        p = jax.tree_util.tree_map(lambda p, m, v: p - cur * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return (p, m, v, t), l

    @jax.jit
    def epoch(p, m, v, t, key):
        perm = random.permutation(key, n)[: nb * bs].reshape(nb, bs)
        batches = {k: val[perm] for k, val in tr.items()}
        (p, m, v, t), ls = jax.lax.scan(step, (p, m, v, t), batches)
        return p, m, v, t, ls.mean()

    m, v, t = m0, jax.tree_util.tree_map(jnp.zeros_like, p), jnp.array(0.0)
    t0 = time.time(); curve = []
    for e in range(epochs):
        key, k = random.split(key)
        p, m, v, t, l = epoch(p, m, v, t, k)
        curve.append(float(l))
        if verbose and e % 10 == 0: print(e, float(l), flush=True)
    return dict(p=p, cfg=cfg, kind=kind, width=width, curve=curve, secs=time.time() - t0, data=data, seed=seed)


def mse(y, t): return float(jnp.mean((y - t) ** 2))


def macs_of(R, d):
    kind, p, cfg, width = R["kind"], R["p"], R["cfg"], R["width"]
    if kind in ("A", "B"):
        return float(nparams(p) - sum(0 for _ in []))  # MAC ~ số trọng số (bỏ qua bias)
    if kind == "E":
        return float(moe_macs(width, K=cfg["K"], T=cfg["T"]))
    y, aux = predict(kind, p, d, cfg, width)
    if cfg["mode"] == "soft":  # cổng mềm dày đặc: chi phí như đồ thị đầy đủ
        aux = dict(aux); aux["G"] = jnp.broadcast_to(jnp.asarray(cfg["valid"]), aux["G"].shape); aux["act_nodes"] = jnp.full_like(aux["act_nodes"], cfg["T"] * 8.0)
    m = graph_macs(p, cfg, aux, width).mean()
    if cfg["mode"] in ("attn", "sup", "soft", "cat"): m = m + attn_router_macs(cfg["T"])
    return float(m)


def align_perm(R):
    """Ước lượng phép hoán vị module->phép toán chỉ từ dữ liệu huấn luyện (công bằng với OOD)."""
    d = {k: jnp.asarray(v) for k, v in R["data"]["train"].items()}
    y, aux = predict("D", R["p"], d, R["cfg"], R["width"])
    G = np.asarray(aux["G"]); gt = np.asarray(d["g"])
    # đếm đồng xuất hiện giữa module đích thật k và module đích được bật j
    tgt_pred = G.sum(2); tgt_gt = gt.sum(2)  # (B,T,M)
    cnt = np.einsum("btk,btj->kj", tgt_gt, tgt_pred)
    r, c = linear_sum_assignment(-cnt)
    perm = np.zeros(TK.M, int); perm[r] = c  # op k -> module perm[k]
    return perm


def route_metrics(R, d, perm):
    y, aux = predict("D", R["p"], {k: jnp.asarray(v) for k, v in d.items()}, R["cfg"], R["width"])
    G = np.asarray(aux["G"]) > 0.5; gt = np.asarray(d["g"]) > 0.5
    Gp = np.zeros_like(gt)  # ánh xạ module -> op theo hoán vị: module j tương ứng op k nếu perm[k]=j
    inv = np.argsort(perm)
    # hàng 0 = đầu vào; hàng 1..8 = nguồn; cột = đích
    Gp[:, :, 0, :] = G[:, :, 0, :][..., perm]
    Gp[:, :, 1:, :] = G[:, :, 1:, :][:, :, perm, :][..., perm]
    tp = (Gp & gt).sum((1, 2, 3)); act = Gp.sum((1, 2, 3)); n_gt = gt.sum((1, 2, 3))
    soft = np.asarray(aux["soft"]); ent = -(soft * np.log(soft + 1e-9) + (1 - soft) * np.log(1 - soft + 1e-9))
    mask = np.asarray(R["cfg"]["valid"]) > 0
    tgt = Gp.sum(2) > 0; gtt = gt.sum(2) > 0          # nút đích được bật theo từng vòng
    ntp = (tgt & gtt).sum((1, 2))
    use = G.sum((1, 2)).sum(0).astype(float) + 1e-9; use = use / use.sum()   # phân bố sử dụng module (theo module thật, chưa hoán vị)
    ue = float(-(use * np.log(use)).sum() / np.log(len(use)))
    return dict(node_recall=float((ntp / np.maximum(gtt.sum((1, 2)), 1)).mean()), node_precision=float((ntp / np.maximum(tgt.sum((1, 2)), 1)).mean()),
                usage_entropy=ue, precision=float((tp / np.maximum(act, 1)).mean()), recall=float((tp / n_gt).mean()),
                exact=float((Gp == gt).all((1, 2, 3)).mean()), unnecessary=float(((act - tp) / np.maximum(act, 1)).mean()),
                edges_per_sample=float(act.mean()), gate_entropy=float(ent[:, :, mask].mean()))


def mechanism(R, d, perm):
    """Cắt bỏ cạnh/module, hoán đổi tuyến (route intervention)."""
    dj = {k: jnp.asarray(v) for k, v in d.items()}
    base_y, aux = predict("D", R["p"], dj, R["cfg"], R["width"]); base = mse(base_y, dj["y"])
    G = aux["G"]; out = dict(base=base)
    # cắt từng cạnh đang bật (trung bình trên các cạnh bật)
    deltas = []
    Gn = np.asarray(G) > 0.5
    idxs = np.argwhere(Gn[0])  # tuyến cố định theo họ -> lấy mẫu từ từng ca thực
    # dùng cắt theo từng vòng lặp t: tắt toàn bộ cạnh ở vòng t
    per_iter = []
    for t in range(R["cfg"]["T"]):
        Gm = np.array(G); Gm[:, t] = 0
        y, _ = predict("D", R["p"], dj, R["cfg"], R["width"], g_override=jnp.asarray(Gm))
        per_iter.append(mse(y, dj["y"]) - base)
    out["ablate_iter_delta"] = per_iter
    # cắt module m (đưa trạng thái cuối về 0)
    mod = []
    for m_ in range(TK.M):
        hm = np.ones((len(dj["x"]), TK.M), np.float32); hm[:, m_] = 0
        y, _ = predict("D", R["p"], dj, R["cfg"], R["width"], h_mask=jnp.asarray(hm))
        mod.append(mse(y, dj["y"]) - base)
    out["ablate_module_delta"] = mod
    # hoán đổi tuyến: bộ định tuyến thấy bộ mô tả của ca khác
    perm_i = np.roll(np.arange(len(dj["x"])), len(dj["x"]) // 3)
    y, _ = predict("D", R["p"], dj, R["cfg"], R["width"], c_router=dj["c"][perm_i])
    out["route_swap_mse"] = mse(y, dj["y"])
    return out


def evaluate(R, full=False):
    d = R["data"]; kind = R["kind"]; res = dict(kind=kind, cfg=R.get("name"), width=R["width"], seed=R["seed"])
    res["params"] = nparams(R["p"]); res["secs"] = R["secs"]; res["final_train_loss"] = R["curve"][-1]
    triv = float(np.mean((d["test"]["y"] - d["train"]["y"].mean(0)) ** 2)); res["trivial_mse"] = triv
    for nm in ("train", "val", "test", "ood"):
        dj = {k: jnp.asarray(v) for k, v in d[nm].items()}
        y, _ = predict(kind, R["p"], dj, R["cfg"], R["width"])
        res[f"mse_{nm}"] = mse(y, dj["y"])
    res["trivial_ood"] = float(np.mean((d["ood"]["y"] - d["train"]["y"].mean(0)) ** 2))
    res["macs_test"] = macs_of(R, {k: jnp.asarray(v) for k, v in d["test"].items()})
    res["macs_ood"] = macs_of(R, {k: jnp.asarray(v) for k, v in d["ood"].items()})
    res["finite"] = bool(np.isfinite(res["mse_ood"]) and np.isfinite(res["mse_test"]))
    if R.get("name") == "D_sup":   # can thiệp lúc đánh giá: thay cổng / đầu đọc đã học bằng bản đúng (mô tả, v4)
        iv = {}
        for nm in ("test", "ood"):
            dj = {k: jnp.asarray(v) for k, v in d[nm].items()}
            for tag, kw in (("gate_true", dict(g_override=dj["g"])), ("read_true", dict(o_override=dj["o"])), ("both_true", dict(g_override=dj["g"], o_override=dj["o"]))):
                y, _ = predict("D", R["p"], dj, R["cfg"], R["width"], **kw); iv[f"{tag}_{nm}"] = mse(y, dj["y"])
        res["interv"] = iv
    if kind in ("C", "D") and R["cfg"]["mode"] in ROUTED:
        perm = align_perm(R); res["perm"] = perm.tolist()
        res["route_test"] = route_metrics(R, d["test"], perm); res["route_ood"] = route_metrics(R, d["ood"], perm)
        if full:
            res["mech_test"] = mechanism(R, d["test"], perm); res["mech_ood"] = mechanism(R, d["ood"], perm)
    elif kind in ("C", "D") and R["cfg"]["mode"] not in ("moe",):
        y, aux = predict(kind, R["p"], {k: jnp.asarray(v) for k, v in d["ood"].items()}, R["cfg"], R["width"])
        res["edges_per_sample_ood"] = float((np.asarray(aux["G"]) > 0.5).sum((1, 2, 3)).mean())
    return res
