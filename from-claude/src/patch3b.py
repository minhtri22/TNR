t = open("train.py").read()
t = t.replace("from models import *", "from models import *\nfrom models_v3 import *\nROUTED = (\"mlp\", \"fixed\", \"ptr\", \"attn\", \"sup\", \"soft\")\nGAMMA = 0.1   # trọng số tổn thất giám sát bố cục (chỉ D_sup)\nBAL = 0.01    # trọng số cân bằng tải MoE")
t = t.replace('''         "D_oracle": dict(mode="oracle", router="none")}[name]''', '''         "D_oracle": dict(mode="oracle", router="none"),
         "D_attn": dict(mode="attn", router="attn"), "D_sup": dict(mode="sup", router="attn"),
         "D_soft": dict(mode="soft", router="attn"), "E": dict(mode="moe", router="none")}[name]''')
t = t.replace('    if kind == "A": return init_mlp(key, width)', '    if kind == "E": return init_moe(key, mh=width, T=cfg["T"])\n    if cfg is not None and cfg.get("router") == "attn": return init_attn(key, mh=width, T=cfg["T"])\n    if kind == "A": return init_mlp(key, width)')
t = t.replace('    return fwd_graph(p, x, c, cfg, gt_g=d["g"], gt_o=d["o"], **kw)', '''    if kind == "E": return fwd_moe(p, x, c, cfg["K"], cfg["T"])
    if cfg["mode"] in ("attn", "sup", "soft"):
        return fwd_attn(p, x, c, cfg, **{k: v for k, v in kw.items() if k in ("g_override", "h_mask", "c_router")})
    return fwd_graph(p, x, c, cfg, gt_g=d["g"], gt_o=d["o"], **kw)''')
t = t.replace('''        if aux is not None and cfg["mode"] in ("mlp", "fixed", "ptr"):
            l = l + lam * aux["soft"].sum((1, 2, 3)).mean()
        return l''', '''        if aux is not None and cfg["mode"] in ROUTED:
            l = l + lam * aux["soft"].sum((1, 2, 3)).mean()
        if cfg["mode"] == "sup" and kind == "D":
            lg = aux["lg"]; l = l + GAMMA * (jax.nn.softplus(lg) - lg * b["g"]).sum((1, 2, 3)).mean()
        if kind == "E":
            l = l + BAL * (M * (aux["probs"].mean(0) ** 2).sum(-1)).mean()
        return l''')
t = t.replace('''    if kind in ("A", "B"):
        return float(nparams(p) - sum(0 for _ in []))  # MAC ~ số trọng số (bỏ qua bias)
    y, aux = predict(kind, p, d, cfg, width)
    return float(graph_macs(p, cfg, aux, width).mean())''', '''    if kind in ("A", "B"):
        return float(nparams(p) - sum(0 for _ in []))  # MAC ~ số trọng số (bỏ qua bias)
    if kind == "E":
        return float(moe_macs(width, K=cfg["K"], T=cfg["T"]))
    y, aux = predict(kind, p, d, cfg, width)
    if cfg["mode"] == "soft":  # cổng mềm dày đặc: chi phí như đồ thị đầy đủ
        aux = dict(aux); aux["G"] = jnp.broadcast_to(jnp.asarray(cfg["valid"]), aux["G"].shape); aux["act_nodes"] = jnp.full_like(aux["act_nodes"], cfg["T"] * 8.0)
    m = graph_macs(p, cfg, aux, width).mean()
    if cfg["mode"] in ("attn", "sup", "soft"): m = m + attn_router_macs(cfg["T"])
    return float(m)''')
# route_metrics: thêm chỉ số nút và entropy sử dụng module
t = t.replace('''    return dict(precision=float((tp / np.maximum(act, 1)).mean()), recall=float((tp / n_gt).mean()),''', '''    tgt = Gp.sum(2) > 0; gtt = gt.sum(2) > 0          # nút đích được bật theo từng vòng
    ntp = (tgt & gtt).sum((1, 2))
    use = G.sum((1, 2)).sum(0).astype(float) + 1e-9; use = use / use.sum()   # phân bố sử dụng module (theo module thật, chưa hoán vị)
    ue = float(-(use * np.log(use)).sum() / np.log(len(use)))
    return dict(node_recall=float((ntp / np.maximum(gtt.sum((1, 2)), 1)).mean()), node_precision=float((ntp / np.maximum(tgt.sum((1, 2)), 1)).mean()),
                usage_entropy=ue, precision=float((tp / np.maximum(act, 1)).mean()), recall=float((tp / n_gt).mean()),''')
t = t.replace('if kind in ("C", "D") and R["cfg"]["mode"] in ("mlp", "fixed", "ptr"):', 'if kind in ("C", "D") and R["cfg"]["mode"] in ROUTED:')
t = t.replace('''    elif kind in ("C", "D"):
        y, aux = predict(kind''', '''    elif kind in ("C", "D") and R["cfg"]["mode"] not in ("moe",):
        y, aux = predict(kind''')
open("train.py", "w").write(t)
print("ok")
