s = open("models.py").read()
i = s.index("def init_graph")
head = s[:i]
new = '''def init_graph(key, mh=32, rh=64, T=4, shared=True, router="mlp", cond_in=False):
    """Module KHÔNG nhìn thấy bộ mô tả c. c chỉ đi vào (i) bộ định tuyến, (ii) đầu đọc ra o,
    và riêng mô hình C (tĩnh) còn nhận c ở nút đầu vào để C là baseline hợp lệ (không bị tước điều kiện)."""
    ks = random.split(key, 12)
    nT = 1 if shared else T
    din = 2 * H_DIM
    dx = D + (DC if cond_in else 0)
    p = dict(Win=random.normal(ks[0], (M, dx, H_DIM)) * 0.3,
             W=random.normal(ks[1], (M, M, H_DIM, H_DIM)) * (1 / np.sqrt(H_DIM)),
             F1=random.normal(ks[2], (nT, M, din, mh)) * np.sqrt(1.0 / din), b1=jnp.zeros((nT, M, mh)),
             F2=random.normal(ks[3], (nT, M, mh, H_DIM)) * np.sqrt(1.0 / mh), b2=jnp.zeros((nT, M, H_DIM)),
             Dout=random.normal(ks[4], (M, H_DIM, D)) * np.sqrt(1.0 / H_DIM), bo=jnp.zeros((M, D)))
    if router != "ptr":
        p.update(o1=random.normal(ks[5], (DC, 64)) * np.sqrt(1.0 / DC), ob1=jnp.zeros(64),
                 o2=random.normal(ks[6], (64, M)) * 0.1, ob2=jnp.zeros(M))
    if router in ("mlp", "fixed"):
        p.update(r1=random.normal(ks[7], (DC + T, rh)) * np.sqrt(1.0 / (DC + T)), rb1=jnp.zeros(rh),
                 r2=random.normal(ks[8], (rh, (M + 1) * M)) * 0.1, rb2=jnp.zeros((M + 1) * M))
    if router == "ptr":  # bộ định tuyến con trỏ (pointer): logit(i->j, t) = A[i].c_{t-1} + B[j].c_t + b
        p.update(pA=random.normal(ks[7], (M + 1, M + 1)) * 0.3, pB=random.normal(ks[8], (M, M + 1)) * 0.3,
                 pb=jnp.zeros(()), pWo=random.normal(ks[9], (M, M)) * 0.3, pbo=jnp.zeros(M))
    return p


def fwd_graph(p, x, c, cfg, gt_g=None, gt_o=None, g_override=None, h_mask=None, c_router=None):
    """cfg: dict(T,K,mode in static/mlp/fixed/ptr/oracle, valid, shared, cond_in). Trả về (y, aux)."""
    T, K, mode = cfg["T"], cfg["K"], cfg["mode"]
    valid = jnp.asarray(cfg["valid"])
    B = x.shape[0]
    if mode != "ptr":
        o = jax.nn.softmax(jnp.tanh(c @ p["o1"] + p["ob1"]) @ p["o2"] + p["ob2"])
    soft = jnp.zeros((B, T, M + 1, M))
    cr = c if c_router is None else c_router
    if mode == "static":
        G = jnp.broadcast_to(valid, (B, T, M + 1, M))
    elif mode == "oracle":
        G, o = gt_g[:, :T], gt_o
    else:
        if mode == "fixed":
            cr = jnp.zeros_like(c)
        if mode == "ptr":
            cp = cr.reshape(B, L, M + 1)
            e0 = jnp.broadcast_to(jnp.eye(M + 1)[0], (B, M + 1))
            lgs = []
            for t in range(T):
                prev = e0 if t == 0 else cp[:, t - 1]
                nxt = cp[:, t] if t < L else e0
                lgs.append((prev @ p["pA"].T)[:, :, None] + (nxt @ p["pB"].T)[:, None, :] + p["pb"])
            lg = jnp.stack(lgs, 1)
            nonpad = 1 - cp[:, :, 0]
            nn_ = jnp.concatenate([nonpad[:, 1:], jnp.zeros((B, 1))], 1)
            last = ((nonpad * (1 - nn_))[:, :, None] * cp[:, :, 1:]).sum(1)
            o = jax.nn.softmax(last @ p["pWo"].T + p["pbo"])
        else:
            rin = jnp.concatenate([jnp.broadcast_to(cr[:, None, :], (B, T, cr.shape[-1])),
                                   jnp.broadcast_to(jnp.eye(T)[None], (B, T, T))], -1)
            lg = (jnp.tanh(rin @ p["r1"] + p["rb1"]) @ p["r2"] + p["rb2"]).reshape(B, T, M + 1, M)
        lg = jnp.where(valid > 0, lg, -1e9)
        soft = jax.nn.sigmoid(lg) * valid
        flat = lg.reshape(B, T, -1)
        kth = jax.lax.top_k(flat, K)[0][..., -1:]
        hard = ((flat >= kth) & (soft.reshape(B, T, -1) > 0.5)).astype(jnp.float32).reshape(B, T, M + 1, M)
        G = hard + soft - jax.lax.stop_gradient(soft)
    if g_override is not None:
        G = g_override
    xin = jnp.concatenate([x, c], -1) if cfg.get("cond_in") else x
    Hs = jnp.zeros((B, M, H_DIM))
    act_nodes = 0.0
    for t in range(T):
        Gt = G[:, t]
        msg = jnp.einsum("bih,ijhk,bij->bjk", Hs, p["W"], Gt[:, 1:, :]) + jnp.einsum("bx,jxk,bj->bjk", xin, p["Win"], Gt[:, 0, :])
        a = jnp.clip(Gt[:, 0, :] + Gt[:, 1:, :].sum(1), 0, 1)
        i = 0 if cfg["shared"] else t
        cat = jnp.concatenate([Hs, msg], -1)
        z = jnp.tanh(jnp.einsum("bmd,mdk->bmk", cat, p["F1"][i]) + p["b1"][i])
        Hn = jnp.einsum("bmk,mkh->bmh", z, p["F2"][i]) + p["b2"][i]
        Hs = a[..., None] * Hn + (1 - a[..., None]) * Hs
        act_nodes = act_nodes + jax.lax.stop_gradient(a).sum(-1)
    if h_mask is not None:
        Hs = Hs * h_mask[:, :, None]
    y = jnp.einsum("bm,bmh,mhd->bd", o, Hs, p["Dout"]) + o @ p["bo"]
    return y, dict(G=G, soft=soft, o=o, act_nodes=act_nodes)


def graph_macs(p, cfg, aux, mh):
    """Proxy chi phí tính toán (MAC) trên mỗi mẫu, dựa trên số cạnh/nút thực sự được kích hoạt."""
    G = jax.lax.stop_gradient(aux["G"])
    dx = D + (DC if cfg.get("cond_in") else 0)
    e_in = G[:, :, 0, :].sum((1, 2)); e_mod = G[:, :, 1:, :].sum((1, 2, 3))
    node = 2 * H_DIM * mh + mh * H_DIM
    macs = e_in * dx * H_DIM + e_mod * H_DIM * H_DIM + aux["act_nodes"] * node
    macs = macs + M * H_DIM * D
    if cfg["mode"] != "ptr":
        macs = macs + DC * 64 + 64 * M
    if cfg["mode"] in ("mlp", "fixed"):
        macs = macs + cfg["T"] * ((DC + cfg["T"]) * p["r1"].shape[1] + p["r1"].shape[1] * p["r2"].shape[1])
    if cfg["mode"] == "ptr":
        macs = macs + cfg["T"] * ((M + 1) * (M + 1) + M * (M + 1)) + M * M
    return macs
'''
s = head + new
s = s.replace("from tasks import M, D, DC", "from tasks import M, D, DC, L")
open("models.py", "w").write(s)

t = open("train.py").read()
t = t.replace('''    base = dict(T=T, K=K, mode="mlp", valid=valid_mask("full"), shared=True, router="mlp", shuffle=False)
    v = {"C": dict(mode="static", router="none"),''', '''    base = dict(T=T, K=K, mode="mlp", valid=valid_mask("full"), shared=True, router="mlp", shuffle=False, cond_in=False)
    v = {"C": dict(mode="static", router="none", cond_in=True),''')
t = t.replace('"D_fact": dict(mode="fact", router="fact"),', '"D_ptr": dict(mode="ptr", router="ptr"),')
t = t.replace('router=cfg["router"])', 'router=cfg["router"], cond_in=cfg["cond_in"])')
t = t.replace('("mlp", "fixed", "fact")', '("mlp", "fixed", "ptr")')
open("train.py", "w").write(t)
print("đã vá")
