"""Mô hình v3: bộ định tuyến theo nội dung (attention), biến thể cổng mềm, giám sát bố cục, và baseline MoE."""
import jax, jax.numpy as jnp, numpy as np
from jax import random
from tasks import M, D, DC, L
from models import H_DIM, init_graph

DK = 16


def init_attn(key, mh=32, T=4):
    p = init_graph(key, mh=mh, T=T, shared=True, router="ptr")
    p.pop("pA"); p.pop("pB")
    ks = random.split(random.fold_in(key, 99), 8)
    p.update(Qs=random.normal(ks[0], (M + 1, DK)) * 0.3, Qd=random.normal(ks[1], (M + 1, DK)) * 0.3,
             Es=random.normal(ks[2], (M + 1, DK)) * 0.3, Ed=random.normal(ks[3], (M, DK)) * 0.3,
             Us=random.normal(ks[4], (H_DIM, DK)) * 0.1, Ud=random.normal(ks[5], (H_DIM, DK)) * 0.1,
             Ux=random.normal(ks[6], (D, DK)) * 0.1)
    return p


def fwd_attn(p, x, c, cfg, g_override=None, h_mask=None, c_router=None, gt_g=None, gt_o=None, o_override=None, tau=None, rkey=None):
    """Bộ định tuyến attention: logit(i->j,t) = <q_src(c_{t-1}), k_i(h_i)> + <q_dst(c_t), k_j(h_j)> + b.
    Khóa k phụ thuộc trạng thái module (định tuyến theo nội dung). mode: attn (top-K cứng), sup (như attn, có
    thêm tổn thất giám sát khi huấn luyện), soft (cổng mềm dày đặc, không top-K)."""
    T, K, mode = cfg["T"], cfg["K"], cfg["mode"]
    valid = jnp.asarray(cfg["valid"]); B = x.shape[0]
    cr = c if c_router is None else c_router
    cp = cr.reshape(B, L, M + 1)
    e0 = jnp.broadcast_to(jnp.eye(M + 1)[0], (B, M + 1))
    nonpad = 1 - cp[:, :, 0]
    nn_ = jnp.concatenate([nonpad[:, 1:], jnp.zeros((B, 1))], 1)
    last = ((nonpad * (1 - nn_))[:, :, None] * cp[:, :, 1:]).sum(1)
    lo = last @ p["pWo"].T + p["pbo"]
    o = jax.nn.softmax(lo)
    if cfg.get("ro") == "oh":   # đầu đọc học nhưng ép one-hot (xuyên thẳng) — v5
        o = jax.nn.one_hot(jnp.argmax(lo, -1), M) + o - jax.lax.stop_gradient(o)
    if cfg.get("ro") == "tau":  # v6: nhiệt độ giảm dần; tau=None (đánh giá) -> argmax cứng
        o = jax.nn.one_hot(jnp.argmax(lo, -1), M) if tau is None else jax.nn.softmax(lo / tau)
    if cfg.get("ro") == "rm":   # v6: ReinMax cho đầu đọc phân loại; không có khóa (đánh giá) -> argmax
        from models_v4 import pick
        o = pick(lo, "rm", rkey)[0]
    if cfg.get("true_o"): o = gt_o            # chẩn đoán: đầu đọc đúng (v4)
    if o_override is not None: o = o_override
    Hs = jnp.zeros((B, M, H_DIM)); Gs, lgs, softs = [], [], []; act_nodes = 0.0
    for t in range(T):
        prev = e0 if t == 0 else cp[:, t - 1]; nxt = cp[:, t]
        qs = prev @ p["Qs"]; qd = nxt @ p["Qd"]
        ks = p["Es"][None] + jnp.concatenate([(x @ p["Ux"])[:, None, :], jnp.einsum("bih,hk->bik", Hs, p["Us"])], 1)
        kd = p["Ed"][None] + jnp.einsum("bjh,hk->bjk", Hs, p["Ud"])
        s = jnp.einsum("bk,bik->bi", qs, ks) / np.sqrt(DK); d = jnp.einsum("bk,bjk->bj", qd, kd) / np.sqrt(DK)
        lg = s[:, :, None] + d[:, None, :] + p["pb"]
        lg = jnp.where(valid > 0, lg, -1e9); soft = jax.nn.sigmoid(lg) * valid
        if mode == "soft":
            Gt = soft
        else:
            flat = lg.reshape(B, -1); kth = jax.lax.top_k(flat, K)[0][:, -1:]
            hard = ((flat >= kth) & (soft.reshape(B, -1) > 0.5)).astype(jnp.float32).reshape(B, M + 1, M)
            Gt = hard if cfg.get("sg") else hard + soft - jax.lax.stop_gradient(soft)   # sg: dừng gradient MSE vào cổng (v5)
        if cfg.get("true_g"): Gt = gt_g[:, t]    # chẩn đoán: cổng đúng (v4)
        if g_override is not None:
            Gt = g_override[:, t]
        proj = jnp.einsum("bih,ihk->bik", Hs, p["Bs"])
        bus = jnp.einsum("bik,bij->bjk", proj, Gt[:, 1:, :]) + Gt[:, 0, :, None] * (x @ p["Bin"])[:, None, :]
        msg = jnp.einsum("bjk,jkl->bjl", bus, p["As"])
        a = jnp.clip(Gt[:, 0, :] + Gt[:, 1:, :].sum(1), 0, 1)
        cat = jnp.concatenate([Hs, msg], -1)
        z = jnp.tanh(jnp.einsum("bmd,mdk->bmk", cat, p["F1"][0]) + p["b1"][0])
        Hn = jnp.einsum("bmk,mkh->bmh", z, p["F2"][0]) + p["b2"][0]
        Hs = a[..., None] * Hn + (1 - a[..., None]) * Hs
        act_nodes = act_nodes + jax.lax.stop_gradient(a).sum(-1)
        Gs.append(Gt); lgs.append(lg); softs.append(soft)
    if h_mask is not None:
        Hs = Hs * h_mask[:, :, None]
    y = jnp.einsum("bm,bmh,mhd->bd", o, Hs, p["Dout"]) + o @ p["bo"]
    return y, dict(G=jnp.stack(Gs, 1), soft=jnp.stack(softs, 1), lg=jnp.stack(lgs, 1), o=o, act_nodes=act_nodes)


def attn_router_macs(T):
    return T * ((2 * M + 1) * H_DIM * DK + 2 * (M + 1) * DK + (M + 1) * M)


# ---------- baseline MoE (chuyên gia dùng chung qua T vòng lặp, top-2) ----------
DM = 16


def init_moe(key, mh=64, rh=64, T=4):
    ks = random.split(key, 8)
    return dict(Wi=random.normal(ks[0], (D, DM)) * np.sqrt(1.0 / D), bi=jnp.zeros(DM),
                E1=random.normal(ks[1], (M, DM, mh)) * np.sqrt(2.0 / DM), eb1=jnp.zeros((M, mh)),
                E2=random.normal(ks[2], (M, mh, DM)) * np.sqrt(1.0 / mh) * 0.5, eb2=jnp.zeros((M, DM)),
                r1=random.normal(ks[3], (DC + T + DM, rh)) * np.sqrt(1.0 / (DC + T + DM)), rb1=jnp.zeros(rh),
                r2=random.normal(ks[4], (rh, M)) * 0.1, rb2=jnp.zeros(M),
                Wo=random.normal(ks[5], (DM, D)) * np.sqrt(1.0 / DM), bo=jnp.zeros(D))


def fwd_moe(p, x, c, K=2, T=4):
    B = x.shape[0]; s = x @ p["Wi"] + p["bi"]; probs = []
    for t in range(T):
        rin = jnp.concatenate([c, jnp.broadcast_to(jnp.eye(T)[t], (B, T)), s], -1)
        lg = jnp.tanh(rin @ p["r1"] + p["rb1"]) @ p["r2"] + p["rb2"]
        v, idx = jax.lax.top_k(lg, K); g = jax.nn.softmax(v)
        h = jax.nn.gelu(jnp.einsum("bd,mdk->bmk", s, p["E1"]) + p["eb1"])
        out = jnp.einsum("bmk,mkd->bmd", h, p["E2"]) + p["eb2"]
        sel = jnp.take_along_axis(out, idx[:, :, None], axis=1)
        s = s + (g[:, :, None] * sel).sum(1)
        probs.append(jax.nn.softmax(lg))
    return s @ p["Wo"] + p["bo"], dict(probs=jnp.stack(probs, 1))


def moe_macs(mh, rh=64, K=2, T=4):
    return D * DM + T * ((DC + T + DM) * rh + rh * M + K * 2 * DM * mh) + DM * D
