"""Mô hình v4: định tuyến phân loại (categorical) với các ước lượng gradient ST / Gumbel-ST / ReinMax / REINFORCE."""
import jax, jax.numpy as jnp, numpy as np
from jax import random
from tasks import M, D, DC, L
from models import H_DIM
from models_v3 import init_attn, DK


def init_cat(key, mh=32, T=4):
    p = init_attn(key, mh=mh, T=T)
    p["Wstop"] = random.normal(random.fold_in(key, 7), (M + 1,)) * 0.3
    p["bstop"] = jnp.zeros(())
    return p


@jax.custom_vjp
def _reinmax(logits, onehot):
    return onehot


def _rm_fwd(logits, onehot):
    return onehot, (logits, onehot)


def _rm_bwd(res, g_s):
    """Hậu hướng của ReinMax (tau = 1), theo mã gốc microsoft/ReinMax (reinmax.py, ReinMax_Auto.backward)."""
    logits, onehot = res
    y = jax.nn.softmax(logits)
    shifted = 0.5 * (y + onehot)
    gi1 = (2 * g_s) * shifted
    gi1 = gi1 - shifted * gi1.sum(-1, keepdims=True)
    gi0 = (-0.5 * g_s) * y
    gi0 = gi0 - y * gi0.sum(-1, keepdims=True)
    g = gi0 + gi1
    return g - g.mean(-1, keepdims=True), jnp.zeros_like(onehot)


_reinmax.defvjp(_rm_fwd, _rm_bwd)


def pick(logits, est, key):
    """Chọn một mục phân loại. key=None: lúc đánh giá (argmax, không gradient)."""
    n = logits.shape[-1]
    if key is None:
        return jax.nn.one_hot(jnp.argmax(logits, -1), n), jnp.zeros(logits.shape[:-1])
    if est == "gs":
        z = logits - jnp.log(-jnp.log(random.uniform(key, logits.shape, minval=1e-9, maxval=1 - 1e-9)))
        oh = jax.nn.one_hot(jnp.argmax(z, -1), n); soft = jax.nn.softmax(z)
        return oh + soft - jax.lax.stop_gradient(soft), jnp.zeros(logits.shape[:-1])
    idx = random.categorical(key, jax.lax.stop_gradient(logits)); oh = jax.nn.one_hot(idx, n)
    if est == "st":
        soft = jax.nn.softmax(logits); return oh + soft - jax.lax.stop_gradient(soft), jnp.zeros(logits.shape[:-1])
    if est == "rm":
        return _reinmax(logits, oh), jnp.zeros(logits.shape[:-1])
    if est == "rf":
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits), idx[..., None], -1)[..., 0]
        return jax.lax.stop_gradient(oh), lp
    raise ValueError(est)


def fwd_cat(p, x, c, cfg, g_override=None, h_mask=None, c_router=None, rkey=None, gt_g=None, gt_o=None, o_override=None):
    """Mỗi vòng t chọn MỘT nguồn (9 mục: nút vào + 8 module) và MỘT đích (9 mục: dừng + 8 module)."""
    T, est = cfg["T"], cfg["est"]; valid = jnp.asarray(cfg["valid"]); B = x.shape[0]
    cr = c if c_router is None else c_router
    cp = cr.reshape(B, L, M + 1); e0 = jnp.broadcast_to(jnp.eye(M + 1)[0], (B, M + 1))
    nonpad = 1 - cp[:, :, 0]; nn_ = jnp.concatenate([nonpad[:, 1:], jnp.zeros((B, 1))], 1)
    last = ((nonpad * (1 - nn_))[:, :, None] * cp[:, :, 1:]).sum(1)
    o = jax.nn.softmax(last @ p["pWo"].T + p["pbo"])
    if o_override is not None: o = o_override
    Hs = jnp.zeros((B, M, H_DIM)); Gs = []; act_nodes = 0.0; logp = jnp.zeros(B)
    keys = random.split(rkey, 2 * T) if rkey is not None else [None] * (2 * T)
    for t in range(T):
        prev = e0 if t == 0 else cp[:, t - 1]; nxt = cp[:, t]
        qs = prev @ p["Qs"]; qd = nxt @ p["Qd"]
        ks = p["Es"][None] + jnp.concatenate([(x @ p["Ux"])[:, None, :], jnp.einsum("bih,hk->bik", Hs, p["Us"])], 1)
        kd = p["Ed"][None] + jnp.einsum("bjh,hk->bjk", Hs, p["Ud"])
        s_lg = jnp.einsum("bk,bik->bi", qs, ks) / np.sqrt(DK)
        d_mod = jnp.einsum("bk,bjk->bj", qd, kd) / np.sqrt(DK)
        d_lg = jnp.concatenate([(nxt @ p["Wstop"] + p["bstop"])[:, None], d_mod], 1)
        ys, lps = pick(s_lg, est, keys[2 * t]); yd, lpd = pick(d_lg, est, keys[2 * t + 1])
        logp = logp + lps + lpd
        Gt = ys[:, :, None] * yd[:, None, 1:] * valid[None]
        if g_override is not None: Gt = g_override[:, t]
        proj = jnp.einsum("bih,ihk->bik", Hs, p["Bs"])
        bus = jnp.einsum("bik,bij->bjk", proj, Gt[:, 1:, :]) + Gt[:, 0, :, None] * (x @ p["Bin"])[:, None, :]
        msg = jnp.einsum("bjk,jkl->bjl", bus, p["As"])
        a = jnp.clip(Gt[:, 0, :] + Gt[:, 1:, :].sum(1), 0, 1)
        z = jnp.tanh(jnp.einsum("bmd,mdk->bmk", jnp.concatenate([Hs, msg], -1), p["F1"][0]) + p["b1"][0])
        Hn = jnp.einsum("bmk,mkh->bmh", z, p["F2"][0]) + p["b2"][0]
        Hs = a[..., None] * Hn + (1 - a[..., None]) * Hs
        act_nodes = act_nodes + jax.lax.stop_gradient(a).sum(-1); Gs.append(Gt)
    if h_mask is not None: Hs = Hs * h_mask[:, :, None]
    y = jnp.einsum("bm,bmh,mhd->bd", o, Hs, p["Dout"]) + o @ p["bo"]
    G = jnp.stack(Gs, 1)
    return y, dict(G=G, soft=jnp.zeros_like(G), o=o, act_nodes=act_nodes, logp=logp)
