m = open("models.py").read()
m = m.replace("def fwd_graph(p, x, c, cfg, gt_g=None, gt_o=None, g_override=None, h_mask=None, c_router=None):",
              "def fwd_graph(p, x, c, cfg, gt_g=None, gt_o=None, g_override=None, h_mask=None, c_router=None, mix=0.0, rkey=None, sigma=0.0):")
m = m.replace("        lg = jnp.where(valid > 0, lg, -1e9)\n        soft = jax.nn.sigmoid(lg) * valid",
"""        if rkey is not None:  # nhiễu logistic khi huấn luyện để cổng không chốt sớm
            u = jax.random.uniform(rkey, lg.shape, minval=1e-6, maxval=1 - 1e-6)
            lg = lg + sigma * (jnp.log(u) - jnp.log(1 - u))
        lg = jnp.where(valid > 0, lg, -1e9)
        soft = jax.nn.sigmoid(lg) * valid""")
m = m.replace("        G = hard + soft - jax.lax.stop_gradient(soft)\n    if g_override",
"        G = hard + soft - jax.lax.stop_gradient(soft)\n        G = mix * soft + (1 - mix) * G  # mix=1: cổng mềm dày đặc; mix=0: cổng cứng top-K\n    if g_override")
open("models.py", "w").write(m)
t = open("train.py").read()
t = t.replace("def run(kind, cfgname, seed, width, lr, epochs=60, bs=150, data=None, verbose=False):",
              "def run(kind, cfgname, seed, width, lr, epochs=60, bs=150, data=None, verbose=False, sched=None):")
t = t.replace("""    def loss_fn(p, b):
        y, aux = predict(kind, p, b, cfg, width)
        l = jnp.mean((y - b["y"]) ** 2)
        if aux is not None and cfg["mode"] in ("mlp", "fixed", "ptr"):
            l = l + LAM * aux["soft"].sum((1, 2, 3)).mean()
        return l""", """    def loss_fn(p, b, t):
        kw = {}
        lam = LAM
        if sched == "anneal" and kind in ("C", "D") and cfg["mode"] in ("mlp", "fixed", "ptr"):
            frac = t / total
            mix = jnp.clip((0.6 - frac) / 0.3, 0.0, 1.0)        # mềm hoàn toàn đến 30%, giảm dần đến 60%
            lam = LAM * jnp.clip((frac - 0.3) / 0.3, 0.0, 1.0)   # λ tăng từ 0 trong giai đoạn chuyển
            kw = dict(mix=mix, sigma=mix, rkey=jax.random.fold_in(random.PRNGKey(seed + 7), t.astype(jnp.int32)))
        y, aux = predict(kind, p, b, cfg, width, **kw)
        l = jnp.mean((y - b["y"]) ** 2)
        if aux is not None and cfg["mode"] in ("mlp", "fixed", "ptr"):
            l = l + lam * aux["soft"].sum((1, 2, 3)).mean()
        return l""")
t = t.replace("l, g = jax.value_and_grad(loss_fn)(p, b)", "l, g = jax.value_and_grad(loss_fn)(p, b, t)")
open("train.py", "w").write(t)
