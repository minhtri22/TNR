s = open("models.py").read()
# tham số hóa cạnh theo "bus" chung: msg_j = A_j * sum_i g_ij B_i h_i ; nút vào có B_in chung
s = s.replace('''    p = dict(Win=random.normal(ks[0], (M, dx, H_DIM)) * 0.3,
             W=random.normal(ks[1], (M, M, H_DIM, H_DIM)) * (1 / np.sqrt(H_DIM)),''',
'''    p = dict(Bin=random.normal(ks[0], (dx, H_DIM)) * np.sqrt(1.0 / dx),
             Bs=random.normal(ks[1], (M, H_DIM, H_DIM)) * np.sqrt(1.0 / H_DIM),
             As=random.normal(ks[10], (M, H_DIM, H_DIM)) * np.sqrt(1.0 / H_DIM),''')
s = s.replace('''        msg = jnp.einsum("bih,ijhk,bij->bjk", Hs, p["W"], Gt[:, 1:, :]) + jnp.einsum("bx,jxk,bj->bjk", xin, p["Win"], Gt[:, 0, :])''',
'''        proj = jnp.einsum("bih,ihk->bik", Hs, p["Bs"])
        bus = jnp.einsum("bik,bij->bjk", proj, Gt[:, 1:, :]) + Gt[:, 0, :, None] * (xin @ p["Bin"])[:, None, :]
        msg = jnp.einsum("bjk,jkl->bjl", bus, p["As"])''')
s = s.replace("e_in * dx * H_DIM + e_mod * H_DIM * H_DIM + aux[\"act_nodes\"] * node", "aux_in * dx * H_DIM / jnp.maximum(aux_in, 1) * (aux_in > 0) + (e_in + e_mod) * 0 + e_mod * H_DIM * H_DIM + (e_in + e_mod) * H_DIM * H_DIM + aux[\"act_nodes\"] * node")
open("models.py", "w").write(s)
