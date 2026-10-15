"""Bộ sinh nhiệm vụ tổng hợp (synthetic) với đồ thị phụ thuộc (dependency graph) thay đổi theo ca.

Mỗi ca = (x, tuyến). Đích y = f_{r_L}(...f_{r_2}(f_{r_1}(x))), với f_k(s)=tanh(W_k s + b_k).
Mô hình nhận x và bộ mô tả tuyến (descriptor) c; KHÔNG nhận trạng thái trung gian
và KHÔNG nhận cạnh nào cần bật. Tuyến thật chỉ dùng để đánh giá và cho đối chứng oracle.
"""
import numpy as np

M = 8            # số module / số phép toán
GAIN = 1.0       # độ mạnh phi tuyến của mỗi phép toán
L = 4            # độ dài tuyến tối đa
D = 8            # chiều của x và y
DC = L * (M + 1) # chiều bộ mô tả

# Tuyến OOD: mọi cặp chuyển tiếp (bigram) đều đã thấy khi huấn luyện, nhưng cả tuyến thì chưa.
OOD_ROUTES = {"O1": [1, 4, 7, 2], "O2": [1, 6, 3, 8], "O3": [4, 7, 2, 8], "O4": [3, 1, 6, 3], "O5": [7, 2, 8, 5]}


def _make_train_routes(n=40, seed=777):
    """Sinh cố định 40 tuyến huấn luyện (độ dài 3-4, không lặp module), loại mọi tuyến OOD."""
    rng = np.random.default_rng(seed)
    ood = {tuple(r) for r in OOD_ROUTES.values()}
    out = {}
    while len(out) < n:
        ln = int(rng.integers(3, 5))
        r = tuple(int(v) + 1 for v in rng.permutation(M)[:ln])
        if r in ood or r in out.values():
            continue
        out[f"R{len(out):02d}"] = r
    return {k: list(v) for k, v in out.items()}


def _bigrams(rs):
    return {(a, b) for r in rs.values() for a, b in zip(r[:-1], r[1:])}


_need = _bigrams(OOD_ROUTES)
_s = 777
while not _need <= _bigrams(_make_train_routes(seed=_s)):  # tìm hạt giống thiết kế cho phủ đủ bigram OOD
    _s += 1
TRAIN_ROUTES = _make_train_routes(seed=_s)
_seen = _bigrams(TRAIN_ROUTES)
assert all((a, b) in _seen for r in OOD_ROUTES.values() for a, b in zip(r[:-1], r[1:])), "bigram OOD chưa được thấy"


def make_world(seed):
    rng = np.random.default_rng(10_000 + seed)
    Ws, bs = [], []
    for _ in range(M):
        q, _ = np.linalg.qr(rng.normal(size=(D, D)))
        Ws.append(GAIN * q)
        bs.append(0.3 * rng.normal(size=D))
    return np.stack(Ws).astype(np.float32), np.stack(bs).astype(np.float32)


def descriptor(route):
    c = np.zeros((L, M + 1), np.float32)
    for p in range(L):
        c[p, route[p] if p < len(route) else 0] = 1.0
    return c.reshape(-1)


def gt_gates(route, T=L):
    """Cạnh thật theo từng vòng lặp: hàng 0 = nút đầu vào, hàng 1..8 = module nguồn."""
    G = np.zeros((T, M + 1, M), np.float32)
    G[0, 0, route[0] - 1] = 1
    for t in range(1, len(route)):
        G[t, route[t - 1], route[t] - 1] = 1
    o = np.zeros(M, np.float32)
    o[route[-1] - 1] = 1
    return G, o


def sample(world, routes, n_per, rng):
    Ws, bs = world
    X, Y, C, G, O, F = [], [], [], [], [], []
    for fi, (name, r) in enumerate(routes.items()):
        x = rng.normal(size=(n_per, D)).astype(np.float32)
        s = x
        for k in r:
            s = np.tanh(s @ Ws[k - 1].T + bs[k - 1])
        g, o = gt_gates(r)
        X.append(x); Y.append(s.astype(np.float32))
        C.append(np.tile(descriptor(r), (n_per, 1)))
        G.append(np.tile(g, (n_per, 1, 1, 1))); O.append(np.tile(o, (n_per, 1)))
        F.append(np.full(n_per, fi))
    cat = lambda a: np.concatenate(a, 0)
    return dict(x=cat(X), y=cat(Y), c=cat(C), g=cat(G), o=cat(O), fam=cat(F))


def build(seed, n_train=500, n_val=100, n_test=200, n_ood=400):
    world = make_world(seed)
    rng = np.random.default_rng(20_000 + seed)
    return dict(train=sample(world, TRAIN_ROUTES, n_train, rng), val=sample(world, TRAIN_ROUTES, n_val, rng),
                test=sample(world, TRAIN_ROUTES, n_test, rng), ood=sample(world, OOD_ROUTES, n_ood, rng))
