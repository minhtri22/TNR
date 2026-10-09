#!/usr/bin/env python3
"""
Phase E — Algebraic composition (họ tiếp theo sau discrete)

Giả thuyết quick win:
  D2 fail vì module MLP tự do không khớp group action.
  Module parameter hóa Givens+scale (cùng họ op) + router chọn op
  → PureC_struct thắng HR trên group-action task.

So sánh:
  HR: residual MLP
  PureC_MLP: modules MLP (như D2 — baseline kiến trúc cũ)
  PureC_Struct: mỗi module = learnable Givens plane + scale (đúng class algebraic)

Preregister:
  PASS nếu Struct_OOD < 0.95*HR AND Struct_OOD <= 0.98*MLP_OOD AND wins_vs_HR >= 2/3
  FAIL nếu Struct_OOD > 1.05*HR
  else UNRESOLVED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEEDS = [42, 7, 123]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_OPS = 6
STATE_DIM = 8
HINT_DIM = 6
INPUT_DIM = STATE_DIM + HINT_DIM
OUTPUT_DIM = 4
HIDDEN = 48
N_MOD = 6
SEQ_LEN = 3
KV_DIM = 24
BATCH_SIZE = 128
N_TRAIN = 3200
N_VAL = 500
N_OOD = 700
EPOCHS = 26
LR = 1e-3
TEMP_START, TEMP_END = 1.3, 0.35

TRAIN_PATHS = {
    0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5],
    4: [4, 0], 5: [5, 1], 6: [0, 4], 7: [1, 5],
}
OOD_PATHS = {
    0: [0, 2, 4], 1: [1, 3, 5], 2: [2, 4, 1],
    3: [3, 5, 0], 4: [4, 0, 2], 5: [5, 1, 3],
    6: [0, 3, 1], 7: [2, 5, 4],
}

def build_ops():
    torch.manual_seed(19)
    ops = []
    for i in range(N_OPS):
        W = torch.eye(STATE_DIM)
        a, b = i % (STATE_DIM - 1), (i % (STATE_DIM - 1) + 1) % STATE_DIM
        theta = (i + 1) * np.pi / 7
        c, s = np.cos(theta), np.sin(theta)
        W[a, a], W[a, b], W[b, a], W[b, b] = c, -s, s, c
        scale_ax = (i * 2 + 3) % STATE_DIM
        W[scale_ax, scale_ax] = 1.15 if i % 2 == 0 else 0.85
        bias = torch.zeros(STATE_DIM)
        bias[a] = 0.15 * ((i % 3) - 1)
        ops.append((W, bias))
    torch.manual_seed(42)
    return ops

OPS = build_ops()
OP_KEYS = F.normalize(torch.randn(N_OPS, HINT_DIM), dim=-1)

def apply_path(state, path):
    h = state.clone()
    for idx in path:
        W, b = OPS[idx % N_OPS]
        h = torch.tanh(h @ W.T + b)
    return h[:, :OUTPUT_DIM]

def make_input(state, path):
    B = state.size(0)
    hint = 0.5 * OP_KEYS[path[0] % N_OPS] + 0.5 * OP_KEYS[path[-1] % N_OPS]
    hint = hint.unsqueeze(0).expand(B, -1) + 0.05 * torch.randn(B, HINT_DIM)
    return torch.cat([state, hint], dim=-1)

def generate_dataset(n, path_dict, seed_offset=0):
    g = torch.Generator().manual_seed(42 + seed_offset)
    states = torch.randn(n, STATE_DIM, generator=g) * 0.9
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,), generator=g)
    X = torch.zeros(n, INPUT_DIM)
    Y = torch.zeros(n, OUTPUT_DIM)
    paths = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        paths.append(p)
        X[i] = make_input(states[i:i+1], p).squeeze(0)
        Y[i] = apply_path(states[i:i+1], p).squeeze(0)
    Y = Y + 0.01 * torch.randn_like(Y)
    return X, Y, paths

class StructuredGivensModule(nn.Module):
    """Module trong họ algebraic: rotation 2D + scale + bias trên HIDDEN."""
    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.randn(1) * 0.3)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(dim))
        # fixed plane indices per module instance set outside or random fixed
        self.register_buffer("idx_a", torch.tensor(0))
        self.register_buffer("idx_b", torch.tensor(1))

    def set_plane(self, a, b):
        self.idx_a.fill_(a)
        self.idx_b.fill_(b)

    def forward(self, h):
        # h: B, dim
        a, b = int(self.idx_a), int(self.idx_b)
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        out = h.clone()
        ha, hb = h[:, a], h[:, b]
        out[:, a] = c * ha - s * hb
        out[:, b] = s * ha + c * hb
        scale = torch.exp(self.log_scale.clamp(-1, 1))
        out = out * scale + self.bias
        return torch.tanh(out)

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, KV_DIM)
        self.w_k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(KV_DIM, 1)
        self.out = nn.Linear(dim, dim)
    def forward(self, query, keys, values, mask=None):
        q = self.w_q(query).unsqueeze(1)
        k = self.w_k(keys)
        score = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return self.out(ctx), attn

class CausalKVLayer(nn.Module):
    def __init__(self, n_mod=N_MOD, topo="ring", structured=False):
        super().__init__()
        self.n_mod = n_mod
        self.structured = structured
        if structured:
            self.mods = nn.ModuleList([StructuredGivensModule(HIDDEN) for _ in range(n_mod)])
            for i, m in enumerate(self.mods):
                a = i % (HIDDEN - 1)
                b = (a + 1 + i // 2) % HIDDEN
                if a == b:
                    b = (a + 1) % HIDDEN
                m.set_plane(a, b)
        else:
            self.mods = nn.ModuleList([
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                for _ in range(n_mod)
            ])
        self.attn = AdditiveAttention(HIDDEN)
        self.controller = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, n_mod))
        self.kv_init = nn.Parameter(torch.randn(n_mod, HIDDEN) * 0.02)
        adj = torch.ones(n_mod, n_mod) if topo == "complete" else torch.zeros(n_mod, n_mod)
        if topo == "ring":
            for i in range(n_mod):
                adj[i, i] = adj[i, (i+1)%n_mod] = adj[i, (i-1)%n_mod] = 1
        elif topo == "sparse":
            rng = np.random.RandomState(42)
            for i in range(n_mod):
                adj[i, i] = 1
                cands = [j for j in range(n_mod) if j != i]
                adj[i, rng.choice(cands, min(2, len(cands)), replace=False)] = 1
        self.register_buffer("topo_mask", (adj.sum(0) > 0).float())

    def forward(self, h, temperature=1.0, return_traj=False):
        B = h.size(0)
        kv = self.kv_init.unsqueeze(0).expand(B, -1, -1).clone()
        written = torch.zeros(B, self.n_mod, device=h.device)
        traj = []
        for t in range(SEQ_LEN):
            attn_mask = written.clone() if t > 0 else torch.ones_like(written) * 0.15
            ctx, _ = self.attn(h, kv, kv, mask=attn_mask)
            logits = self.controller(ctx)
            logits = logits.masked_fill(self.topo_mask.unsqueeze(0) == 0, -1e9)
            soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
            if self.training:
                hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                gate = hard + (soft - soft.detach())
            else:
                gate = soft
            traj.append(gate)
            updates = [self.mods[mi](h) * gate[:, mi:mi+1] for mi in range(self.n_mod)]
            h = h + 0.5 * torch.stack(updates, 0).sum(0)
            write = gate.unsqueeze(-1) * h.unsqueeze(1)
            kv = kv * (1 - gate.unsqueeze(-1)) + write
            written = torch.clamp(written + gate.detach(), 0, 1)
        stacked = torch.stack(traj, 1)
        if return_traj:
            return h, stacked
        return h

class HierResidual(nn.Module):
    def __init__(self, n_blocks=12):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)

class PureC(nn.Module):
    def __init__(self, structured=False):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.layers = nn.ModuleList([
            CausalKVLayer(N_MOD, "sparse", structured=structured),
            CausalKVLayer(N_MOD, "ring", structured=structured),
            CausalKVLayer(N_MOD, "complete", structured=structured),
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x, temperature=1.0, **kwargs):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, temperature=temperature)
        return self.out(h)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, _ = generate_dataset(N_TRAIN, TRAIN_PATHS, seed)
    Xva, Yva, _ = generate_dataset(N_VAL, TRAIN_PATHS, seed+10)
    Xood, Yood, _ = generate_dataset(N_OOD, OOD_PATHS, seed+20)

    def train(model, use_temp=False):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        X, Y = Xtr.to(DEVICE), Ytr.to(DEVICE)
        Xv, Yv = Xva.to(DEVICE), Yva.to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                pred = model(X[idx], temperature=temp) if use_temp else model(X[idx])
                F.mse_loss(pred, Y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xv, temperature=TEMP_END) if use_temp else model(Xv)
                val = F.mse_loss(pred, Yv).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def eval_ood(model, use_temp=False):
        model.eval()
        with torch.no_grad():
            pred = model(Xood.to(DEVICE), temperature=TEMP_END) if use_temp else model(Xood.to(DEVICE))
            return F.mse_loss(pred, Yood.to(DEVICE)).item()

    out = {"seed": seed}
    hr = train(HierResidual())
    out["HR"] = eval_ood(hr)
    mlp = train(PureC(structured=False), use_temp=True)
    out["MLP"] = eval_ood(mlp, use_temp=True)
    st = train(PureC(structured=True), use_temp=True)
    out["STRUCT"] = eval_ood(st, use_temp=True)
    out["win_hr"] = out["STRUCT"] < out["HR"]
    out["win_mlp"] = out["STRUCT"] <= out["MLP"] * 0.98
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  HR={r['HR']:.4f} MLP={r['MLP']:.4f} STRUCT={r['STRUCT']:.4f} "
          f"winHR={r['win_hr']} winMLP={r['win_mlp']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

hr, mlp, st = ms("HR")[0], ms("MLP")[0], ms("STRUCT")[0]
w_hr = sum(r["win_hr"] for r in all_results)
w_mlp = sum(r["win_mlp"] for r in all_results)
print("\n===== E ALGEBRAIC SUMMARY =====")
print(f"HR={hr:.4f} MLP={mlp:.4f} STRUCT={st:.4f}")
print(f"STRUCT/HR={st/hr:.3f} wins_hr={w_hr}/3 wins_mlp={w_mlp}/3")

if st < hr * 0.95 and st <= mlp * 0.98 and w_hr >= 2:
    outcome = "PASS"
elif st > hr * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"
print(f"=== OUTCOME: {outcome} ===")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase E Algebraic structured modules",
        "seeds": SEEDS,
        "summary": {"HR": hr, "MLP": mlp, "STRUCT": st, "wins_hr": w_hr},
        "per_seed": all_results,
        "outcome": outcome,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase E Algebraic** structured Givens modules vs MLP PureC vs HR
- HR={hr:.4f} MLP={mlp:.4f} STRUCT={st:.4f} wins_hr={w_hr}/3
- Outcome: **{outcome}**
- Discrete claim FREEZE (A+D PASS). Algebraic = họ tiếp theo.
""")
print(f"Saved {run_dir}")
