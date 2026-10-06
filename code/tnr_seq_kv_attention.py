#!/usr/bin/env python3
"""
TNR Hướng 2: Sequential Trajectory + KV Cache trượt theo node + Attention styles

Controller mỗi bước attend vào bảng KV của các module/node đã thăm (sliding).
Thử 3 attention styles:
  - scaled_dot (standard)
  - additive (Bahdanau-style)
  - gated (gate giữa content + position)

So sánh với:
  - HierResidual
  - SeqHard (no memory, control từ trước)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ===================== PREREGISTRATION =====================
"""
Task: Variable Long-Dependency + OOD (giữ từ run trước)
PASS nếu:
  OOD_MSE_best_KV < min(OOD_HierRes, OOD_SeqHard_nomem) * 0.95
  VÀ traj_seq_recovery_OOD >= 0.35
FAIL nếu OOD_MSE_best_KV > min_base * 1.05
else UNRESOLVED
"""

N_LAYERS = 3
MOD_PER_LAYER = 4
HIDDEN = 48
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 5
KV_DIM = 32
BATCH_SIZE = 128
N_TRAIN = 3000
N_VAL = 800
N_OOD = 1000
EPOCHS = 25
LR = 1e-3
TEMP_START, TEMP_END = 1.4, 0.4

SHORT_PATHS = {0: [0, 2, 5], 1: [1, 4, 7], 2: [1, 5, 3]}
MID_PATHS   = {0: [0, 3, 6, 2], 1: [2, 6, 4, 0], 2: [0, 2, 5, 1]}
TRAIN_PATHS = {**SHORT_PATHS, **{k+10: v for k, v in MID_PATHS.items()}}
OOD_PATHS = {
    0: [0, 2, 5, 7, 3],
    1: [1, 4, 7, 0, 6],
    2: [0, 3, 6, 2, 5, 1],
    3: [2, 6, 4, 0, 7],
    4: [1, 5, 3, 6, 2],
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def generate_true_transforms(n=12):
    torch.manual_seed(123)
    transforms = []
    for i in range(n):
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.12
        d1, d2 = i % INPUT_DIM, (i*3+1) % INPUT_DIM
        W[d1, d1] += 1.0
        W[d2, d2] += 0.85
        b = torch.zeros(INPUT_DIM)
        b[d1] = 0.25 * ((i % 5) - 2)
        transforms.append((W, b))
    torch.manual_seed(SEED)
    return transforms

TRUE_TRANSFORMS = generate_true_transforms()

def apply_path(x, path):
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx % 12]
        h = h + 0.55 * torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n, path_dict, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    X = torch.randn(n, INPUT_DIM) * 0.7
    keys = list(path_dict.keys())
    cases = torch.randint(0, len(keys), (n,))
    Y = torch.zeros(n, OUTPUT_DIM)
    paths_list = []
    for i in range(n):
        p = path_dict[keys[cases[i].item()]]
        Y[i] = apply_path(X[i:i+1], p).squeeze(0)
        paths_list.append(p)
    Y = Y + 0.012 * torch.randn_like(Y)
    torch.manual_seed(SEED)
    return X, Y, cases, paths_list

X_train, Y_train, _, paths_train = generate_dataset(N_TRAIN, TRAIN_PATHS, 0)
X_val, Y_val, _, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, 1)
X_ood, Y_ood, _, paths_ood = generate_dataset(N_OOD, OOD_PATHS, 2)
print(f"Train {X_train.shape}  OOD {X_ood.shape}  Y var={Y_train.var():.4f}")

# ===================== ATTENTION STYLES =====================

class ScaledDotAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, KV_DIM)
        self.k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(dim, KV_DIM)
        self.out = nn.Linear(KV_DIM, dim)
    def forward(self, query, keys, values, mask=None):
        # query: B, D | keys/values: B, L, D
        q = self.q(query).unsqueeze(1)          # B, 1, Kv
        k = self.k(keys)                        # B, L, Kv
        v = self.v(values)
        score = torch.bmm(q, k.transpose(1, 2)) / (KV_DIM ** 0.5)  # B, 1, L
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn, v).squeeze(1)     # B, Kv
        return self.out(ctx), attn.squeeze(1)


class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = nn.Linear(dim, KV_DIM)
        self.w_k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(KV_DIM, 1)
        self.out = nn.Linear(dim, dim)
    def forward(self, query, keys, values, mask=None):
        # Bahdanau-style
        q = self.w_q(query).unsqueeze(1)        # B, 1, Kv
        k = self.w_k(keys)                      # B, L, Kv
        score = self.v(torch.tanh(q + k)).squeeze(-1)  # B, L
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return self.out(ctx), attn


class GatedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, KV_DIM)
        self.k = nn.Linear(dim, KV_DIM)
        self.v = nn.Linear(dim, KV_DIM)
        self.gate = nn.Linear(dim + KV_DIM, dim)
        self.out = nn.Linear(KV_DIM, dim)
    def forward(self, query, keys, values, mask=None):
        q = self.q(query).unsqueeze(1)
        k = self.k(keys)
        v = self.v(values)
        score = torch.bmm(q, k.transpose(1, 2)) / (KV_DIM ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn, v).squeeze(1)
        g = torch.sigmoid(self.gate(torch.cat([query, ctx], dim=-1)))
        mixed = g * self.out(ctx) + (1 - g) * query
        return mixed, attn.squeeze(1)


# ===================== MODELS =====================

class HierResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_LAYERS * MOD_PER_LAYER)
        ])
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


class SeqHardNoMem(nn.Module):
    """Control: SeqHard không memory (giống trước)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.mod_lists = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                for _ in range(MOD_PER_LAYER)
            ]) for _ in range(N_LAYERS)
        ])
        self.controllers = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, 48), nn.Tanh(), nn.Linear(48, MOD_PER_LAYER))
            for _ in range(N_LAYERS)
        ])
        self.topo_masks = self._make_masks()
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
    def _make_masks(self):
        masks = []
        for mode in ["sparse", "ring", "complete"]:
            adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
            if mode == "complete":
                adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
            elif mode == "ring":
                for i in range(MOD_PER_LAYER):
                    adj[i, (i+1)%MOD_PER_LAYER] = adj[i, (i-1)%MOD_PER_LAYER] = adj[i, i] = 1
            else:
                rng = np.random.RandomState(42)
                for i in range(MOD_PER_LAYER):
                    adj[i, i] = 1
                    cands = [j for j in range(MOD_PER_LAYER) if j != i]
                    adj[i, rng.choice(cands, 2, replace=False)] = 1
            masks.append(adj)
        return nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in masks])
    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)
        all_traj = []
        for li in range(N_LAYERS):
            mask = (self.topo_masks[li].sum(0) > 0).float().to(x.device)
            traj_layer = []
            for t in range(SEQ_LEN):
                logits = self.controllers[li](h)
                logits = logits.masked_fill(mask.unsqueeze(0) == 0, -1e9)
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.training:
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                traj_layer.append(gate.detach())
                updates = [self.mod_lists[li][mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
                h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            all_traj.append(torch.stack(traj_layer, 1))
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, 1)
        return out


class SeqKVAttention(nn.Module):
    """
    Sequential + sliding KV table per node.
    Mỗi khi chọn module i, ghi (key=state, value=state) vào slot i (overwrite = trượt mới nhất).
    Controller attend vào toàn bộ KV table để quyết định bước tiếp.
    """
    def __init__(self, attn_style="scaled_dot"):
        super().__init__()
        self.attn_style = attn_style
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.mod_lists = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                for _ in range(MOD_PER_LAYER)
            ]) for _ in range(N_LAYERS)
        ])
        # attention modules per layer
        if attn_style == "scaled_dot":
            self.attns = nn.ModuleList([ScaledDotAttention(HIDDEN) for _ in range(N_LAYERS)])
        elif attn_style == "additive":
            self.attns = nn.ModuleList([AdditiveAttention(HIDDEN) for _ in range(N_LAYERS)])
        else:  # gated
            self.attns = nn.ModuleList([GatedAttention(HIDDEN) for _ in range(N_LAYERS)])
        # controller: từ context (attended) → logits module
        self.controllers = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN, 48), nn.Tanh(), nn.Linear(48, MOD_PER_LAYER))
            for _ in range(N_LAYERS)
        ])
        self.topo_masks = self._make_masks()
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        # learnable init KV for each module slot
        self.kv_init = nn.Parameter(torch.randn(N_LAYERS, MOD_PER_LAYER, HIDDEN) * 0.02)

    def _make_masks(self):
        masks = []
        for mode in ["sparse", "ring", "complete"]:
            adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
            if mode == "complete":
                adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
            elif mode == "ring":
                for i in range(MOD_PER_LAYER):
                    adj[i, (i+1)%MOD_PER_LAYER] = adj[i, (i-1)%MOD_PER_LAYER] = adj[i, i] = 1
            else:
                rng = np.random.RandomState(42)
                for i in range(MOD_PER_LAYER):
                    adj[i, i] = 1
                    cands = [j for j in range(MOD_PER_LAYER) if j != i]
                    adj[i, rng.choice(cands, 2, replace=False)] = 1
            masks.append(adj)
        return nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in masks])

    def forward(self, x, return_traj=False, temperature=1.0):
        B = x.size(0)
        h = self.input_proj(x)
        all_traj = []
        for li in range(N_LAYERS):
            # init KV table: B, M, H
            kv = self.kv_init[li].unsqueeze(0).expand(B, -1, -1).clone()
            mask_topo = (self.topo_masks[li].sum(0) > 0).float().to(x.device)
            traj_layer = []
            for t in range(SEQ_LEN):
                # attend to current KV table
                ctx, attn_weights = self.attns[li](h, kv, kv)  # query=h, key/value=kv
                # controller from context
                logits = self.controllers[li](ctx)
                logits = logits.masked_fill(mask_topo.unsqueeze(0) == 0, -1e9)
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.training:
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                traj_layer.append(gate.detach())
                # execute selected modules
                updates = [self.mod_lists[li][mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
                update = torch.stack(updates, 0).sum(0)
                h = h + 0.5 * torch.tanh(update)
                # sliding write: overwrite KV slots of selected modules with new state
                # soft write for gradient
                write = gate.unsqueeze(-1) * h.unsqueeze(1)  # B, M, H
                kv = kv * (1 - gate.unsqueeze(-1)) + write
            all_traj.append(torch.stack(traj_layer, 1))
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, 1)
        return out


# ===================== METRICS & TRAIN =====================

def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_seq_recovery(traj, true_paths, n_sample=350):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [(idx//SEQ_LEN)*MOD_PER_LAYER + c for idx, c in enumerate(chosen)]
        scores.append(lcs_length(pred_seq, true_paths[b]) / max(len(true_paths[b]), 1))
    return float(np.mean(scores))


def train_model(model, name, use_traj=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, Ytr = X_train.to(DEVICE), Y_train.to(DEVICE)
    Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
        model.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        ep_loss, n_b = 0.0, 0
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            pred = model(xb, temperature=temp) if use_traj else model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv, temperature=TEMP_END) if use_traj else model(Xv)
            val_loss = F.mse_loss(pred_v, Yv).item()
        history["train_loss"].append(ep_loss / n_b)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [{name}] Ep {epoch+1}/{EPOCHS} train={history['train_loss'][-1]:.4f} val={val_loss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val


def evaluate(model, name, use_traj=False, X=None, Y=None, paths=None, split="ID"):
    model.eval()
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    with torch.no_grad():
        if use_traj:
            pred, traj = model(X, return_traj=True, temperature=TEMP_END)
            rec = traj_seq_recovery(traj.cpu(), paths)
        else:
            pred = model(X)
            rec = None
        mse = F.mse_loss(pred, Y).item()
        r2 = 1 - ((pred-Y)**2).sum().item() / (((Y-Y.mean(0))**2).sum().item() + 1e-8)
    return {"mse": mse, "r2": r2, "traj_seq_recovery": rec, "params": count_params(model), "split": split}


# ===================== RUN =====================
results, histories = {}, {}

print("\n=== HierResidual ===")
m = HierResidual()
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "HierResidual")
results["HierResidual_ID"] = evaluate(m, "HR", X=X_val, Y=Y_val, paths=paths_val)
results["HierResidual_OOD"] = evaluate(m, "HR", X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["HierResidual"] = h
print(f"ID={results['HierResidual_ID']['mse']:.4f} OOD={results['HierResidual_OOD']['mse']:.4f}")

print("\n=== SeqHard NoMem (control) ===")
m = SeqHardNoMem()
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqNoMem", use_traj=True)
results["SeqNoMem_ID"] = evaluate(m, "SNM", use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
results["SeqNoMem_OOD"] = evaluate(m, "SNM", use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["SeqNoMem"] = h
print(f"ID={results['SeqNoMem_ID']['mse']:.4f} OOD={results['SeqNoMem_OOD']['mse']:.4f} traj={results['SeqNoMem_OOD']['traj_seq_recovery']:.3f}")

for style in ["scaled_dot", "additive", "gated"]:
    print(f"\n=== SeqKV {style} ===")
    m = SeqKVAttention(attn_style=style)
    print(f"Params: {count_params(m)}")
    m, h, _ = train_model(m, f"KV_{style}", use_traj=True)
    results[f"KV_{style}_ID"] = evaluate(m, style, use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
    results[f"KV_{style}_OOD"] = evaluate(m, style, use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
    histories[f"KV_{style}"] = h
    r = results[f"KV_{style}_OOD"]
    print(f"ID={results[f'KV_{style}_ID']['mse']:.4f} OOD={r['mse']:.4f} traj={r['traj_seq_recovery']:.3f}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Sequential + sliding KV per node + attention styles",
        "attn_styles": ["scaled_dot", "additive", "gated"],
        "preregistration": "PASS if best_KV OOD_MSE < min_base*0.95 AND traj_rec>=0.35",
        "results": results
    }, f, indent=2)
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nSaved {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","traj_seq_recovery"]} for k in results if "OOD" in k}, indent=2))

# Outcome
ood_res = results["HierResidual_OOD"]["mse"]
ood_nomem = results["SeqNoMem_OOD"]["mse"]
kv_oods = {s: results[f"KV_{s}_OOD"]["mse"] for s in ["scaled_dot", "additive", "gated"]}
best_kv_name = min(kv_oods, key=kv_oods.get)
best_kv = kv_oods[best_kv_name]
best_base = min(ood_res, ood_nomem)
rec = results[f"KV_{best_kv_name}_OOD"]["traj_seq_recovery"] or 0

if best_kv < best_base * 0.95 and rec >= 0.35:
    outcome = "PASS"
elif best_kv > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"OOD  HierRes={ood_res:.4f}  NoMem={ood_nomem:.4f}  best_KV({best_kv_name})={best_kv:.4f}")
print(f"Traj OOD best_KV={rec:.3f}")

entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Sequential + sliding KV table per node + attention styles (scaled_dot / additive / gated)
- Results OOD MSE: HierRes={ood_res:.4f}, SeqNoMem={ood_nomem:.4f}, scaled_dot={kv_oods['scaled_dot']:.4f}, additive={kv_oods['additive']:.4f}, gated={kv_oods['gated']:.4f}
- Traj recovery OOD: NoMem={results['SeqNoMem_OOD']['traj_seq_recovery']}, best_KV({best_kv_name})={rec}
- Outcome: **{outcome}**
- Notes: KV trượt = overwrite slot module được chọn bằng state mới. Controller attend KV rồi chọn module tiếp.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
