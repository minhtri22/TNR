#!/usr/bin/env python3
"""
TNR Sequential Trajectory trên task Variable Long-Dependency + OOD Composition

Preregistration:
  Metric chính: OOD MSE (thấp hơn tốt) + Traj sequence recovery (LCS / exact-ish)
  PASS nếu:
    OOD_MSE_SeqHard < min(OOD_MSE_HierRes, OOD_MSE_SeqSoft) * 0.95
    VÀ traj_seq_recovery >= 0.35
  FAIL nếu OOD_MSE_SeqHard > min_baseline * 1.05
  else UNRESOLVED

Train families: path ngắn–trung (length 3–4)
OOD families: path dài hơn hoặc composition mới (length 5–6, mix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from itertools import combinations

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_LAYERS = 3
MOD_PER_LAYER = 4
HIDDEN = 48
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 5
BATCH_SIZE = 128
N_TRAIN = 5000
N_VAL = 800
N_OOD = 1000
EPOCHS = 40
LR = 1e-3
TEMP_START, TEMP_END = 1.5, 0.4

# Train paths (seen)
TRAIN_PATHS = {
    0: [0, 2, 5],
    1: [1, 4, 7],
    2: [0, 3, 6, 2],
    3: [1, 5, 3],
    4: [2, 6, 4, 0],
}
# OOD paths (unseen compositions / longer)
OOD_PATHS = {
    0: [0, 2, 5, 7, 3],          # longer
    1: [1, 4, 7, 0, 6],          # longer + mix
    2: [0, 3, 6, 2, 5, 1],       # longer composition
    3: [2, 6, 4, 0, 7],          # reorder + extend
    4: [1, 5, 3, 6, 2],          # new order
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ===================== DATA =====================
def generate_true_transforms(n_mod=12):
    torch.manual_seed(123)
    transforms = []
    for i in range(n_mod):
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.12
        d1, d2 = i % INPUT_DIM, (i * 3 + 1) % INPUT_DIM
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
        W, b = TRUE_TRANSFORMS[m_idx % len(TRUE_TRANSFORMS)]
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

X_train, Y_train, cases_train, paths_train = generate_dataset(N_TRAIN, TRAIN_PATHS, 0)
X_val, Y_val, cases_val, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, 1)
X_ood, Y_ood, cases_ood, paths_ood = generate_dataset(N_OOD, OOD_PATHS, 2)
print(f"Train {X_train.shape}  OOD {X_ood.shape}  Y_train var={Y_train.var():.4f}")

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


class SequentialTrajectory(nn.Module):
    def __init__(self, use_hard=False):
        super().__init__()
        self.use_hard = use_hard
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
        # DiffTopo masks
        self.topo_masks = []
        for mi, mode in enumerate(["sparse", "ring", "complete"]):
            adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
            if mode == "complete":
                adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
            elif mode == "ring":
                for i in range(MOD_PER_LAYER):
                    adj[i, (i+1)%MOD_PER_LAYER] = 1
                    adj[i, (i-1)%MOD_PER_LAYER] = 1
                    adj[i, i] = 1
            else:
                rng = np.random.RandomState(42)
                for i in range(MOD_PER_LAYER):
                    adj[i, i] = 1
                    cands = [j for j in range(MOD_PER_LAYER) if j != i]
                    chosen = rng.choice(cands, size=2, replace=False)
                    adj[i, chosen] = 1
            self.register_buffer(f"mask_{mi}", adj)
            self.topo_masks.append(adj)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.seq_len = SEQ_LEN

    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)
        all_traj = []
        for li in range(N_LAYERS):
            layer_mods = self.mod_lists[li]
            ctrl = self.controllers[li]
            mask = (self.topo_masks[li].sum(0) > 0).float().to(x.device)
            traj_layer = []
            for t in range(self.seq_len):
                logits = ctrl(h)
                logits = logits.masked_fill(mask.unsqueeze(0) == 0, -1e9)
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.use_hard and self.training:
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                traj_layer.append(gate.detach())
                updates = [layer_mods[mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
                update = torch.stack(updates, 0).sum(0)
                h = h + 0.5 * torch.tanh(update)
            all_traj.append(torch.stack(traj_layer, 1))
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, 1)  # B, L, T, M
        return out


# ===================== METRICS =====================

def lcs_length(a, b):
    """Longest Common Subsequence length."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_seq_recovery(traj, true_paths, n_sample=400):
    """
    traj: B, L, T, M  → argmax → sequence of module ids (flattened, mapped 0..11)
    true_paths: list of lists
    Recovery = LCS(pred_seq, true_path) / max(len(true), 1)
    """
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        # predicted sequence: across layers and steps
        chosen = traj[b].argmax(dim=-1).view(-1).tolist()  # L*T
        # map to global-ish ids (layer*4 + local)
        pred_seq = []
        for idx, c in enumerate(chosen):
            layer = idx // SEQ_LEN
            pred_seq.append(layer * MOD_PER_LAYER + c)
        true_p = true_paths[b]
        # normalize true to similar range if needed; use raw for LCS
        lcs = lcs_length(pred_seq, true_p)
        scores.append(lcs / max(len(true_p), 1))
    return float(np.mean(scores))


# ===================== TRAIN / EVAL =====================

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
            if use_traj:
                pred = model(xb, temperature=temp)
            else:
                pred = model(xb)
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


def evaluate(model, name, use_traj=False, X=None, Y=None, paths=None, split="test"):
    model.eval()
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)
    with torch.no_grad():
        if use_traj:
            pred, traj = model(X, return_traj=True, temperature=TEMP_END)
            rec = traj_seq_recovery(traj.cpu(), paths)
        else:
            pred = model(X)
            rec = None
        mse = F.mse_loss(pred, Y).item()
        r2 = 1 - ((pred - Y)**2).sum().item() / (((Y - Y.mean(0))**2).sum().item() + 1e-8)
    return {"mse": mse, "r2": r2, "traj_seq_recovery": rec, "params": count_params(model), "split": split}


# ===================== RUN =====================
results = {}
histories = {}

print("\n=== HierResidual ===")
m = HierResidual()
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "HierResidual")
results["HierResidual_ID"] = evaluate(m, "HierResidual", X=X_val, Y=Y_val, paths=paths_val, split="ID")
results["HierResidual_OOD"] = evaluate(m, "HierResidual", X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["HierResidual"] = h
print(f"ID MSE={results['HierResidual_ID']['mse']:.4f}  OOD MSE={results['HierResidual_OOD']['mse']:.4f}")

print("\n=== SeqSoft ===")
m = SequentialTrajectory(use_hard=False)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqSoft", use_traj=True)
results["SeqSoft_ID"] = evaluate(m, "SeqSoft", use_traj=True, X=X_val, Y=Y_val, paths=paths_val, split="ID")
results["SeqSoft_OOD"] = evaluate(m, "SeqSoft", use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["SeqSoft"] = h
print(f"ID MSE={results['SeqSoft_ID']['mse']:.4f} OOD={results['SeqSoft_OOD']['mse']:.4f} trajOOD={results['SeqSoft_OOD']['traj_seq_recovery']:.3f}")

print("\n=== SeqHard (Gumbel) ===")
m = SequentialTrajectory(use_hard=True)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqHard", use_traj=True)
results["SeqHard_ID"] = evaluate(m, "SeqHard", use_traj=True, X=X_val, Y=Y_val, paths=paths_val, split="ID")
results["SeqHard_OOD"] = evaluate(m, "SeqHard", use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["SeqHard"] = h
print(f"ID MSE={results['SeqHard_ID']['mse']:.4f} OOD={results['SeqHard_OOD']['mse']:.4f} trajOOD={results['SeqHard_OOD']['traj_seq_recovery']:.3f}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Sequential Trajectory on Variable Long-Dependency + OOD Composition",
        "preregistration": {
            "PASS": "OOD_MSE_SeqHard < min(OOD_HierRes, OOD_SeqSoft)*0.95 AND traj_seq_recovery>=0.35",
            "FAIL": "OOD_MSE_SeqHard > min_baseline*1.05",
            "else": "UNRESOLVED"
        },
        "train_paths": {str(k): v for k, v in TRAIN_PATHS.items()},
        "ood_paths": {str(k): v for k, v in OOD_PATHS.items()},
        "results": results
    }, f, indent=2)
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nSaved {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","r2","traj_seq_recovery"]} for k in results}, indent=2))

# Outcome
ood_res = results["HierResidual_OOD"]["mse"]
ood_soft = results["SeqSoft_OOD"]["mse"]
ood_hard = results["SeqHard_OOD"]["mse"]
best_base = min(ood_res, ood_soft)
rec = results["SeqHard_OOD"]["traj_seq_recovery"] or 0

if ood_hard < best_base * 0.95 and rec >= 0.35:
    outcome = "PASS"
elif ood_hard > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"OOD MSE  HierRes={ood_res:.4f}  SeqSoft={ood_soft:.4f}  SeqHard={ood_hard:.4f}")
print(f"Traj seq recovery OOD SeqHard={rec:.3f}")

entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Sequential Trajectory on Variable Long-Dependency + OOD Composition
- Train paths length 3–4 | OOD paths length 5–6 + new compositions
- Results OOD MSE: HierRes={ood_res:.4f}, SeqSoft={ood_soft:.4f}, SeqHard={ood_hard:.4f}
- Traj seq recovery OOD: Soft={results['SeqSoft_OOD']['traj_seq_recovery']}, Hard={rec}
- ID MSE: HierRes={results['HierResidual_ID']['mse']:.4f}, Soft={results['SeqSoft_ID']['mse']:.4f}, Hard={results['SeqHard_ID']['mse']:.4f}
- Outcome: **{outcome}**
- Notes: Metric chính OOD MSE + LCS-based traj sequence recovery. SeqHard ưu tiên.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
