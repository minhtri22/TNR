#!/usr/bin/env python3
"""
TNR Sequential + Curriculum path-length + Auxiliary sequence loss nhẹ

Giữ SeqHard + DiffTopo hierarchical.
Thêm:
  1. Curriculum: epoch đầu chỉ train path ngắn, dần mở path dài hơn
  2. Auxiliary loss: khuyến khích predicted module sequence khớp latent path (soft CE trên node presence)

Preregistration (giữ tinh thần trước):
  PASS nếu OOD_MSE_SeqHard < min(OOD_HierRes, OOD_SeqSoft)*0.95
           VÀ traj_seq_recovery_OOD >= 0.35
  FAIL nếu OOD_MSE_SeqHard > min_base*1.05
  else UNRESOLVED
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
EPOCHS = 35
LR = 1e-3
TEMP_START, TEMP_END = 1.5, 0.4
AUX_LAMBDA = 0.02          # auxiliary sequence loss
CURRICULUM_WARMUP = 10     # epochs chỉ path ngắn

# Train pools theo độ dài
SHORT_PATHS = {0: [0, 2, 5], 1: [1, 4, 7], 2: [1, 5, 3]}
MID_PATHS   = {0: [0, 3, 6, 2], 1: [2, 6, 4, 0], 2: [0, 2, 5, 1]}
# Full train = short + mid
TRAIN_PATHS = {**{k: v for k, v in SHORT_PATHS.items()},
               **{k+10: v for k, v in MID_PATHS.items()}}
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

X_train_short, Y_train_short, _, paths_short = generate_dataset(N_TRAIN//2, SHORT_PATHS, 0)
X_train_mid, Y_train_mid, _, paths_mid = generate_dataset(N_TRAIN//2, MID_PATHS, 1)
X_train = torch.cat([X_train_short, X_train_mid], 0)
Y_train = torch.cat([Y_train_short, Y_train_mid], 0)
paths_train = paths_short + paths_mid
# shuffle
perm = torch.randperm(N_TRAIN)
X_train, Y_train = X_train[perm], Y_train[perm]
paths_train = [paths_train[i] for i in perm.tolist()]

X_val, Y_val, _, paths_val = generate_dataset(N_VAL, TRAIN_PATHS, 2)
X_ood, Y_ood, _, paths_ood = generate_dataset(N_OOD, OOD_PATHS, 3)
print(f"Train {X_train.shape}  OOD {X_ood.shape}  Y var={Y_train.var():.4f}")

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
        self.topo_masks = []
        for mi, mode in enumerate(["sparse", "ring", "complete"]):
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
                traj_layer.append(gate)  # keep soft for aux loss
                updates = [layer_mods[mi](h) * gate[:, mi:mi+1] for mi in range(MOD_PER_LAYER)]
                h = h + 0.5 * torch.tanh(torch.stack(updates, 0).sum(0))
            all_traj.append(torch.stack(traj_layer, 1))
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, 1)  # B, L, T, M
        return out

    def aux_sequence_loss(self, traj, true_paths):
        """Soft CE: khuyến khích mass trên các module xuất hiện trong true path."""
        # traj: B, L, T, M
        B = traj.size(0)
        # average gate over L,T → B, M  (local module preference)
        avg_gate = traj.mean(dim=(1, 2))  # B, M
        loss = 0.0
        for b in range(B):
            path = true_paths[b]
            # target: modules that appear (mapped mod 4)
            target = torch.zeros(MOD_PER_LAYER, device=traj.device)
            for m in path:
                target[m % MOD_PER_LAYER] = 1.0
            target = target / (target.sum() + 1e-8)
            p = avg_gate[b] / (avg_gate[b].sum() + 1e-8)
            loss = loss + -(target * (p + 1e-8).log()).sum()
        return loss / B


# ===================== METRICS =====================
def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def traj_seq_recovery(traj, true_paths, n_sample=400):
    B = min(n_sample, traj.size(0))
    traj = traj[:B]
    scores = []
    for b in range(B):
        chosen = traj[b].argmax(-1).view(-1).tolist()
        pred_seq = [ (idx//SEQ_LEN)*MOD_PER_LAYER + c for idx, c in enumerate(chosen) ]
        true_p = true_paths[b]
        scores.append(lcs_length(pred_seq, true_p) / max(len(true_p), 1))
    return float(np.mean(scores))


# ===================== TRAIN =====================
def train_model(model, name, use_traj=False, use_aux=False, use_curriculum=False):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / max(EPOCHS-1, 1))
        # curriculum: first WARMUP epochs only short paths
        if use_curriculum and epoch < CURRICULUM_WARMUP:
            Xtr = X_train_short.to(DEVICE)
            Ytr = Y_train_short.to(DEVICE)
            ptr = paths_short
            n_tr = X_train_short.size(0)
        else:
            Xtr = X_train.to(DEVICE)
            Ytr = Y_train.to(DEVICE)
            ptr = paths_train
            n_tr = N_TRAIN

        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        ep_loss, n_b = 0.0, 0
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]
            batch_paths = [ptr[j] for j in idx.cpu().tolist()]
            opt.zero_grad()
            if use_traj:
                pred, traj = model(xb, return_traj=True, temperature=temp)
                loss = F.mse_loss(pred, yb)
                if use_aux:
                    loss = loss + AUX_LAMBDA * model.aux_sequence_loss(traj, batch_paths)
            else:
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            Xv, Yv = X_val.to(DEVICE), Y_val.to(DEVICE)
            if use_traj:
                pred_v = model(Xv, temperature=TEMP_END)
            else:
                pred_v = model(Xv)
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
results["HierResidual_ID"] = evaluate(m, "HierRes", X=X_val, Y=Y_val, paths=paths_val)
results["HierResidual_OOD"] = evaluate(m, "HierRes", X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["HierResidual"] = h
print(f"ID={results['HierResidual_ID']['mse']:.4f} OOD={results['HierResidual_OOD']['mse']:.4f}")

print("\n=== SeqHard + Curriculum + Aux ===")
m = SequentialTrajectory(use_hard=True)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqHard_CurrAux", use_traj=True, use_aux=True, use_curriculum=True)
results["SeqHard_CurrAux_ID"] = evaluate(m, "SeqHardCA", use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
results["SeqHard_CurrAux_OOD"] = evaluate(m, "SeqHardCA", use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["SeqHard_CurrAux"] = h
print(f"ID={results['SeqHard_CurrAux_ID']['mse']:.4f} OOD={results['SeqHard_CurrAux_OOD']['mse']:.4f} trajOOD={results['SeqHard_CurrAux_OOD']['traj_seq_recovery']:.3f}")

print("\n=== SeqHard (no curr/aux, control) ===")
m = SequentialTrajectory(use_hard=True)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqHard_plain", use_traj=True, use_aux=False, use_curriculum=False)
results["SeqHard_plain_ID"] = evaluate(m, "SeqHardP", use_traj=True, X=X_val, Y=Y_val, paths=paths_val)
results["SeqHard_plain_OOD"] = evaluate(m, "SeqHardP", use_traj=True, X=X_ood, Y=Y_ood, paths=paths_ood, split="OOD")
histories["SeqHard_plain"] = h
print(f"ID={results['SeqHard_plain_ID']['mse']:.4f} OOD={results['SeqHard_plain_OOD']['mse']:.4f} trajOOD={results['SeqHard_plain_OOD']['traj_seq_recovery']:.3f}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "SeqHard + Curriculum path-length + Aux sequence loss",
        "preregistration": "PASS if OOD_MSE_SeqHardCA < min_base*0.95 AND traj_rec>=0.35",
        "results": results
    }, f, indent=2)
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nSaved {run_dir}")
print(json.dumps({k: {kk: results[k][kk] for kk in ["mse","r2","traj_seq_recovery"]} for k in results}, indent=2))

ood_res = results["HierResidual_OOD"]["mse"]
ood_plain = results["SeqHard_plain_OOD"]["mse"]
ood_ca = results["SeqHard_CurrAux_OOD"]["mse"]
best_base = min(ood_res, ood_plain)
rec = results["SeqHard_CurrAux_OOD"]["traj_seq_recovery"] or 0

if ood_ca < best_base * 0.95 and rec >= 0.35:
    outcome = "PASS"
elif ood_ca > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"OOD  HierRes={ood_res:.4f}  SeqHard_plain={ood_plain:.4f}  SeqHard_CurrAux={ood_ca:.4f}")
print(f"Traj OOD CurrAux={rec:.3f}")

entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: SeqHard + Curriculum path-length + Aux sequence loss
- Results OOD MSE: HierRes={ood_res:.4f}, SeqHard_plain={ood_plain:.4f}, SeqHard_CurrAux={ood_ca:.4f}
- Traj seq recovery OOD: plain={results['SeqHard_plain_OOD']['traj_seq_recovery']}, CurrAux={rec}
- Outcome: **{outcome}**
- Notes: Curriculum warmup {CURRICULUM_WARMUP} ep short paths; AUX_LAMBDA={AUX_LAMBDA}.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
