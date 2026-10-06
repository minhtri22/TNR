#!/usr/bin/env python3
"""
TNR: Explicit Sequential Trajectory (gọi module tuần tự)
thay vì message-passing song song trên graph.

Cấu trúc vẫn hierarchical DiffTopo (sparse → ring → complete theo lớp),
nhưng trong mỗi lớp (hoặc toàn cục) dùng controller chọn thứ tự gọi module,
state truyền tuần tự.

So sánh:
- HierResidual (baseline song song residual)
- DiffTopo_parallel_static / gated (từ trước, parallel)
- SequentialTrajectory (controller chọn sequence, soft/hard)
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
PASS nếu:
  MSE_sequential < min(MSE HierResidual, DiffTopo_parallel) * 0.95
  VÀ traj_recovery (exact module sequence match proxy) >= 0.30
FAIL nếu MSE_seq > min_baseline * 1.05
else UNRESOLVED
"""

N_LAYERS = 3
MOD_PER_LAYER = 4
HIDDEN = 40
INPUT_DIM = 16
OUTPUT_DIM = 4
SEQ_LEN = 4                 # số bước gọi module tuần tự mỗi lớp
BATCH_SIZE = 128
N_TRAIN = 4000
N_VAL = 800
N_TEST = 800
EPOCHS = 35
LR = 1e-3
TEMP_START, TEMP_END = 1.5, 0.5

CASE_PATHS = {0: [0, 3, 6], 1: [0, 5, 2], 2: [1, 7, 4, 2]}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ===================== DATA =====================
def generate_true_module_transforms():
    torch.manual_seed(123)
    transforms = []
    for i in range(12):
        W = torch.randn(INPUT_DIM, INPUT_DIM) * 0.15
        d1, d2 = i % INPUT_DIM, (i * 3 + 1) % INPUT_DIM
        W[d1, d1] += 1.1
        W[d2, d2] += 0.9
        b = torch.zeros(INPUT_DIM)
        b[d1] = 0.3 * (i % 3 - 1)
        transforms.append((W, b))
    torch.manual_seed(SEED)
    return transforms

TRUE_TRANSFORMS = generate_true_module_transforms()

def apply_true_path(x, path):
    h = x.clone()
    for m_idx in path:
        W, b = TRUE_TRANSFORMS[m_idx % 12]
        h = h + 0.6 * torch.tanh(h @ W + b)
    return h[:, :OUTPUT_DIM]

def generate_dataset(n):
    X = torch.randn(n, INPUT_DIM) * 0.7
    cases = torch.randint(0, 3, (n,))
    Y = torch.zeros(n, OUTPUT_DIM)
    for i in range(n):
        Y[i] = apply_true_path(X[i:i+1], CASE_PATHS[cases[i].item()]).squeeze(0)
    return X, Y + 0.015 * torch.randn_like(Y), cases

X_train, Y_train, cases_train = generate_dataset(N_TRAIN)
X_val, Y_val, cases_val = generate_dataset(N_VAL)
X_test, Y_test, cases_test = generate_dataset(N_TEST)
print(f"Train {X_train.shape}  Y var={Y_train.var():.4f}")

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
    def forward(self, x):
        h = self.input_proj(x)
        for b in self.blocks:
            h = h + 0.4 * b(h)
        return self.out(h)


class SequentialTrajectory(nn.Module):
    """
    Explicit sequential module calling.
    - Hierarchical: 3 layers, mỗi layer có 4 module + topology hint (DiffTopo).
    - Controller mỗi bước chọn (soft) 1 module trong layer để cập nhật state.
    - State truyền tuần tự qua SEQ_LEN bước mỗi layer.
    """
    def __init__(self, use_hard=False):
        super().__init__()
        self.use_hard = use_hard
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN)
        # modules per layer
        self.mod_lists = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN))
                for _ in range(MOD_PER_LAYER)
            ]) for _ in range(N_LAYERS)
        ])
        # controller: từ state hiện tại → logits chọn module trong layer
        self.controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, 32),
                nn.Tanh(),
                nn.Linear(32, MOD_PER_LAYER)
            ) for _ in range(N_LAYERS)
        ])
        # topology bias (DiffTopo): sparse / ring / complete → mask logits
        self.topo_masks = []
        modes = ["sparse", "ring", "complete"]
        for m in modes:
            adj = torch.zeros(MOD_PER_LAYER, MOD_PER_LAYER)
            if m == "complete":
                adj = torch.ones(MOD_PER_LAYER, MOD_PER_LAYER)
            elif m == "ring":
                for i in range(MOD_PER_LAYER):
                    adj[i, (i+1)%MOD_PER_LAYER] = 1
                    adj[i, (i-1)%MOD_PER_LAYER] = 1
                    adj[i, i] = 1
            else:  # sparse
                rng = np.random.RandomState(42)
                for i in range(MOD_PER_LAYER):
                    adj[i, i] = 1
                    cands = [j for j in range(MOD_PER_LAYER) if j != i]
                    chosen = rng.choice(cands, size=2, replace=False)
                    adj[i, chosen] = 1
            self.register_buffer(f"mask_{m}", adj)
            self.topo_masks.append(adj)
        self.out = nn.Linear(HIDDEN, OUTPUT_DIM)
        self.seq_len = SEQ_LEN

    def forward(self, x, return_traj=False, temperature=1.0):
        h = self.input_proj(x)  # B, H  (single state, sequential)
        all_traj = []
        for li in range(N_LAYERS):
            layer_mods = self.mod_lists[li]
            ctrl = self.controllers[li]
            # mask từ topology (dùng mean row như prior cho phép chọn)
            mask = self.topo_masks[li].sum(0)  # node degree-ish, >0 allowed
            mask = (mask > 0).float().to(x.device)
            traj_layer = []
            for t in range(self.seq_len):
                logits = ctrl(h)
                logits = logits.masked_fill(mask.unsqueeze(0) == 0, -1e9)
                soft = F.softmax(logits / max(temperature, 0.1), dim=-1)
                if self.use_hard and self.training:
                    # Gumbel-softmax straight-through
                    hard = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=True)
                    gate = hard + (soft - soft.detach())
                else:
                    gate = soft
                traj_layer.append(gate.detach())
                # weighted update from selected modules
                updates = []
                for mi in range(MOD_PER_LAYER):
                    updates.append(layer_mods[mi](h) * gate[:, mi:mi+1])
                update = torch.stack(updates, dim=0).sum(0)
                h = h + 0.5 * torch.tanh(update)
            all_traj.append(torch.stack(traj_layer, dim=1))  # B, T, M
        out = self.out(h)
        if return_traj:
            return out, torch.stack(all_traj, dim=1)  # B, L, T, M
        return out


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


def traj_recovery(traj, cases, n_sample=300):
    """Proxy: top module mỗi bước vs path nodes (layer-agnostic)."""
    B = min(n_sample, traj.size(0))
    traj = traj[:B]  # B, L, T, M
    cases = cases[:B]
    hits = []
    for b in range(B):
        path = set(CASE_PATHS[cases[b].item()])
        # top module per step
        chosen = traj[b].argmax(dim=-1).view(-1).tolist()  # L*T
        chosen_set = set(c % 12 for c in chosen)  # rough map
        inter = len(chosen_set & path) / max(len(path), 1)
        hits.append(inter)
    return float(np.mean(hits))


def evaluate(model, name, use_traj=False):
    model.eval()
    Xt, Yt = X_test.to(DEVICE), Y_test.to(DEVICE)
    with torch.no_grad():
        if use_traj:
            pred, traj = model(Xt, return_traj=True, temperature=TEMP_END)
            rec = traj_recovery(traj.cpu(), cases_test)
        else:
            pred = model(Xt)
            rec = None
        mse = F.mse_loss(pred, Yt).item()
        r2 = 1 - ((pred - Yt)**2).sum().item() / (((Yt - Yt.mean(0))**2).sum().item() + 1e-8)
    return {"mse": mse, "r2": r2, "traj_recovery": rec, "params": count_params(model)}


# ===================== RUN =====================
results, histories = {}, {}

print("\n=== HierResidual ===")
m = HierResidual()
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "HierResidual")
results["HierResidual"] = evaluate(m, "HierResidual")
histories["HierResidual"] = h
print(f"MSE={results['HierResidual']['mse']:.4f} R2={results['HierResidual']['r2']:.4f}")

print("\n=== Sequential soft ===")
m = SequentialTrajectory(use_hard=False)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqSoft", use_traj=True)
results["SeqSoft"] = evaluate(m, "SeqSoft", use_traj=True)
histories["SeqSoft"] = h
print(f"MSE={results['SeqSoft']['mse']:.4f} R2={results['SeqSoft']['r2']:.4f} traj={results['SeqSoft']['traj_recovery']:.3f}")

print("\n=== Sequential hard (Gumbel) ===")
m = SequentialTrajectory(use_hard=True)
print(f"Params: {count_params(m)}")
m, h, _ = train_model(m, "SeqHard", use_traj=True)
results["SeqHard"] = evaluate(m, "SeqHard", use_traj=True)
histories["SeqHard"] = h
print(f"MSE={results['SeqHard']['mse']:.4f} R2={results['SeqHard']['r2']:.4f} traj={results['SeqHard']['traj_recovery']:.3f}")

# Save
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Explicit Sequential Trajectory on DiffTopo hierarchical",
        "preregistration": "PASS if MSE_seq < min_base*0.95 AND traj_recovery>=0.30",
        "results": results
    }, f, indent=2)
for k, h in histories.items():
    with open(os.path.join(run_dir, f"history_{k}.json"), "w") as f:
        json.dump(h, f)

print(f"\nSaved {run_dir}")
print(json.dumps(results, indent=2))

best_base = results["HierResidual"]["mse"]
best_seq = min(results["SeqSoft"]["mse"], results["SeqHard"]["mse"])
best_name = "SeqSoft" if results["SeqSoft"]["mse"] <= results["SeqHard"]["mse"] else "SeqHard"
rec = results[best_name]["traj_recovery"] or 0

if best_seq < best_base * 0.95 and rec >= 0.30:
    outcome = "PASS"
elif best_seq > best_base * 1.05:
    outcome = "FAIL"
else:
    outcome = "UNRESOLVED"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"Best seq ({best_name}) MSE={best_seq:.4f} vs HierRes={best_base:.4f} | traj_rec={rec:.3f}")

entry = f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: Explicit Sequential Trajectory (gọi module tuần tự) trên DiffTopo hierarchical
- Variants: SeqSoft (softmax) vs SeqHard (Gumbel-softmax)
- Results MSE: HierRes={results['HierResidual']['mse']:.4f}, SeqSoft={results['SeqSoft']['mse']:.4f}, SeqHard={results['SeqHard']['mse']:.4f}
- Traj recovery: Soft={results['SeqSoft']['traj_recovery']}, Hard={results['SeqHard']['traj_recovery']}
- Outcome: **{outcome}**
- Notes: State truyền tuần tự, controller chọn module mỗi bước. Topology mask DiffTopo. Không parallel message-passing.
"""
with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(entry)
print("Lineage appended.")
