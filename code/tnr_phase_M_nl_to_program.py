#!/usr/bin/env python3
"""
Phase M — Bridge toy → real: NL → Program → Execute

Lab PASS dùng parse token cứng. Bước 1 thoát toy:
  Input = câu tiếng Anh template/paraphrase (không phải token DSL thuần)
  Pipeline A (structured): NL encoder → predict program slots → Execute trên grid
  Pipeline B (flat): NL encoder → predict final (x,y) residual

Train: atomic + twice paraphrases
OOD: after / opposite paraphrases (composition systematic)

Preregister:
  Setup: Prog pipeline ID cell-acc >= 0.50
  PASS: Prog OOD > Flat OOD + 0.05 AND Prog OOD >= 0.50 AND wins >= 2/3
  FAIL: Prog OOD + 0.05 < Flat
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

GRID = 5
MOVES = ["north", "south", "east", "west"]
MOVE2DELTA = {
    "north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0),
}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE2ID = {m: i for i, m in enumerate(MOVES)}

# Word vocab for NL
WORDS = ["<pad>", "<unk>",
         "go", "move", "walk", "head", "step",
         "north", "south", "east", "west",
         "up", "down", "left", "right",
         "twice", "two", "times", "again",
         "opposite", "opposite_of", "reverse", "other_way",
         "after", "then", "following", "before",
         "the", "direction", "once"]
W2ID = {w: i for i, w in enumerate(WORDS)}
MAX_NL = 12
N_WORD = len(WORDS)

HIDDEN = 96
BATCH_SIZE = 64
N_TRAIN = 4000
N_ID = 500
N_OOD = 700
EPOCHS = 30
LR = 1e-3

def clip(x, y):
    return max(0, min(GRID - 1, x)), max(0, min(GRID - 1, y))

def apply_moves(x0, y0, moves):
    x, y = x0, y0
    for m in moves:
        dx, dy = MOVE2DELTA[m]
        x, y = clip(x + dx, y + dy)
    return x, y

def expand_program(direction, twice=False, opposite=False, after_dir=None):
    """Build move list from structured slots."""
    moves = []
    if after_dir is not None:
        # X after Y => Y then X
        moves.extend(expand_program(after_dir, False, False, None))
    d = OPP[direction] if opposite else direction
    unit = [d]
    if twice:
        unit = unit * 2
    moves.extend(unit)
    return moves

# NL templates → (direction, twice, opposite, after_dir or None)
def templates_train():
    ex = []
    for m in MOVES:
        for phr in [
            f"go {m}", f"move {m}", f"walk {m}", f"head {m}",
            f"step {m}", f"go {m} once",
        ]:
            ex.append((phr, m, False, False, None))
        for phr in [
            f"go {m} twice", f"move {m} two times", f"walk {m} twice",
            f"go {m} again", f"step {m} twice",
        ]:
            ex.append((phr, m, True, False, None))
    return ex

def templates_ood():
    ex = []
    for m in MOVES:
        for phr in [
            f"go opposite {m}", f"move opposite_of {m}",
            f"walk reverse {m}", f"head other_way {m}",
            f"go the opposite direction {m}",
        ]:
            ex.append((phr, m, False, True, None))
        for a in MOVES:
            if a == m:
                continue
            for phr in [
                f"go {m} after {a}", f"move {m} then {a}",
                f"walk {m} following {a}", f"head {m} after {a}",
                f"go {m} after go {a}",
            ]:
                # "m after a" => a then m
                ex.append((phr, m, False, False, a))
    return ex

TRAIN_TPL = templates_train()
OOD_TPL = templates_ood()
print(f"Train templates={len(TRAIN_TPL)} OOD={len(OOD_TPL)}")

def tokenize(text):
    ids = []
    for w in text.lower().split():
        ids.append(W2ID.get(w, W2ID["<unk>"]))
    ids = ids + [0] * MAX_NL
    return ids[:MAX_NL]

def sample_dataset(templates, n, seed):
    rng = np.random.RandomState(seed)
    X_nl, X_pos, Y_pos = [], [], []
    # labels for program head
    lab_dir, lab_twice, lab_opp, lab_after, lab_has_after = [], [], [], [], []
    for _ in range(n):
        phr, direction, twice, opposite, after_dir = templates[rng.randint(0, len(templates))]
        x0, y0 = rng.randint(0, GRID), rng.randint(0, GRID)
        moves = expand_program(direction, twice, opposite, after_dir)
        xf, yf = apply_moves(x0, y0, moves)
        X_nl.append(tokenize(phr))
        X_pos.append([x0, y0])
        Y_pos.append([xf, yf])
        lab_dir.append(MOVE2ID[direction])
        lab_twice.append(1 if twice else 0)
        lab_opp.append(1 if opposite else 0)
        lab_has_after.append(1 if after_dir is not None else 0)
        lab_after.append(MOVE2ID[after_dir] if after_dir else 0)
    return {
        "nl": torch.tensor(X_nl, dtype=torch.long),
        "pos": torch.tensor(X_pos, dtype=torch.float32),
        "y": torch.tensor(Y_pos, dtype=torch.float32),
        "dir": torch.tensor(lab_dir, dtype=torch.long),
        "twice": torch.tensor(lab_twice, dtype=torch.long),
        "opp": torch.tensor(lab_opp, dtype=torch.long),
        "has_after": torch.tensor(lab_has_after, dtype=torch.long),
        "after": torch.tensor(lab_after, dtype=torch.long),
    }

def cell_acc(pred, y):
    pr = pred.round().clamp(0, GRID - 1)
    yt = y.round().clamp(0, GRID - 1)
    return (pr == yt).all(-1).float().mean().item()

class NLEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_WORD, HIDDEN, padding_idx=0)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(HIDDEN * 2, HIDDEN)
    def forward(self, nl):
        e = self.emb(nl)
        lengths = (nl != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], -1)
        return self.proj(h)

class FlatNL2Pos(nn.Module):
    """Unstructured: NL + pos → final cell."""
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.pos_proj = nn.Linear(2, HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
    def forward(self, nl, pos):
        h = self.enc(nl)
        return self.mlp(torch.cat([h, self.pos_proj(pos)], -1))

class NL2ProgramExecute(nn.Module):
    """Structured: NL → program slots → execute on grid."""
    def __init__(self):
        super().__init__()
        self.enc = NLEncoder()
        self.dir_head = nn.Linear(HIDDEN, len(MOVES))
        self.twice_head = nn.Linear(HIDDEN, 2)
        self.opp_head = nn.Linear(HIDDEN, 2)
        self.has_after_head = nn.Linear(HIDDEN, 2)
        self.after_head = nn.Linear(HIDDEN, len(MOVES))

    def program_logits(self, nl):
        h = self.enc(nl)
        return {
            "dir": self.dir_head(h),
            "twice": self.twice_head(h),
            "opp": self.opp_head(h),
            "has_after": self.has_after_head(h),
            "after": self.after_head(h),
        }

    def execute_from_logits(self, logits, pos):
        B = pos.size(0)
        dirs = logits["dir"].argmax(-1)
        twice = logits["twice"].argmax(-1)
        opp = logits["opp"].argmax(-1)
        has_after = logits["has_after"].argmax(-1)
        afters = logits["after"].argmax(-1)
        outs = []
        for i in range(B):
            d = MOVES[int(dirs[i])]
            t = bool(int(twice[i]))
            o = bool(int(opp[i]))
            ad = MOVES[int(afters[i])] if int(has_after[i]) == 1 else None
            moves = expand_program(d, t, o, ad)
            x0, y0 = int(round(float(pos[i, 0]))), int(round(float(pos[i, 1])))
            xf, yf = apply_moves(x0, y0, moves)
            outs.append([float(xf), float(yf)])
        return torch.tensor(outs, device=pos.device, dtype=torch.float32)


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = sample_dataset(TRAIN_TPL, N_TRAIN, seed)
    idd = sample_dataset(TRAIN_TPL, N_ID, seed + 15)
    ood = sample_dataset(OOD_TPL, N_OOD, seed + 20)

    def train_flat():
        model = FlatNL2Pos().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl, pos, y = tr["nl"].to(DEVICE), tr["pos"].to(DEVICE), tr["y"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                F.mse_loss(model(nl[idx], pos[idx]), y[idx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(nl[:512], pos[:512]), y[:512]).item()
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    def train_prog():
        model = NL2ProgramExecute().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        nl = tr["nl"].to(DEVICE)
        best, st = float("inf"), None
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(N_TRAIN, device=DEVICE)
            total = 0
            nb = 0
            for i in range(0, N_TRAIN, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                logits = model.program_logits(nl[idx])
                loss = (
                    F.cross_entropy(logits["dir"], tr["dir"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["twice"], tr["twice"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["opp"], tr["opp"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["has_after"], tr["has_after"].to(DEVICE)[idx])
                    + F.cross_entropy(logits["after"], tr["after"].to(DEVICE)[idx])
                )
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            val = total / max(nb, 1)
            if val < best:
                best = val
                st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if st:
            model.load_state_dict(st)
        return model

    out = {"seed": seed}
    flat = train_flat()
    prog = train_prog()
    flat.eval()
    prog.eval()
    with torch.no_grad():
        for name, data in [("id", idd), ("ood", ood)]:
            nl, pos, y = data["nl"].to(DEVICE), data["pos"].to(DEVICE), data["y"].to(DEVICE)
            out[f"Flat_{name}"] = cell_acc(flat(nl, pos), y)
            logits = prog.program_logits(nl)
            out[f"Prog_{name}"] = cell_acc(prog.execute_from_logits(logits, pos), y)
    out["win"] = out["Prog_ood"] > out["Flat_ood"] + 0.02
    return out


all_results = []
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r = run_seed(s)
    all_results.append(r)
    print(f"  Flat ID={r['Flat_id']:.3f} OOD={r['Flat_ood']:.3f}", flush=True)
    print(f"  Prog ID={r['Prog_id']:.3f} OOD={r['Prog_ood']:.3f} win={r['win']}", flush=True)

def ms(key):
    v = [r[key] for r in all_results]
    return float(np.mean(v)), float(np.std(v))

print("\n===== M NL→PROGRAM SUMMARY =====")
print(f"Flat: ID={ms('Flat_id')[0]:.3f} OOD={ms('Flat_ood')[0]:.3f}")
print(f"Prog: ID={ms('Prog_id')[0]:.3f} OOD={ms('Prog_ood')[0]:.3f}")

fid, fod = ms("Flat_id")[0], ms("Flat_ood")[0]
pid, pod = ms("Prog_id")[0], ms("Prog_ood")[0]
wins = sum(r["win"] for r in all_results)
setup_ok = pid >= 0.50

if not setup_ok:
    outcome, note = "UNRESOLVED", "Prog ID < 0.50"
elif pod > fod + 0.05 and pod >= 0.50 and wins >= 2:
    outcome, note = "PASS", "NL→program→execute beats flat on compositional OOD"
elif pod + 0.05 < fod:
    outcome, note = "FAIL", "structured pipeline worse than flat"
else:
    outcome, note = "UNRESOLVED", "gap insufficient"

print(f"setup_ok={setup_ok} wins={wins}/3")
print(f"=== OUTCOME: {outcome} === ({note})")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"/home/workdir/artifacts/r/uns/run_{run_id}"
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "experiment": "Phase M NL→Program→Execute bridge",
        "seeds": SEEDS,
        "summary": {k: list(ms(k)) for k in ["Flat_id","Flat_ood","Prog_id","Prog_ood"]},
        "per_seed": all_results,
        "outcome": outcome,
        "note": note,
    }, f, indent=2)

with open("/home/workdir/artifacts/Lineage.md", "a") as f:
    f.write(f"""
## Run {run_id}
- Date: {datetime.now().isoformat()}
- Experiment: **Phase M** NL→Program→Execute (paraphrase bridge toy→real)
- Flat ID={fid:.3f} OOD={fod:.3f} | Prog ID={pid:.3f} OOD={pod:.3f}
- Outcome: **{outcome}** ({note})
- Step 1 of real-world ladder: structured NL slots + grounded execute
""")
print(f"Saved {run_dir}")
