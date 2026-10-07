#!/usr/bin/env python3
"""TNR P1-VDR1 core implementation.

Implements the preregistered generator, model families, parameter/MAC accounting,
routing metrics, and causal interventions for TNR P1.

Scientific firewall:
- Importing this module never materializes TEST samples.
- TEST samples are materialized only through materialize_split(..., "TEST"),
  which the static preflight intentionally never calls.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

P1_TAG = "TNR_P1_VDR1"
N_ATOMIC = 8
ROUTE_LEN = 4
PAYLOAD_DIM = 16
CONTEXT_DIM = 8
INPUT_DIM = PAYLOAD_DIM + ROUTE_LEN * CONTEXT_DIM
OUTPUT_DIM = 4

TRAIN_ROUTE_COUNT = 1176
DEV_ROUTE_COUNT = 168
TEST_ROUTE_COUNT = 336
PAYLOADS_PER_ROUTE = 4

LOCKED_ROUTE_HASHES = {
    "TRAIN": "54a3e16e19a46354bd77928cd2f02713f80c8166416cf87fbe5c56ec1eba91e3",
    "DEV": "b6bb4d6be8f4f86d28d6c6cf75961a10b11f66d35d7ae37301848f3719273cf7",
    "TEST": "75fa374ea83fb8c185f759324d0d3f5727ea6808f68e9f358adae0910ca3bfca",
}

CONFIRMATORY_WORLD_SEEDS = (314159, 271828, 161803, 141421, 173205)
BANNED_DEVELOPMENT_SEEDS = (42, 7, 123, 99, 2024)


def _route_serialization(route: Sequence[int]) -> str:
    return f"{P1_TAG}|" + "-".join(str(int(v)) for v in route)


def _route_sort_key(route: Sequence[int]) -> str:
    return hashlib.sha256(_route_serialization(route).encode("utf-8")).hexdigest()


def all_routes_sorted() -> List[Tuple[int, int, int, int]]:
    routes = list(itertools.permutations(range(N_ATOMIC), ROUTE_LEN))
    return sorted(routes, key=_route_sort_key)


def split_routes() -> Dict[str, List[Tuple[int, int, int, int]]]:
    routes = all_routes_sorted()
    return {
        "TRAIN": routes[:TRAIN_ROUTE_COUNT],
        "DEV": routes[TRAIN_ROUTE_COUNT:TRAIN_ROUTE_COUNT + DEV_ROUTE_COUNT],
        "TEST": routes[TRAIN_ROUTE_COUNT + DEV_ROUTE_COUNT:],
    }


def route_list_hash(routes: Sequence[Sequence[int]]) -> str:
    payload = "\n".join(",".join(str(int(v)) for v in route) for route in routes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def route_coverage(routes: Sequence[Sequence[int]]) -> Tuple[set[int], set[Tuple[int, int]]]:
    nodes: set[int] = set()
    edges: set[Tuple[int, int]] = set()
    for route in routes:
        nodes.update(int(v) for v in route)
        edges.update((int(a), int(b)) for a, b in zip(route[:-1], route[1:]))
    return nodes, edges


def verify_route_contract() -> Dict[str, object]:
    splits = split_routes()
    observed = {name: route_list_hash(routes) for name, routes in splits.items()}
    if observed != LOCKED_ROUTE_HASHES:
        raise AssertionError(f"Route hash mismatch: {observed}")
    if set(splits["TRAIN"]) & set(splits["DEV"]):
        raise AssertionError("TRAIN/DEV overlap")
    if set(splits["TRAIN"]) & set(splits["TEST"]):
        raise AssertionError("TRAIN/TEST overlap")
    if set(splits["DEV"]) & set(splits["TEST"]):
        raise AssertionError("DEV/TEST overlap")
    coverage = {}
    for name, routes in splits.items():
        nodes, edges = route_coverage(routes)
        coverage[name] = {"nodes": len(nodes), "edges": len(edges)}
        if len(nodes) != 8 or len(edges) != 56:
            raise AssertionError(f"{name} coverage invalid: {coverage[name]}")
    return {"hashes": observed, "coverage": coverage}


@dataclass(frozen=True)
class WorldSeeds:
    world: int
    transform: int
    context: int
    payload: int
    model_init: int


def derive_world_seeds(world_seed: int) -> WorldSeeds:
    if world_seed not in CONFIRMATORY_WORLD_SEEDS:
        raise ValueError(f"World seed {world_seed} is not preregistered")
    if world_seed in BANNED_DEVELOPMENT_SEEDS:
        raise ValueError("Development seed is forbidden for P1 confirmation")
    return WorldSeeds(
        world=world_seed,
        transform=world_seed + 1000,
        context=world_seed + 2000,
        payload=world_seed + 3000,
        model_init=world_seed + 4000,
    )


def generate_true_transforms(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    W = rng.normal(0.0, 0.12, size=(N_ATOMIC, PAYLOAD_DIM, PAYLOAD_DIM))
    b = np.zeros((N_ATOMIC, PAYLOAD_DIM), dtype=np.float64)
    for i in range(N_ATOMIC):
        d1 = i % PAYLOAD_DIM
        d2 = (3 * i + 1) % PAYLOAD_DIM
        W[i, d1, d1] += 1.0
        W[i, d2, d2] += 0.85
        b[i, d1] = 0.25 * ((i % 5) - 2)
    return W.astype(np.float64), b


def generate_context_codebook(seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    raw = rng.normal(0.0, 1.0, size=(N_ATOMIC, CONTEXT_DIM))
    q, r = np.linalg.qr(raw)
    diag = np.diag(r)
    signs = np.where(diag < 0.0, -1.0, 1.0)
    q = q * signs[np.newaxis, :]
    return q.astype(np.float64)


def _apply_true_route(payload: np.ndarray, route: Sequence[int], W: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = payload.astype(np.float64, copy=True)
    for idx in route:
        h = h + 0.5 * np.tanh(h @ W[int(idx)] + b[int(idx)])
    return h[:OUTPUT_DIM]


def _payload_rng_advanced_to_split(payload_seed: int, split: str) -> np.random.Generator:
    split = split.upper()
    if split not in {"TRAIN", "DEV", "TEST"}:
        raise ValueError(split)
    rng = np.random.Generator(np.random.PCG64(payload_seed))
    if split in {"DEV", "TEST"}:
        n_train = TRAIN_ROUTE_COUNT * PAYLOADS_PER_ROUTE
        rng.normal(0.0, 0.7, size=(n_train, PAYLOAD_DIM))
    if split == "TEST":
        n_dev = DEV_ROUTE_COUNT * PAYLOADS_PER_ROUTE
        rng.normal(0.0, 0.7, size=(n_dev, PAYLOAD_DIM))
    return rng


@dataclass
class SplitData:
    x: torch.Tensor
    y: torch.Tensor
    routes: torch.Tensor
    payload: torch.Tensor
    context_tokens: torch.Tensor


def materialize_split(world_seed: int, split: str, *, allow_test: bool = False) -> SplitData:
    split = split.upper()
    if split == "TEST" and not allow_test:
        raise PermissionError("TEST materialization is locked until confirmatory execution")
    seeds = derive_world_seeds(world_seed)
    routes = split_routes()[split]
    W, b = generate_true_transforms(seeds.transform)
    codes = generate_context_codebook(seeds.context)
    rng = _payload_rng_advanced_to_split(seeds.payload, split)

    n = len(routes) * PAYLOADS_PER_ROUTE
    payloads = rng.normal(0.0, 0.7, size=(n, PAYLOAD_DIM)).astype(np.float64)
    route_arr = np.empty((n, ROUTE_LEN), dtype=np.int64)
    context = np.empty((n, ROUTE_LEN, CONTEXT_DIM), dtype=np.float64)
    targets = np.empty((n, OUTPUT_DIM), dtype=np.float64)

    row = 0
    for route in routes:
        for _ in range(PAYLOADS_PER_ROUTE):
            route_arr[row] = route
            context[row] = codes[np.asarray(route, dtype=np.int64)]
            targets[row] = _apply_true_route(payloads[row], route, W, b)
            row += 1

    flat_context = context.reshape(n, ROUTE_LEN * CONTEXT_DIM)
    x = np.concatenate([payloads, flat_context], axis=1)
    return SplitData(
        x=torch.from_numpy(x.astype(np.float32)),
        y=torch.from_numpy(targets.astype(np.float32)),
        routes=torch.from_numpy(route_arr),
        payload=torch.from_numpy(payloads.astype(np.float32)),
        context_tokens=torch.from_numpy(context.astype(np.float32)),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FixedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 72
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_DIM, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
            nn.Linear(width, width),
        ])
        self.out = nn.Linear(width, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.out(h)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.l2(torch.tanh(self.l1(h)))


class FixedResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 46
        self.input_proj = nn.Linear(INPUT_DIM, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(4)])
        self.out = nn.Linear(width, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class AtomicModule(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(h)))


class RouterBase(nn.Module):
    hidden = 32

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, self.hidden)
        self.modules_bank = nn.ModuleList([AtomicModule(self.hidden) for _ in range(N_ATOMIC)])
        self.controller = nn.Sequential(
            nn.Linear(self.hidden + CONTEXT_DIM, 24),
            nn.Tanh(),
            nn.Linear(24, N_ATOMIC),
        )
        self.out = nn.Linear(self.hidden, OUTPUT_DIM)

    @staticmethod
    def _context_from_x(x: torch.Tensor) -> torch.Tensor:
        return x[:, PAYLOAD_DIM:].reshape(-1, ROUTE_LEN, CONTEXT_DIM)

    def _logits(self, h: torch.Tensor, context_t: torch.Tensor, forbidden: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.controller(torch.cat([h, context_t], dim=-1))
        if forbidden is not None:
            logits = logits.clone()
            logits.scatter_(1, forbidden[:, None], float("-inf"))
        return logits


class DenseSoftRouter(RouterBase):
    def forward(self, x: torch.Tensor, *, return_route: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        context = self._context_from_x(x)
        selected = []
        for t in range(ROUTE_LEN):
            logits = self._logits(h, context[:, t])
            weights = torch.softmax(logits, dim=-1)
            outputs = torch.stack([m(h) for m in self.modules_bank], dim=1)
            update = torch.sum(weights.unsqueeze(-1) * outputs, dim=1)
            h = h + update
            selected.append(torch.argmax(weights, dim=-1))
        y = self.out(h)
        if return_route:
            return y, torch.stack(selected, dim=1)
        return y


class SparseHardRouter(RouterBase):
    def __init__(self) -> None:
        super().__init__()
        self.last_active_calls_per_sample: float | None = None

    def _apply_selected(self, h: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        update = torch.zeros_like(h)
        for module_idx in torch.unique(selected).tolist():
            mask = selected == int(module_idx)
            if torch.any(mask):
                update[mask] = self.modules_bank[int(module_idx)](h[mask])
        return update

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_route: bool = False,
        forced_route: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        context = self._context_from_x(x)
        selected_steps = []
        prev_selected: torch.Tensor | None = None
        active_calls = torch.zeros(x.size(0), device=x.device)

        for t in range(ROUTE_LEN):
            if forced_route is not None:
                selected = forced_route[:, t].long()
                if prev_selected is not None and torch.any(selected == prev_selected):
                    raise ValueError("Forced route violates no-self-loop K8 contract")
                update = self._apply_selected(h, selected)
            else:
                logits = self._logits(h, context[:, t], prev_selected)
                if self.training:
                    gate = F.gumbel_softmax(logits, tau=max(float(temperature), 0.1), hard=True, dim=-1)
                    outputs = torch.stack([m(h) for m in self.modules_bank], dim=1)
                    update = torch.sum(gate.unsqueeze(-1) * outputs, dim=1)
                    selected = torch.argmax(gate.detach(), dim=-1)
                else:
                    selected = torch.argmax(logits, dim=-1)
                    update = self._apply_selected(h, selected)
            h = h + update
            selected_steps.append(selected)
            active_calls += 1.0
            prev_selected = selected

        self.last_active_calls_per_sample = float(active_calls.mean().item())
        y = self.out(h)
        route = torch.stack(selected_steps, dim=1)
        if return_route:
            return y, route
        return y


LOCKED_PARAMETER_COUNTS = {
    "A": 19588,
    "B": 19738,
    "C": 19780,
    "D": 19780,
}

LOCKED_MAC_ESTIMATES = {
    "A": 19296,
    "B": 19320,
    "D": 14464,
}


def build_models(model_seed: int) -> Dict[str, nn.Module]:
    torch.manual_seed(int(model_seed))
    return {
        "A": FixedMLP(),
        "B": FixedResidual(),
        "C": DenseSoftRouter(),
        "D": SparseHardRouter(),
    }


def verify_model_contract(model_seed: int = CONFIRMATORY_WORLD_SEEDS[0] + 4000) -> Dict[str, object]:
    models = build_models(model_seed)
    counts = {name: count_trainable_parameters(model) for name, model in models.items()}
    if counts != LOCKED_PARAMETER_COUNTS:
        raise AssertionError(f"Parameter mismatch: {counts}")
    ratios = {
        "A_over_D": counts["A"] / counts["D"],
        "B_over_D": counts["B"] / counts["D"],
        "C_over_D": counts["C"] / counts["D"],
    }
    if not all(0.95 <= value <= 1.05 for value in ratios.values()):
        raise AssertionError(f"Parity failed: {ratios}")
    if LOCKED_MAC_ESTIMATES["D"] > 1.20 * min(LOCKED_MAC_ESTIMATES["A"], LOCKED_MAC_ESTIMATES["B"]):
        raise AssertionError("MAC contract failed")
    return {"parameters": counts, "ratios": ratios, "macs": dict(LOCKED_MAC_ESTIMATES)}


def fit_dev_module_mapping(true_routes: torch.Tensor, predicted_routes: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    true_np = true_routes.detach().cpu().numpy().reshape(-1)
    pred_np = predicted_routes.detach().cpu().numpy().reshape(-1)
    confusion = np.zeros((N_ATOMIC, N_ATOMIC), dtype=np.int64)
    for t, p in zip(true_np, pred_np):
        confusion[int(t), int(p)] += 1
    rows, cols = linear_sum_assignment(-confusion)
    true_to_pred = np.empty(N_ATOMIC, dtype=np.int64)
    pred_to_true = np.empty(N_ATOMIC, dtype=np.int64)
    true_to_pred[rows] = cols
    pred_to_true[cols] = rows
    return true_to_pred, pred_to_true


def route_position_accuracy(
    true_routes: torch.Tensor,
    predicted_routes: torch.Tensor,
    pred_to_true: np.ndarray,
) -> float:
    pred_np = predicted_routes.detach().cpu().numpy()
    mapped = pred_to_true[pred_np]
    true_np = true_routes.detach().cpu().numpy()
    return float(np.mean(mapped == true_np))


def exact_route_accuracy(
    true_routes: torch.Tensor,
    predicted_routes: torch.Tensor,
    pred_to_true: np.ndarray,
) -> float:
    pred_np = predicted_routes.detach().cpu().numpy()
    mapped = pred_to_true[pred_np]
    true_np = true_routes.detach().cpu().numpy()
    return float(np.mean(np.all(mapped == true_np, axis=1)))


def forced_module_routes(
    true_routes: torch.Tensor,
    true_to_pred: np.ndarray,
    *,
    wrong: bool = False,
) -> torch.Tensor:
    labels = true_routes.detach().cpu().numpy()
    if wrong:
        labels = (labels + 1) % N_ATOMIC
    forced = true_to_pred[labels]
    return torch.from_numpy(forced.astype(np.int64)).to(true_routes.device)


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    var = torch.var(target, unbiased=False).item()
    return float(mse / max(var, 1e-12))
