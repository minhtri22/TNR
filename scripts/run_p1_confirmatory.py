#!/usr/bin/env python3
"""Runner xác nhận P1-VDR1.

CẢNH BÁO KHOA HỌC:
- Script từ chối materialize TEST nếu execution lock không tồn tại hoặc hash sai.
- Không được chạy script này trong bước static preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnr_p1_vdr1 import (
    CONFIRMATORY_WORLD_SEEDS,
    SparseHardRouter,
    build_models,
    derive_world_seeds,
    exact_route_accuracy,
    fit_dev_module_mapping,
    forced_module_routes,
    materialize_split,
    normalized_mse,
    route_position_accuracy,
    verify_model_contract,
    verify_route_contract,
)

LOCK_PATH = ROOT / "configs" / "p1_execution_lock.json"
RESULT_DIR = ROOT / "results" / "p1_vdr1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_execution_lock() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError("Execution lock missing; TEST remains forbidden")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "EXECUTION_LOCKED":
        raise RuntimeError("Execution lock is not active")
    for rel, expected in lock["locked_files_sha256"].items():
        path = ROOT / rel
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Lock hash mismatch for {rel}: {actual} != {expected}")
    verify_route_contract()
    verify_model_contract()
    return lock


def set_deterministic_cpu(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def train_model(model, train, dev, model_seed: int, is_sparse_router: bool = False):
    model = model.cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(model_seed + 1)
    n = train.x.size(0)
    batch_size = 128
    best_mse = math.inf
    best_state = None
    stale = 0

    for epoch in range(60):
        model.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train.x[idx], train.y[idx]
            optimizer.zero_grad(set_to_none=True)
            if is_sparse_router:
                pred = model(xb, temperature=1.0)
            else:
                pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_dev = model(dev.x)
            dev_mse = F.mse_loss(pred_dev, dev.y).item()

        if dev_mse < best_mse - 1e-12:
            best_mse = dev_mse
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break

    if best_state is None:
        raise RuntimeError("No best state captured")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_dev_mse": best_mse, "epochs_completed": epoch + 1}


def evaluate_basic(model, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x)
        mse = F.mse_loss(pred, data.y).item()
        nmse = normalized_mse(pred, data.y)
    return {"mse": mse, "nmse": nmse}


def evaluate_sparse_d(model: SparseHardRouter, dev, test):
    model.eval()
    with torch.no_grad():
        _, dev_pred_route = model(dev.x, return_route=True)
        true_to_pred, pred_to_true = fit_dev_module_mapping(dev.routes, dev_pred_route)

        native_pred, test_pred_route = model(test.x, return_route=True)
        native_mse = F.mse_loss(native_pred, test.y).item()
        native_nmse = normalized_mse(native_pred, test.y)
        route_pos = route_position_accuracy(test.routes, test_pred_route, pred_to_true)
        exact_route = exact_route_accuracy(test.routes, test_pred_route, pred_to_true)

        oracle_route = forced_module_routes(test.routes, true_to_pred, wrong=False)
        wrong_route = forced_module_routes(test.routes, true_to_pred, wrong=True)
        oracle_pred = model(test.x, forced_route=oracle_route)
        wrong_pred = model(test.x, forced_route=wrong_route)
        oracle_mse = F.mse_loss(oracle_pred, test.y).item()
        wrong_mse = F.mse_loss(wrong_pred, test.y).item()

    if model.last_active_calls_per_sample != 4.0:
        raise RuntimeError("Sparse inference active-call contract violated")

    return {
        "mse": native_mse,
        "nmse": native_nmse,
        "route_position_accuracy": route_pos,
        "exact_route_accuracy": exact_route,
        "mse_native": native_mse,
        "mse_oracle": oracle_mse,
        "mse_wrong": wrong_mse,
        "wrong_over_native": wrong_mse / native_mse,
        "oracle_over_native": oracle_mse / native_mse,
        "active_module_calls_per_sample": model.last_active_calls_per_sample,
        "true_to_pred_mapping": true_to_pred.tolist(),
        "pred_to_true_mapping": pred_to_true.tolist(),
    }


def adjudicate(worlds: list[dict]) -> dict:
    ratios = []
    wins = 0
    route_accs = []
    wrong_ratios = []
    oracle_ratios = []

    for w in worlds:
        baseline = min(w["A"]["nmse"], w["B"]["nmse"])
        ratio = w["D"]["nmse"] / baseline
        ratios.append(ratio)
        if w["D"]["nmse"] < baseline:
            wins += 1
        route_accs.append(w["D"]["route_position_accuracy"])
        wrong_ratios.append(w["D"]["wrong_over_native"])
        oracle_ratios.append(w["D"]["oracle_over_native"])

    r_mean = float(np.mean(ratios))
    route_mean = float(np.mean(route_accs))
    wrong_mean = float(np.mean(wrong_ratios))
    oracle_mean = float(np.mean(oracle_ratios))

    pass_all = (
        r_mean <= 0.95
        and wins >= 4
        and route_mean >= 0.60
        and wrong_mean >= 1.10
        and oracle_mean <= 1.02
    )
    fail_any = r_mean >= 1.05 or wins <= 1

    if pass_all:
        outcome = "PASS"
    elif fail_any:
        outcome = "FAIL"
    else:
        outcome = "UNRESOLVED"

    return {
        "outcome": outcome,
        "R_mean": r_mean,
        "world_wins": wins,
        "route_position_accuracy_mean": route_mean,
        "wrong_over_native_mean": wrong_mean,
        "oracle_over_native_mean": oracle_mean,
        "per_world_R": ratios,
    }


def main() -> int:
    lock = verify_execution_lock()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    worlds = []
    for world_seed in CONFIRMATORY_WORLD_SEEDS:
        seeds = derive_world_seeds(world_seed)
        set_deterministic_cpu(seeds.model_init)

        # TRAIN/DEV are materialized first. TEST remains untouched until all
        # models for this world finish training and the global execution lock
        # has already been verified above.
        train = materialize_split(world_seed, "TRAIN")
        dev = materialize_split(world_seed, "DEV")

        models = build_models(seeds.model_init)
        train_meta = {}
        trained = {}
        for name in ("A", "B", "C", "D"):
            is_sparse = name == "D"
            trained[name], train_meta[name] = train_model(
                models[name],
                train,
                dev,
                seeds.model_init,
                is_sparse_router=is_sparse,
            )

        # First TEST materialization is authorized only here, after lock check
        # and after model selection is complete.
        test = materialize_split(world_seed, "TEST", allow_test=True)

        world_result = {
            "world_seed": world_seed,
            "training": train_meta,
            "A": evaluate_basic(trained["A"], test),
            "B": evaluate_basic(trained["B"], test),
            "C": evaluate_basic(trained["C"], test),
            "D": evaluate_sparse_d(trained["D"], dev, test),
        }
        worlds.append(world_result)

        (RESULT_DIR / f"world_{world_seed}.json").write_text(
            json.dumps(world_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    verdict = adjudicate(worlds)
    result = {
        "program": "TNR_P1_VDR1",
        "execution_lock": lock,
        "worlds": worlds,
        "verdict": verdict,
    }
    (RESULT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
