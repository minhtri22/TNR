#!/usr/bin/env python3
"""Tiền kiểm tra tĩnh/DEV cho TNR P1-VDR1.

Script này tuyệt đối không materialize mẫu TEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnr_p1_vdr1 import (
    CONFIRMATORY_WORLD_SEEDS,
    DEV_ROUTE_COUNT,
    INPUT_DIM,
    LOCKED_PARAMETER_COUNTS,
    PAYLOADS_PER_ROUTE,
    SparseHardRouter,
    build_models,
    derive_world_seeds,
    fit_dev_module_mapping,
    generate_context_codebook,
    materialize_split,
    route_position_accuracy,
    verify_model_contract,
    verify_route_contract,
)


def main() -> int:
    out = {
        "preflight": "TNR_P1_IMPLEMENTATION_STATIC_PREFLIGHT",
        "test_materialized": False,
        "checks": {},
    }

    out["checks"]["route_contract"] = verify_route_contract()
    out["checks"]["model_contract"] = verify_model_contract()

    world = CONFIRMATORY_WORLD_SEEDS[0]
    seeds = derive_world_seeds(world)
    codebook = generate_context_codebook(seeds.context)
    gram = codebook @ codebook.T
    ortho_err = float(np.max(np.abs(gram - np.eye(8))))
    if ortho_err > 1e-10:
        raise AssertionError(f"Context codebook is not orthonormal: {ortho_err}")
    out["checks"]["context_codebook_max_orthogonality_error"] = ortho_err

    train = materialize_split(world, "TRAIN")
    dev = materialize_split(world, "DEV")
    if train.x.shape != (1176 * PAYLOADS_PER_ROUTE, INPUT_DIM):
        raise AssertionError(train.x.shape)
    if dev.x.shape != (DEV_ROUTE_COUNT * PAYLOADS_PER_ROUTE, INPUT_DIM):
        raise AssertionError(dev.x.shape)
    out["checks"]["train_shape"] = list(train.x.shape)
    out["checks"]["dev_shape"] = list(dev.x.shape)

    try:
        materialize_split(world, "TEST")
        raise AssertionError("TEST firewall failed")
    except PermissionError:
        out["checks"]["test_firewall"] = "PASS"

    models = build_models(seeds.model_init)
    x = train.x[:16]
    y = train.y[:16]
    for name, model in models.items():
        model.train()
        pred = model(x)
        if pred.shape != y.shape:
            raise AssertionError(f"{name} shape mismatch")
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
        if not finite:
            raise AssertionError(f"{name} non-finite gradients")
    out["checks"]["train_fixture_forward_backward"] = "PASS"

    d = models["D"]
    assert isinstance(d, SparseHardRouter)
    d.eval()
    with torch.no_grad():
        _, pred_route = d(dev.x[:64], return_route=True)
    if d.last_active_calls_per_sample != 4.0:
        raise AssertionError(d.last_active_calls_per_sample)
    if pred_route.shape != (64, 4):
        raise AssertionError(pred_route.shape)
    out["checks"]["sparse_active_module_calls_per_sample"] = d.last_active_calls_per_sample

    true_routes = dev.routes[:128]
    permutation_true_to_pred = np.array([3, 5, 7, 1, 6, 0, 4, 2], dtype=np.int64)
    synthetic_pred = torch.from_numpy(permutation_true_to_pred[true_routes.numpy()])
    _, pred_to_true = fit_dev_module_mapping(true_routes, synthetic_pred)
    acc = route_position_accuracy(true_routes, synthetic_pred, pred_to_true)
    if abs(acc - 1.0) > 1e-12:
        raise AssertionError(f"Hungarian mapping fixture failed: {acc}")
    out["checks"]["hungarian_mapping_fixture_accuracy"] = acc

    out["checks"]["locked_parameter_counts"] = LOCKED_PARAMETER_COUNTS
    out["status"] = "PASS"

    evidence_dir = ROOT / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    (evidence_dir / "p1_static_preflight.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
