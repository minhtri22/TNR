# Báo cáo: Phương án B — Multi-hop routing-critical task v1

**Run ID:** 20261008_050920  
**Outcome:** **UNRESOLVED**

## Task mới (routing-critical)

- 6 operator **rời rạc** (không residual cộng dồn mượt): mỗi op ghi slot dims khác nhau + nonlinear.
- Train: path length 3, 5 families.
- OOD: path length 5 + composition/reorder mới.
- Mục tiêu: residual depth khó đoán đúng thứ tự operator.

## Kết quả (3 seeds)

| Model | OOD MSE ↓ | Traj recovery | OOD R² |
|-------|-----------|---------------|--------|
| HierResidual | 0.5035 ± 0.008 | — | ~0.09 |
| **PureC (CCC)** | **0.5006 ± 0.005** | **0.457 ± 0.029** | ~0.10 |
| Spark (CCFC) | 0.5069 ± 0.007 | 0.427 ± 0.053 | ~0.08 |

- **Mọi model đều khó** (R² ~0.07–0.11) → task đã routing-critical hơn task cũ (trước R² > 0.85).
- PureC hơi tốt hơn residual về OOD (wins 2/3), **traj 0.46 ≥ 0.40**.
- Chưa đạt PASS (cần OOD < 0.95×HR = 0.478).
- Spark (thêm Full) **không giúp** trên task này — thậm chí hơi kém.

## Ý nghĩa cho chuỗi PASS/FAIL

| Tín hiệu | Giá trị |
|----------|---------|
| Task đủ khó cho residual | ✓ (R² thấp đồng đều) |
| Traj recovery đo được và > 0.40 | ✓ PureC |
| Adaptive chưa vượt residual đủ mạnh | UNRESOLVED |
| Full layer không tự động giúp | FAIL nhẹ cho Spark trên task này |

Đây là **điểm xuất phát đúng** của phương án B: residual không còn “dễ thắng”, traj có tín hiệu, kiến trúc bắt đầu bị phân hóa bởi task.

## Next trong chuỗi B

1. Làm task **dễ hơn một bậc** (path train length 2–3, OOD length 4, giảm noise) để residual đạt R² ~0.3–0.5 — vùng có headroom.
2. Hoặc tăng capacity / số bước SEQ_LEN cho PureC, giữ residual parity.
3. Preregister lại trên task đã tune độ khó.

## Artifact
- Code: `code/tnr_task_multihop_v1.py`
- Run: `r/uns/run_20261008_050920/`
- Lineage đã append
