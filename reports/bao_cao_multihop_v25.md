# Báo cáo: Phương án B — Multi-hop v2.5 (middle difficulty)

**Run ID:** 20261008_053654  
**Outcome:** **UNRESOLVED**

## Độ khó đạt được

| Version | Residual OOD R² | Đánh giá |
|---------|-----------------|----------|
| v1 | ~0.09 | Quá khó |
| v2 | ~0.76 | Quá dễ |
| **v2.5** | **0.56** | Gần target (0.3–0.5), hơi cao một chút |

## Kết quả (3 seeds)

| Model | OOD MSE ↓ | Traj recovery | OOD R² |
|-------|-----------|---------------|--------|
| HierResidual | 0.1286 ± 0.004 | — | 0.565 |
| **PureC** | **0.1255 ± 0.001** | **0.404 ± 0.059** | 0.575 |
| Spark | 0.1285 ± 0.000 | 0.453 ± 0.053 | 0.565 |

- PureC thắng OOD **3/3 seeds**
- Traj PureC ≈ 0.40 (đạt ngưỡng)
- Ratio OOD PureC/HR ≈ 0.976 → **chưa** < 0.95 → UNRESOLVED
- Spark traj cao hơn nhưng OOD không hơn residual

## Ý nghĩa trong chuỗi PASS/FAIL

1. Độ khó đã vào vùng có headroom (R² ~0.56, không còn 0.09 hay 0.76).
2. PureC **nhất quán hơn residual** trên OOD (3/3) nhưng gap nhỏ (~2.4%).
3. Spark (thêm Full) không cải thiện OOD trên task multi-hop này.
4. Traj đo được ổn (~0.40–0.45).

## Next gợi ý trong chuỗi B

- Giữ v2.5 task; siết PureC (SEQ_LEN, capacity) hoặc nới residual parity để xem gap có mở không.
- Hoặc tăng OOD độ khó nhẹ (path dài hơn) trên cùng operator để residual R² xuống ~0.4.
- Ghi nhận: **PureC > Spark** trên multi-hop → Full layer không phải ingredient mặc định.

## Artifact
- Code: `code/tnr_task_multihop_v25.py`
- Run: `r/uns/run_20261008_053654/`
- Lineage đã append
