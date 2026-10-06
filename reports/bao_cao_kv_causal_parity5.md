# Báo cáo: KV_causal parity (ratio 0.80) + đủ 5 seeds

**Run ID:** 20261006_173757  
**Outcome:** **UNRESOLVED**

## Thiết kế
- HierRes HIDDEN=42, KV HIDDEN=32, KV_DIM=16 → **param ratio 0.80** (KV nhẹ hơn residual)
- Đủ 5 seeds: 42, 7, 123, 99, 2024
- Task & kiến trúc giữ nguyên (causal sliding KV)

## Kết quả

| Seed | HierRes OOD | KV OOD | Traj | Win |
|------|-------------|--------|------|-----|
| 42 | 0.1875 | **0.1794** | 0.278 | ✓ |
| 7 | **0.1743** | 0.1769 | 0.305 | ✗ |
| 123 | 0.1856 | **0.1826** | 0.227 | ✓ |
| 99 | 0.1878 | **0.1865** | 0.243 | ✓ |
| 2024 | 0.1866 | **0.1834** | 0.319 | ✓ |

**Tổng hợp:**
- HierRes OOD: **0.1844 ± 0.005**
- KV_causal OOD: **0.1817 ± 0.003**
- Traj: **0.274 ± 0.035**
- **Wins: 4/5**
- Param ratio: **0.80**

## Đánh giá
- Khi KV **nhẹ hơn** residual, gap OOD thu hẹp còn ~1.5% và 1 seed thua.
- Traj recovery vẫn quanh 0.27.
- Tín hiệu vẫn nghiêng về KV (4/5) nhưng **không đủ mạnh** để PASS theo ngưỡng đã khóa.

## Kết luận
Param parity (thậm chí KV nhỏ hơn) làm lợi thế OOD của KV_causal **mỏng đi**. Tín hiệu còn nhưng fragile. Cần cân nhắc:
- Chấp nhận UNRESOLVED ổn định và chuyển sang task routing-critical hơn, hoặc
- Giữ capacity KV ngang residual và báo cáo rõ trade-off param vs OOD.

## Artifact
- Code: `code/tnr_kv_causal_parity5.py`
- Run: `r/uns/run_20261006_173757/`
- Lineage đã append.
