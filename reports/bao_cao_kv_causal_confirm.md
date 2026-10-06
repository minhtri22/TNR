# Báo cáo: Củng cố KV_causal – multi-seed + param parity

**Run:** partial 4/5 seeds (42, 7, 123, 99) — seed 2024 bị timeout  
**Outcome:** **UNRESOLVED**

## Thiết kế
- 5 seeds dự kiến, hoàn thành 4
- Param: HierRes 32 728 | KV_causal 45 115 (ratio 1.38 — vẫn hơi cao, chưa đạt ±10%)
- HIDDEN=36, KV_DIM=20, controller 28
- Không ledger, không aux
- Task giữ nguyên

## Kết quả (4 seeds)

| Seed | HierRes OOD | KV_causal OOD | Traj | Win |
|------|-------------|---------------|------|-----|
| 42 | 0.1839 | **0.1752** | 0.287 | ✓ |
| 7 | 0.1859 | **0.1792** | 0.244 | ✓ |
| 123 | 0.1886 | **0.1836** | 0.219 | ✓ |
| 99 | 0.1774 | **0.1700** | 0.273 | ✓ |

**Tổng hợp:**
- HierRes OOD: **0.1840 ± 0.004**
- KV_causal OOD: **0.1770 ± 0.005**
- Traj recovery: **0.256 ± 0.026**
- **Wins: 4/4 seeds**
- Ratio OOD: 0.962 (chưa < 0.97)

## Đánh giá theo preregistration
- mean OOD_KV < mean_Hier × 0.97? → 0.962 > 0.97 → **chưa**
- traj ≥ 0.33? → 0.256 → **chưa**
- wins ≥ 4/5? → 4/4 đã đạt hướng

→ **UNRESOLVED**

## Nhận xét
1. **Tín hiệu nhất quán**: KV_causal thắng OOD ở cả 4 seeds — không phải nhiễu.
2. Gap nhỏ (~3.8%), chưa đủ ngưỡng 3% (0.97×) đã preregister.
3. Traj recovery thấp hơn run trước (có thể do giảm capacity khi cân param).
4. Param ratio 1.38 vẫn chưa parity thật sự — cần giảm thêm hoặc tăng residual width.

## Kết luận
Tín hiệu OOD của KV_causal **ổn định qua multi-seed**, nhưng khi siết param thì:
- Gap thu hẹp
- Traj recovery giảm

Cần thêm 1 vòng cân param tốt hơn + hoàn tất seed còn thiếu trước khi tuyên bố PASS hoặc chuyển sang tối ưu khác.

## Artifact
- Code: `code/tnr_kv_causal_confirm.py`
- Log: `r/uns/kv_causal_confirm_log.txt`
- Lineage sẽ append.
