# Báo cáo: So sánh KV_additive vs KV_causal

**Run ID:** 20261006_165820  
**Seeds:** 42, 7

## Thiết kế
- Cùng backbone sliding KV + additive attention + DiffTopo hierarchical.
- **KV_additive**: attend toàn bộ bảng KV.
- **KV_causal**: chỉ attend các slot đã được ghi (written mask) — phong cách autoregressive.

## Kết quả (mean ± std)

| Model | OOD MSE ↓ | Traj recovery |
|-------|-----------|---------------|
| HierResidual | 0.1808 ± 0.002 | — |
| KV_additive | 0.1753 ± 0.002 | 0.306 ± 0.012 |
| **KV_causal** | **0.1731 ± 0.004** | **0.347 ± 0.038** |

- Cả hai đều tốt hơn residual trên OOD.
- **KV_causal** tốt hơn additive một chút về OOD MSE và **rõ rệt hơn về traj recovery** (0.35 vs 0.31).
- Causal đạt gần ngưỡng recovery 0.35.

## Phân tích
1. Causal mask buộc controller chỉ nhìn lịch sử đã viết → phù hợp inductive bias của trajectory tuần tự.
2. Traj recovery cao hơn cho thấy model học thứ tự tốt hơn khi không bị “nhìn tương lai” (slot chưa ghi).
3. Additive full-table vẫn mạnh, nhưng causal mang lại lợi thế đo được trên recovery.

## Kết luận
**KV_causal** là biến thể tốt hơn trong so sánh trực tiếp này. Nên lấy causal làm baseline mới cho các run tiếp theo (multi-seed đầy đủ hơn, cân param, hoặc task khó hơn).

## Artifact
- Code: `code/tnr_kv_additive_vs_causal.py`
- Run: `r/uns/run_20261006_165820/`
- Lineage đã append.
