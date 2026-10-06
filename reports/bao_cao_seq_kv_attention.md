# Báo cáo: Sequential + Sliding KV per node + Attention styles

**Run ID:** 20261006_164230  
**Outcome:** **UNRESOLVED** (nhưng tín hiệu tích cực rõ)

## Thiết kế
- Controller mỗi bước **attend vào bảng KV** của các module (sliding: overwrite slot khi module được chọn).
- 3 attention styles: scaled_dot, additive (Bahdanau), gated.
- So sánh với HierResidual và SeqHard NoMem.
- Task: Variable Long-Dependency + OOD (giữ nguyên).

## Kết quả OOD

| Model | OOD MSE ↓ | Traj seq recovery |
|-------|-----------|-------------------|
| HierResidual | 0.1764 | — |
| SeqNoMem | 0.1743 | 0.243 |
| KV_scaled_dot | 0.1721 | 0.291 |
| **KV_additive** | **0.1709** | **0.314** |
| KV_gated | 0.1736 | 0.273 |

- **KV_additive lần đầu tiên tốt hơn residual trên OOD** (0.1709 vs 0.1764).
- Traj recovery tăng rõ so với NoMem (0.24 → 0.31).
- Vẫn chưa đạt ngưỡng PASS (cần < 0.95× baseline và recovery ≥ 0.35), nhưng khoảng cách đã đảo chiều.

## Phân tích
1. Sliding KV cho controller “nhớ” state của module đã thăm → hỗ trợ compositionality tốt hơn feed-forward thuần.
2. Additive attention (Bahdanau) phù hợp nhất trong 3 styles trên task này.
3. Đây là **tín hiệu tích cực đầu tiên** của adaptive routing trên OOD kể từ đầu lineage.
4. Param KV cao hơn (~82–96k vs residual 57k) → cần kiểm soát chặt hơn ở run sau.

## Ý nghĩa cho TNR
Hướng “bảng KV trượt theo node + attention” đang mở ra khả năng vượt residual trên OOD. Cần:
- Cân bằng param,
- Tăng recovery (có thể aux nhẹ đúng hướng),
- Chạy multi-seed để xác nhận tín hiệu không phải nhiễu.

## Artifact
- Code: `code/tnr_seq_kv_attention.py`
- Run: `r/uns/run_20261006_164230/`
- Lineage đã append.
