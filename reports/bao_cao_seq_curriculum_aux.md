# Báo cáo: SeqHard + Curriculum path-length + Auxiliary sequence loss

**Run ID:** 20261006_162524  
**Outcome:** **FAIL** (đối với biến thể CurrAux)

## Thiết kế
- Giữ SeqHard + DiffTopo hierarchical
- **Curriculum**: 10 epoch đầu chỉ train path ngắn, sau đó mở full
- **Aux loss**: soft CE khuyến khích mass trên module xuất hiện trong latent path (λ=0.02)
- Control: SeqHard plain (không curr/aux)

## Kết quả

| Model | ID MSE | OOD MSE | Traj seq recovery OOD |
|-------|--------|---------|------------------------|
| HierResidual | **0.089** | **0.1745** | — |
| SeqHard_plain | 0.090 | 0.1793 | 0.281 |
| SeqHard_CurrAux | 0.186 | **0.3093** | 0.308 |

- CurrAux **làm xấu đi rõ rệt** cả ID và OOD (OOD 0.309 vs 0.174).
- Traj recovery tăng nhẹ (0.28 → 0.31) nhưng không bù được mất mát accuracy.
- SeqHard plain vẫn gần residual, không vượt.
- **Outcome: FAIL** (CurrAux kém baseline > 5%).

## Phân tích
1. Auxiliary loss dạng node-presence soft CE **không khớp** với cần sequence order → model bị kéo lệch.
2. Curriculum path-length chưa đủ mạnh hoặc bị aux làm hỏng.
3. Traj recovery có tín hiệu tăng nhưng trade-off accuracy quá lớn.
4. Kết quả này **có giá trị khoa học**: cho thấy prior/aux không đúng cách có thể hại hơn lợi (đúng tinh thần TNR: FAIL có kiểm soát vẫn hữu ích).

## Kết luận tạm thời
- Sequential trajectory plain vẫn là biến thể tốt nhất hiện tại (gần residual).
- Curriculum + aux dạng hiện tại **không phải** hướng đi tiếp.
- Cần thiết kế lại auxiliary (nếu còn dùng) theo hướng sequence matching thật (LCS differentiable proxy hoặc teacher forcing nhẹ có schedule), hoặc chấp nhận rằng trên task family này residual vẫn mạnh hơn và chuyển sang phân tích giới hạn.

## Artifact
- Code: `code/tnr_seq_ood_curriculum.py`
- Run: `r/uns/run_20261006_162524/`
- Lineage đã append **FAIL**
