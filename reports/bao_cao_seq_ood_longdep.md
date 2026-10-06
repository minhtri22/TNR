# Báo cáo: Sequential Trajectory trên Variable Long-Dependency + OOD Composition

**Run ID:** 20261006_161628  
**Outcome:** **UNRESOLVED**

## Thiết kế task mới

- **Train**: 5 family, path length 3–4  
- **OOD**: 5 family, path length 5–6 + composition / reorder chưa thấy lúc train  
- Metric chính (preregistered): **OOD MSE** + **Traj sequence recovery (LCS)**

## Kết quả

| Model | ID MSE | OOD MSE | Traj seq recovery (OOD) |
|-------|--------|---------|--------------------------|
| HierResidual | **0.0951** | **0.1775** | — |
| SeqSoft | 0.0985 | 0.1840 | 0.268 |
| SeqHard | 0.0967 | 0.1809 | 0.249 |

- Tất cả model đều bị **OOD gap** rõ (ID ~0.095–0.098 → OOD ~0.18).
- SeqHard tốt hơn SeqSoft một chút trên OOD, nhưng **vẫn kém HierResidual**.
- Traj sequence recovery OOD ~0.25 (dưới ngưỡng 0.35).
- **Outcome: UNRESOLVED**

## Phân tích

1. Task đã khó hơn thực sự: residual cũng bị OOD degradation mạnh → đúng hướng.
2. Sequential trajectory chưa tận dụng được lợi thế generalization trên route mới.
3. Controller vẫn chưa học được sequence đủ tốt để transfer sang path dài / composition lạ.
4. LCS recovery thấp cho thấy predicted trajectory chưa khớp mạnh với latent path.

## Ý nghĩa cho TNR

Đây là lần đầu tiên ta thấy **OOD gap rõ ràng** trên mọi model. Residual vẫn robust hơn. Adaptive sequential routing cần thêm tín hiệu (auxiliary trajectory loss nhẹ, hoặc curriculum length) hoặc inductive bias mạnh hơn về compositionality để vượt residual trên OOD.

## Next step gợi ý

- Thêm auxiliary LCS / sequence matching loss nhẹ (chỉ trên train paths).
- Hoặc curriculum: train path length tăng dần.
- Giữ SeqHard + DiffTopo hierarchical.

## Artifact
- Code: `code/tnr_seq_ood_longdep.py`
- Run: `r/uns/run_20261006_161628/`
- Lineage đã append.
