# Báo cáo: Explicit Sequential Trajectory (gọi module tuần tự)

**Run ID:** 20261006_160550  
**Outcome:** **UNRESOLVED**

## Mục tiêu
Thay message-passing song song bằng **gọi module tuần tự** (trajectory tường minh) trên cấu trúc hierarchical DiffTopo.

- Controller mỗi bước chọn 1 module (soft hoặc Gumbel-hard).
- State truyền tuần tự qua SEQ_LEN = 4 bước mỗi lớp.
- Topology mask DiffTopo (sparse → ring → complete) giới hạn module được chọn.

## Kết quả

| Model | MSE | R² | Traj recovery | Params |
|-------|-----|-----|---------------|--------|
| HierResidual | **0.1580** | **0.858** | — | 40 204 |
| SeqSoft | 0.1628 | 0.854 | **0.416** | 44 536 |
| SeqHard (Gumbel) | 0.1586 | 0.858 | 0.387 | 44 536 |

- SeqHard gần bằng HierResidual về MSE (0.1586 vs 0.1580).
- Traj recovery ~0.39–0.42 (vượt ngưỡng 0.30 proxy).
- **Chưa đạt PASS** vì MSE không thấp hơn baseline × 0.95.

## Phân tích
- Trajectory tường minh giúp đo recovery rõ ràng hơn parallel gating.
- SeqHard ổn định và gần baseline nhất.
- Vẫn chưa vượt residual thuần về accuracy trên task compositional hiện tại.
- Inductive bias tuần tự + topology mask đủ để controller học một phần latent nodes, nhưng chưa tạo lợi thế compute/accuracy rõ.

## Kết luận
Chuyển sang sequential trajectory là bước tiến về **measurement** (traj recovery đo được) và gần bắt kịp residual. Cần task khó hơn (dependency dài, OOD composition) hoặc auxiliary trajectory loss nhẹ để đẩy recovery và accuracy vượt baseline.

## Artifact
- Code: `code/tnr_sequential_trajectory.py`
- Run: `r/uns/run_20261006_160550/`
- Lineage đã append **UNRESOLVED**
