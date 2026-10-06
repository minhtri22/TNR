# Báo cáo: Hierarchical DiffTopo + Prior tăng có kiểm soát

**Run ID:** 20261006_155655  
**Outcome:** **UNRESOLVED**

## Mục tiêu
Giữ hierarchical DiffTopo (sparse → ring → complete), so sánh 3 mức prior:
- **P0**: temperature + soft top-K (như run A trước)
- **P1**: P0 + entropy penalty (khuyến khích gate tập trung)
- **P2**: P1 + soft curriculum path bias (giảm dần theo epoch)

Không dùng full teacher forcing.

## Kết quả

| Model | MSE | R² | Entropy | Active |
|-------|-----|-----|---------|--------|
| HierResidual | **0.1580** | **0.858** | — | — |
| DiffTopo_static | 0.1602 | 0.856 | — | — |
| P0_light | 0.1629 | 0.854 | 0.544 | — |
| **P1_entropy** | 0.1621 | 0.854 | **0.006** | 0.67 |
| P2_curriculum | 0.1645 | 0.852 | 0.174 | 0.81 |

- Entropy giảm mạnh với P1 (0.544 → 0.006) → router tập trung hơn rõ rệt.
- MSE của P1/P2 vẫn không vượt HierResidual (0.1580).
- **Outcome: UNRESOLVED**

## Phân tích
- Tăng prior (entropy) **thành công làm router thưa và tập trung**.
- Nhưng task compositional hiện tại vẫn được giải tốt nhất bởi pure residual hierarchical → routing adaptive chưa tạo lợi thế accuracy.
- Curriculum soft bias (P2) làm training loss cao hơn và không giúp MSE.

## Kết luận tạm thời
Hierarchical stacking + prior vừa phải đủ để điều khiển sparsity, nhưng **chưa đủ để chứng minh adaptive topological routing vượt fixed-depth/residual** trên task family này.

Hướng tiếp theo có thể cân nhắc:
1. Task khó hơn / dependency dài hơn / OOD route composition.
2. Explicit trajectory controller thay vì parallel message-passing.
3. Hoặc chấp nhận kết quả âm có kiểm soát (FAIL có giá trị) nếu lặp lại trên nhiều seed/task.

## Artifact
- Code: `code/tnr_hierarchical_A_prior.py`
- Run: `r/uns/run_20261006_155655/`
- Lineage đã append.
