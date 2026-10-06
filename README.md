# TNR Research Package

## Cấu trúc
- code/: mã nguồn thí nghiệm
- r/uns/: các run đã lưu (metrics, history, weights)
- reports/: báo cáo khoa học
- Lineage.md: nhật ký append-only PASS/FAIL/UNRESOLVED

## Nội dung chính
Báo cáo: reports/bao_cao_khoa_hoc_TNR_thi_nghiem_toi_thieu.md


## Run tiếp theo (v2)
- Báo cáo: reports/bao_cao_khoa_hoc_TNR_run_tiep_theo.md
- Run: r/uns/run_20261006_150521
- Outcome: UNRESOLVED (R² baseline ~0.38, route F1 thấp, MSE_D không tốt hơn baseline)


## Run v3 (20261006_151416)
- Báo cáo: reports/bao_cao_khoa_hoc_TNR_v3.md
- R² baseline đạt ~0.85–0.86 (vượt mục tiêu >0.6)
- Outcome: UNRESOLVED (D không tốt hơn baseline, edge F1 thấp)
- Đã thực hiện: compositional transform, soft+temperature, traj metric, E balanced


## Hierarchical A (20261006_154757)
- Báo cáo: reports/bao_cao_phuong_an_A_hierarchical.md
- SameTopo (ring lặp) vs DiffTopo (sparse→ring→complete)
- Outcome: UNRESOLVED (gated không vượt HierResidual)
- DiffTopo hơi tốt hơn SameTopo nhưng khoảng cách nhỏ

