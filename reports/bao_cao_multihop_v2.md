# Báo cáo: Phương án B — Multi-hop v2 (tune độ khó)

**Run:** partial 2/3 seeds (seed 123 timeout)  
**Outcome:** **UNRESOLVED** + **độ khó lệch mục tiêu**

## Mục tiêu tune
Đưa residual OOD R² về vùng **0.3–0.5** (v1 quá khó ~0.09, cần headroom).

## Thay đổi so với v1
- Operator êm hơn (residual mix mạnh hơn)
- Train path length 2–3; OOD length 4
- Noise thấp hơn

## Kết quả (2 seeds)

| Model | OOD MSE | Traj | OOD R² |
|-------|---------|------|--------|
| HierResidual | **0.130** | — | **0.76** |
| PureC | 0.134 | 0.41 | 0.76 |
| Spark | 0.132 | **0.49** | 0.76 |

## Chẩn đoán
- **Tune quá đà sang dễ**: R² ~0.76 (vượt xa target 0.3–0.5).
- Residual lại ngang hoặc tốt hơn adaptive trên OOD — giống regime task cũ.
- Traj vẫn đo được (Spark 0.49, PureC 0.41).

## Khoảng độ khó cần tìm

| Version | Residual OOD R² | Nhận xét |
|---------|-----------------|----------|
| v1 | ~0.09 | Quá khó, mọi model kém, khó tách architecture |
| **v2** | **~0.76** | Quá dễ, residual thắng lại |
| **v2.5 (cần làm)** | **0.3–0.5** | Vùng headroom để PASS/FAIL tách bias routing |

## Next
Multi-hop **v2.5**: trung gian giữa v1 và v2
- Operator: coupling vừa (giữa hard v1 và soft v2)
- Path train length 3, OOD length 4–5
- Noise vừa
- Kiểm tra residual R² trước khi so full architecture

## Artifact
- Code: `code/tnr_task_multihop_v2.py`
- Log: `r/uns/multihop_v2_log.txt`
