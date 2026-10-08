# Báo cáo: C–C–Spiral–C–C–F

**Run:** partial 2/3 seeds (seed 123 timeout — stack 6 lớp chậm)  
**Outcome:** **UNRESOLVED** (partial)

## Kiến trúc
```
C → C → Spiral(xoay ốc) → C → C → F
```
- Spiral: Givens rotation theo cặp dims + radial force (θ learnable)
- Full ở cuối (all-to-all)

## Kết quả (2 seeds)

| Model | OOD MSE | Traj | Params |
|-------|---------|------|--------|
| HierResidual | 0.128 | — | 40k |
| PureC (CCC) | **0.127** | 0.39 | 55k |
| **Spiral stack** | 0.128 | **0.53** | 94k |

## Nhận xét
1. **Traj recovery tăng rõ** (0.39 → 0.53) — lớp xoáy có thể làm “pseudo-traj” hoặc phase dễ khớp LCS hơn (cần thận trọng giải thích).
2. OOD **không tốt hơn** PureC / residual (hòa).
3. Param gấp ~2.4× residual → không parity.
4. Khớp phân tích trước: xoáy giúp **phase / quỹ đạo state**, không tự động giảm OOD trên path thẳng multi-hop.

## Kết luận tạm trong chuỗi B
- Spiral **không FAIL** về OOD (không phá hỏng).
- Chưa PASS (OOD không vượt 5%).
- Lợi ích chính thấy được: traj metric cao hơn — có thể do bias xoay, cần ablation (chỉ spiral vs full stack).

## Artifact
- Code: `code/tnr_spiral_stack.py`
- Log: `r/uns/spiral_stack_log.txt`
