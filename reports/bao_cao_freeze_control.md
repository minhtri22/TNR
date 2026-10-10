# Freeze họ Control — ranh giới

**Ngày:** 2026-10-10  
**Status:** **CLOSED**

## Claim

Trên point-mass 2D + skill impulses:
- **Oracle path + true skills** → MSE ≈ noise (I4 PASS).
- **Learned free routing** (Gumbel, TF→free, closed-loop, hybrid rollout+residual) **không** thắng HierResidual (I2–J2 FAIL).
- Bag hint làm HR khó hơn nhưng SkillComp vẫn thua (P1).

**Ranh giới:** continuous dynamics composition đúng khi biết program; OOD discrete routing dưới dynamics + residual shortcut = điểm gãy. Khác discrete-bank / algebraic matrix (đã PASS).

## Không làm thêm trong control

P2 anneal, thêm layer, goal-leak, oracle lúc test.
