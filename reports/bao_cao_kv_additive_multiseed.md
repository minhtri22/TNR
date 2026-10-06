# Báo cáo: KV_additive cân bằng param + multi-seed

**Run ID:** 20261006_165236  
**Outcome:** **UNRESOLVED** (tín hiệu OOD tích cực được xác nhận multi-seed)

## Thiết kế
- Cân bằng param: HIDDEN=40, KV_DIM=24, controller nhỏ → KV ≈ 56k vs HierRes ≈ 40k
- Multi-seed: {42, 123, 7}
- Aux tắt (λ=0) vì phiên bản trước hại performance
- Task: Variable Long-Dependency + OOD (giữ nguyên)

## Kết quả (mean ± std trên 3 seeds)

| Model | ID MSE | **OOD MSE** | Traj recovery OOD |
|-------|--------|-------------|-------------------|
| HierResidual | 0.090 ± 0.001 | 0.1866 ± 0.003 | — |
| **KV_additive** | 0.091 ± 0.002 | **0.1791 ± 0.006** | **0.303 ± 0.024** |

- KV tốt hơn residual trên OOD ở **cả 3 seeds**.
- Khoảng cách ~4% (0.179 vs 0.187) → chưa đạt ngưỡng 5% của PASS, nhưng **hướng và độ ổn định đã xác nhận**.
- Traj recovery trung bình 0.30 (gần ngưỡng 0.35).

## Phân tích
1. Tín hiệu OOD của KV_additive **không phải nhiễu** (multi-seed consistent).
2. Cân bằng param vẫn để KV hơi nặng hơn residual; có thể giảm thêm.
3. Aux nhẹ dạng soft path consistency trước đó hại → cần thiết kế lại nếu muốn đẩy recovery.
4. Đây là bằng chứng thực nghiệm mạnh nhất đến nay ủng hộ adaptive routing (sequential + sliding KV + additive attention) trên OOD.

## Kết luận
Hướng KV_additive đã vượt qua kiểm tra multi-seed. Next step hợp lý:
- Giảm param KV thêm (hoặc tăng residual width cho công bằng tuyệt đối),
- Hoặc tăng độ khó OOD / số seed,
- Hoặc thử aux đúng hướng hơn (sequence order loss thay vì node presence).

## Artifact
- Code: `code/tnr_kv_additive_multiseed.py`
- Run: `r/uns/run_20261006_165236/`
- Lineage đã append.
