# TNR P1R-HSA1 — Phụ lục đăng ký trước A: khóa tự do triển khai

**Định danh:** `TNR_P1R_HSA1_PREREGISTRATION_AMENDMENT_A`
**Ngày:** 2026-10-11
**Trạng thái:** CLARIFICATION-ONLY / PRE-TEST / PRE-EXECUTION-LOCK

Phụ lục này không đổi câu hỏi, treatment, seed, route split, λ, threshold hay gate của P1R-HSA1. Nó chỉ khóa các freedom triển khai còn chưa được nêu đủ cụ thể trước khi static preflight và trước mọi TEST.

## A1. Khởi tạo và randomness

Trong mỗi world:

- A/B dùng deterministic initialization theo `model_init_seed`.
- S0/S15/H0/H15 đều được tạo từ **byte-identical initial router weights** bằng cùng `model_init_seed`.
- bốn router dùng cùng deterministic minibatch order với seed `model_init_seed + 1`.
- H0 và H15 dùng cùng Gumbel random stream seed `model_init_seed + 2`.
- S0/S15 không có stochastic gate.
- Không treatment-specific seed offset nào khác được phép.

## A2. Early stopping

Để tách hard-training khỏi hard-inference, **mọi router treatment** S0/S15/H0/H15 chọn epoch bằng:

**DEV task MSE dưới soft-eval trên cùng current weights.**

Route metrics, semantic loss, hard-eval MSE và TEST không được dùng để chọn epoch.

A/B tiếp tục chọn epoch bằng native DEV task MSE.

## A3. Hai pha execution

Một one-shot P1R hợp lệ phải chạy theo hai pha:

1. **TRAIN/FREEZE:** train và freeze A/B/S0/S15/H0/H15 cho **cả 5 world** chỉ bằng TRAIN/DEV.
2. **TEST:** chỉ sau khi toàn bộ 30 frozen model đã được freeze và manifest checkpoint được ghi xong mới được materialize TEST đầu tiên.

Không được xen kẽ TEST world sớm với training world sau.

## A4. Soft↔hard evaluation

Với từng treatment trong từng world:

- soft-eval và hard-eval dùng **chính xác cùng state_dict frozen**;
- không fine-tune giữa hai eval mode;
- soft-eval execute weighted mixture của cả 8 modules mỗi step;
- hard-eval dùng argmax và execute đúng 1 module mỗi step;
- hard-eval phải có đúng 4 active module calls/sample.

## A5. DEV mapping

Mapping được fit riêng cho từng treatment **và từng eval mode** trên DEV prediction ở chính eval mode đó.

- mapped route metric trên TEST dùng mapping DEV tương ứng;
- mapping stability fit independent trên DEV even/odd rows ở cùng eval mode;
- hard causal intervention dùng mapping của **hard-eval DEV**;
- direct-label accuracy của S15/H15 không dùng Hungarian mapping.

## A6. Ratio và baseline aggregation

Tất cả causal decomposition ratio được tính **per-world trước**, sau đó mean trên đúng 5 world.

Best fixed baseline ở mỗi world là:

`min(NMSE_A, NMSE_B)`

G4 relative-to-fixed dùng ratio hard-cell/best-fixed theo world rồi mean 5 world.

G4 “wins or ties” dùng:

`NMSE_hard <= NMSE_best_fixed + 1e-12`

Best hard cell cho G4 là cell có mean relative-to-fixed thấp nhất sau khi tất cả hard-eval cells đã được tính; lựa chọn này chỉ phục vụ adjudication, không được dùng để retrain/tune.

## A7. G3 semantics

G3 so `λ=0.15` với `λ=0` **trong cùng train mode** và dùng hard-eval semantic metrics.

Route-position gain và mapping-stability gain được tính per-world rồi lấy mean; directional seed count dùng per-world route-position improvement > 0.

## A8. Raw evidence cho independent QA

Execution phải freeze:

- state_dict của A/B/S0/S15/H0/H15 cho từng world;
- training metadata/best epoch;
- per-world TEST metrics;
- DEV mappings;
- per-treatment soft/hard metrics;
- checkpoint SHA256 manifest;
- final aggregate/adjudication.

Formal scientific close vẫn cần independent QA; runner execution không được tự append `lineage.md`.