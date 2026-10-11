# TNR P1R — Đăng ký trước phân rã hard routing và semantic alignment

**Định danh:** `TNR_P1R_HARD_ROUTING_SEMANTIC_ALIGNMENT_DECOMPOSITION_PREREGISTRATION`
**Mã study:** `TNR_P1R_HSA1`
**Ngày:** 2026-10-09
**Chế độ:** Measurement / Confirmatory mechanism decomposition
**Trạng thái:** PREREGISTERED — IMPLEMENTATION VÀ EXECUTION CHƯA ĐƯỢC PHÉP

## 1. Nguồn khoa học và ranh giới kế thừa

P1-VDR1 đã kết thúc hợp lệ với **FAIL**. Audit sau FAIL định vị vùng nguyên nhân vào hard discrete routing và semantic specialization, nhưng chưa tách được:

1. lỗi do huấn luyện hard bằng Gumbel straight-through;
2. lỗi do hard/sparse argmax khi suy luận;
3. lỗi do module không tự hình thành semantic alignment nếu chỉ dùng task loss.

P1R-HSA1 không được phép đảo verdict P1, không được tune trên P1 TEST và không tái sử dụng P1 như development set.

Bằng chứng phát triển từ `from-grok`, bao gồm tới `7e989131395eb019a896694cd1833043ab149bc2`, chỉ được dùng để khóa design constraints trước study này.

## 2. Bài học từ Grok được tiêu thụ trước khi khóa thiết kế

P1R-HSA1 khóa các bài học sau:

1. **Observability:** route phải quan sát được trực tiếp từ input. Không được chọn route bằng biến latent độc lập với input.
2. **Không collision:** số module semantic bằng đúng số atomic operation. Không dùng ánh xạ kiểu `op % n_mod`.
3. **Matched parameters:** không tăng module width/count giữa các treatment cell.
4. **Semantic supervision control:** dùng đúng hai mức `λ ∈ {0, 0.15}`. `λ=0.15` được chọn từ development evidence; không thử thêm λ trong study.
5. **Không siết prior:** không dùng `λ=0.25`, align-map prior, curriculum hay teacher forcing; các thăm dò Grok cho thấy những thay đổi này có thể làm xấu hoặc tạo confound.
6. **Không architecture search:** không thêm KV, Spark, Spiral, Winnow, Mold, typed modules hoặc topology add-in trong P1R-HSA1.
7. **Multi-seed:** mọi verdict dùng đúng 5 fresh world seed đã khóa.
8. **Soft↔hard counterfactual:** cùng một trọng số phải được đánh giá ở cả soft và hard mode; không so hai architecture khác nhau rồi quy toàn bộ khác biệt cho hardness.
9. **Không topology-mask confound:** P1R-HSA1 bỏ no-self-loop mask ở tất cả router cell. Mỗi bước có đủ 8 lựa chọn; true route vẫn không lặp. Việc lặp module của model được tính là route error, không bị chặn bằng rule.
10. **Không chọn winner trên DEV/TEST:** cả bốn treatment model đều bắt buộc train/eval; không dùng DEV để chọn λ hoặc train mode.

## 3. Câu hỏi khoa học

### Q1 — Hard-training penalty
Giữ evaluation mềm, việc huấn luyện bằng Gumbel straight-through hard có làm giảm task performance so với softmax training không?

### Q2 — Hard-inference penalty
Giữ nguyên cùng trọng số, chuyển từ soft dense evaluation sang hard argmax sparse execution có tạo suy giảm task performance đáng kể không?

### Q3 — Semantic-supervision effect
Route supervision `λ=0.15` có làm tăng semantic route recovery và làm oracle-forced route trở nên hữu ích/ít gây hại hơn mà không phá task performance không?

### Q4 — Residual sparse viability
Sau khi semantic alignment được kiểm soát, hard sparse execution còn giữ được chất lượng đủ gần soft evaluation và fixed baseline để đáng mở successor tối ưu hóa hay không?

## 4. Thế giới dữ liệu mới

### 4.1 Atomic transforms

Giữ family của P1 để cô lập cơ chế:

- 8 atomic transforms `T0..T7`;
- payload dimension = 16;
- output = 4 chiều đầu;
- route length = 4;
- route không lặp atomic operation;
- target không có noise;
- mỗi input chứa payload + bốn route-context token 8 chiều.

Công thức transform giữ nguyên P1:

[
T_i(h)=h+0.5	anh(hW_i+b_i)
]

### 4.2 Fresh route namespace

Tag dùng để sort route:

`TNR_P1R_HSA1|r1-r2-r3-r4`

Toàn bộ `8P4 = 1680` route được sort theo SHA256 của serialization trên.

Split cố định:

- TRAIN = 1176 route;
- DEV = 168 route;
- TEST = 336 route.

Hash route list:

- TRAIN: `3730d7e60ee0a34f12b13e461b7cd4ca8cb7b55e89be73bce3c53c72434934c2`
- DEV: `dcacc18cacd68fededa2bc61b75106eb31139d3405a60732621212470c771adb`
- TEST: `0583c121e213e87942cafbe2874125de570fbaedcf13c0be862355c7a9d361ef`

Mỗi split phải phủ đủ 8 node và 56 directed edges.

### 4.3 Samples

4 payload/route:

- TRAIN = 4704;
- DEV = 672;
- TEST = 1344.

Payload: `Normal(0, 0.7)`, PCG64, deterministic.

### 4.4 Fresh confirmatory seeds

Seed list được khóa từ SHA256 của chuỗi `TNR_P1R_HSA1_CONFIRM_SEEDS`, không chọn theo outcome:

`[505696, 446531, 474580, 615716, 186202]`

Các seed bị cấm:

- P1 confirmatory: `[314159,271828,161803,141421,173205]`;
- Grok/development: `[42,7,123,99,2024]`.

Với world seed `s`:

- transform seed = `s + 1000`;
- context seed = `s + 2000`;
- payload seed = `s + 3000`;
- model-init seed = `s + 4000`;
- minibatch seed = `model_init + 1`.

## 5. Kiến trúc được khóa

### 5.1 Fixed baselines

Giữ A/B của P1 để có continuity:

- A: FixedMLP, expected params = 19,588;
- B: FixedResidual, expected params = 19,738.

### 5.2 UnifiedRouter U

Một architecture duy nhất cho mọi treatment:

- input 48 → hidden 32;
- 8 AtomicModule, mỗi module `32→32→32`;
- controller nhận `[hidden32 + context8]`, `40→24→8`;
- 4 routing steps;
- residual update `h ← h + module_update`;
- output `32→4`;
- expected trainable params = **19,780**;
- không topology mask;
- không no-self-loop mask;
- không teacher forcing;
- không entropy penalty;
- không curriculum;
- không KV/add-in.

Mọi treatment cell phải có **byte-identical architecture definition và cùng parameter count**. Chỉ train gate mode và semantic-loss coefficient được thay đổi.

## 6. Ma trận can thiệp

Chỉ train bốn model router trên mỗi world:

| Model | Train gate | Semantic λ |
|---|---|---:|
| S0 | softmax | 0 |
| S15 | softmax | 0.15 |
| H0 | Gumbel ST hard | 0 |
| H15 | Gumbel ST hard | 0.15 |

Mỗi model sau train phải được evaluate hai lần trên cùng frozen weights:

- **soft-eval:** softmax mixture của cả 8 modules;
- **hard-eval:** argmax, chỉ execute đúng 1 module mỗi step.

Do đó có 8 evaluation cells từ 4 trained models, nhưng chỉ 4 training treatments.

### 6.1 Soft train

[
g_t = softmax(logits_t)
]

Forward update là weighted sum của 8 module outputs.

### 6.2 Hard train

[
g_t = GumbelSoftmax(logits_t,	au=1,hard=True)
]

Straight-through estimator; không anneal temperature.

### 6.3 Semantic supervision

Với `λ=0.15`:

[
L = MSE(y,hat y) + 0.15 cdot
rac{1}{4}sum_t CE(logits_t, route_t)
]

True route label là atomic-operation ID trực tiếp 0..7. Không modulo, không learned align map trong loss.

Với `λ=0`: chỉ MSE.

## 7. Training protocol

Giống P1 trừ các treatment đã ghi rõ:

- CPU deterministic;
- PyTorch deterministic algorithms;
- Adam lr = 0.001;
- betas = (0.9, 0.999);
- eps = 1e-8;
- weight decay = 0;
- batch = 128;
- max = 60 epochs;
- early stop theo DEV task MSE;
- improvement > 1e-12;
- patience = 10;
- restore best DEV task-MSE state;
- không dùng route metric để chọn epoch;
- không dùng TEST để chọn model/hyperparameter.

Tất cả A/B/S0/S15/H0/H15 phải chạy đủ cho mỗi world seed.

## 8. Semantic mapping và diagnostic

Để so công bằng supervised và unsupervised models:

1. trên DEV, lấy argmax route prediction;
2. fit Hungarian mapping true-label ↔ learned-label;
3. freeze mapping;
4. áp dụng mapping lên TEST.

Ngoài mapped route accuracy, report thêm direct-label accuracy cho S15/H15.

### 8.1 Mapping stability

DEV được chia deterministic thành hai nửa theo row parity.

Fit Hungarian mapping độc lập trên DEV-even và DEV-odd.

[
MappingStability =
rac{# atomic labels có cùng mapped module}{8}
]

Không dùng TEST để fit hoặc chọn mapping.

## 9. Causal route interventions

Trên hard-eval của mỗi router:

- native hard route;
- oracle-forced route theo DEV mapping;
- wrong-forced route với true label cyclic shift `i→(i+1) mod 8`.

Report:

- `oracle/native`;
- `wrong/native`;
- route-position accuracy;
- exact-route accuracy;
- mapping stability.

Intervention không được dùng để retrain.

## 10. Primary metrics

Cho từng world và treatment:

- TEST MSE;
- TEST NMSE, population variance `unbiased=False`;
- DEV best MSE;
- mapped route-position accuracy;
- exact-route accuracy;
- direct-label accuracy nếu λ=0.15;
- mapping stability;
- oracle/native;
- wrong/native;
- hard active calls/sample;
- module-use distribution.

### 10.1 Causal decomposition ratios

Cho mỗi λ:

**Training-hardness ratio**

[
TH_lambda =

rac{NMSE(Hlambda, soft eval)}
     {NMSE(Slambda, soft eval)}
]

**Inference-hardness ratio, soft-trained**

[
IH^S_lambda =

rac{NMSE(Slambda, hard eval)}
     {NMSE(Slambda, soft eval)}
]

**Inference-hardness ratio, hard-trained**

[
IH^H_lambda =

rac{NMSE(Hlambda, hard eval)}
     {NMSE(Hlambda, soft eval)}
]

Tính ratio từng seed trước, sau đó lấy mean 5 seed.

## 11. Preregistered mechanism gates

Một mechanism gate chỉ PASS nếu đạt effect threshold **và cùng hướng ít nhất 4/5 seed**.

### G1 — HARD_TRAINING_PENALTY

PASS nếu một trong hai λ có:

- mean `TH_λ >= 1.25`;
- ≥4/5 seed có `TH_λ > 1.0`.

### G2 — HARD_INFERENCE_PENALTY

PASS nếu một trong các `IH` có:

- mean `IH >= 1.25`;
- ≥4/5 seed có `IH > 1.0`.

### G3 — SEMANTIC_ALIGNMENT_REPAIR

So λ=0.15 với λ=0 trong cùng train mode, dùng hard-eval semantics.

PASS nếu ít nhất một train mode thỏa tất cả:

- mapped route-position accuracy tăng ≥ **0.15 absolute**;
- mapping stability tăng ≥ **0.25 absolute** hoặc đạt ≥ **0.75**;
- mean oracle/native ≤ **1.10**;
- mean wrong/native ≥ **1.10**;
- hard-eval NMSE không xấu hơn **1.10×** cell λ=0 tương ứng;
- ≥4/5 seed có route-position accuracy cải thiện.

### G4 — HARD_SPARSE_VIABILITY

PASS nếu cell hard-eval tốt nhất, được xác định **sau khi tính toàn bộ cell nhưng không dùng để tune**, thỏa:

- mean NMSE ≤ **1.10×** best fixed baseline A/B;
- hard active calls/sample = exactly 4;
- route-position accuracy ≥ **0.60**;
- oracle/native ≤ **1.10**;
- wrong/native ≥ **1.10**;
- thắng hoặc hòa best fixed baseline trong ≥4/5 seed.

G4 chỉ là diagnostic authorization gate; không được dùng để tuyên bố TNR tổng quát.

## 12. Study adjudication

### PASS — MECHANISM_LOCALIZED

Nếu validity PASS và ít nhất một trong G1/G2/G3 PASS.

PASS có nghĩa P1R đã nhân quả hóa được ít nhất một phần của failure region.

### FAIL — NO_LOCALIZED_MECHANISM

Nếu validity PASS và:

- G1, G2, G3 đều FAIL;
- mọi mean hardness ratio nằm trong `[0.90, 1.10]`;
- semantic supervision tăng route-position accuracy < 0.05 ở cả soft-train và hard-train.

### UNRESOLVED

Mọi trường hợp hợp lệ còn lại.

G4 được báo riêng để quyết định có đáng mở successor sparse-hard hay không.

## 13. Successor routing theo outcome

- **G1 PASS, G2 FAIL:** ưu tiên sửa training estimator/relaxation; hard inference có thể giữ.
- **G1 FAIL, G2 PASS:** bottleneck nằm ở hard sparse inference; xem top-k/staged hardening nhưng phải preregister mới.
- **G3 PASS:** semantic specialization cần supervision hoặc inductive bias; successor phải giữ semantic control.
- **G3 FAIL + G1/G2 PASS:** hardness là nguyên nhân chính, không phải label semantics.
- **G4 PASS:** cho phép mở fresh confirmatory study về lợi ích sparse routing.
- **G4 FAIL:** không được tuyên bố sparse routing đã phục hồi; successor chỉ được tiếp tục nếu G1/G2/G3 đã định vị được repair rõ.
- **Study FAIL:** đóng nhánh SparseHardRouter hiện tại; không architecture-search trên P1R TEST.

## 14. Validity gates

Tất cả phải PASS trước scientific verdict:

1. exact route hashes như mục 4;
2. TRAIN/DEV/TEST full-route disjoint trong P1R;
3. mỗi split phủ 8 nodes + 56 directed edges;
4. exact 5 fresh seeds;
5. không dùng seed P1/Grok;
6. A/B/U parameter count đúng khóa;
7. bốn trained router cell dùng cùng architecture bytes;
8. λ chỉ `0` hoặc `0.15`;
9. không curriculum/teacher forcing/add-in;
10. TEST không dùng cho epoch/model/hyperparameter choice;
11. source/config hashes khóa trước TEST đầu tiên;
12. hard-eval execute đúng 4 module calls/sample;
13. cả soft-eval và hard-eval chạy trên **cùng frozen weights** của mỗi trained cell;
14. P1/P1R TEST không được dùng làm development loop sau khi nhìn outcome.

Nếu validity gate FAIL: **NO SCIENTIFIC VERDICT** và không append PASS/FAIL vào lineage.

## 15. Những gì study này không được phép kết luận

Ngay cả khi PASS, study này không chứng minh:

- K8 là topology tối ưu;
- sparse routing tốt hơn attention/MoE nói chung;
- TNR scale lên LLM;
- semantic supervision là cần thiết cho mọi task;
- Grok PureC và P1R là cùng architecture;
- P1 verdict cũ bị đảo.

## 16. Trạng thái sau preregistration

Sau khi tài liệu này được commit/push canonical:

**Bước duy nhất được phép:**
`TNR_P1R_HARD_ROUTING_SEMANTIC_ALIGNMENT_DECOMPOSITION_IMPLEMENTATION_STATIC_PREFLIGHT_AND_EXECUTION_LOCK`

Bước đó mới được implement unified router, zero-science fixtures, exact parameter/MAC checks, split hashes, semantic-loss tests, soft↔hard same-weight checks và TEST firewall.

**P1R scientific execution chưa được phép.**