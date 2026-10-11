# TNR P1 — Kiểm toán quy kết cơ chế sau FAIL

**Định danh:** `TNR_P1_POST_FAIL_MECHANISM_ATTRIBUTION_AUDIT`
**Ngày:** 2026-10-09
**Chế độ:** kiểm toán chẩn đoán trên bằng chứng đóng băng (post-fail frozen-evidence diagnostic)
**Không huấn luyện lại. Không tạo TEST mới. Không thay đổi verdict P1.**

## 1. Câu hỏi

Sau FAIL hợp lệ của P1-VDR1, cơ chế thất bại nằm chủ yếu ở đâu trong bốn vùng:

1. tối ưu định tuyến hard (hard-routing optimization);
2. chuyên biệt hóa mô-đun (module specialization);
3. khả định nhãn route (route-label identifiability);
4. kiến trúc thực thi thưa (sparse execution).

Bằng chứng phát triển mới tại `from-grok@7e989131395eb019a896694cd1833043ab149bc2` được dùng như chẩn đoán phụ, không phải bằng chứng xác nhận cho P1.

## 2. Bằng chứng P1 đóng băng

P1 dùng DenseSoftRouter C và SparseHardRouter D có cùng:

- 19.780 tham số;
- input/context;
- ngân hàng 8 atomic module;
- controller family;
- bốn bước route.

Khác biệt cốt lõi là chế độ chọn module mềm/dense so với hard/sparse.

Trên 5 world xác nhận:

- C mean DEV MSE = **0.0057578115**;
- D mean DEV MSE = **0.0517286234**;
- D/C trên DEV ≈ **9.08×**;
- C mean TEST NMSE = **0.0060885948**;
- D mean TEST NMSE = **0.0551542487**;
- D/C trên TEST ≈ **9.15×**.

C chỉ bằng khoảng **17.6%** lỗi của fixed baseline tốt nhất theo tỷ lệ trung bình, trong khi D thua fixed baseline ở 5/5 world.

Điều này rất quan trọng: chênh lệch C–D đã xuất hiện mạnh trên DEV, trước khi đánh giá compositional TEST. Vì vậy FAIL của D không thể quy chủ yếu cho riêng generalization ngoài phân phối.

## 3. Route semantics và can thiệp

D có:

- route-position accuracy trung bình = **0.250744**;
- exact-route accuracy trung bình ≈ **0.003720**;
- wrong/native = **2.288867×**;
- oracle/native = **1.642876×**.

Diễn giải:

- wrong/native > 2 chứng minh route **không vô nghĩa**; thay route sai làm computation xấu mạnh;
- nhưng oracle/native > 1 cho thấy ép theo route thật sau ánh xạ Hungarian từ DEV lại làm xấu đáng kể;
- do đó module/router đã đồng thích nghi thành một mã nội bộ khác, không phải decomposition sạch “mỗi learned module = một atomic transform thật”.

## 4. Route-label identifiability

P1 đã dùng Hungarian mapping trên DEV để loại bỏ hoán vị nhãn tùy ý.

Nếu lỗi chỉ là permutation label, thì sau mapping:

- route recovery phải tăng mạnh hơn;
- đặc biệt oracle-forced route phải xấp xỉ hoặc tốt hơn native.

Thực tế oracle/native = 1.642876×.

**Kết luận:** khả định nhãn thuần túy (pure permutation identifiability) **không đủ** để giải thích FAIL.

## 5. Quy kết hard-routing / sparse execution

C và D cùng tham số nhưng khác soft-vs-hard regime, và D đã xấu hơn khoảng 9× ngay trên DEV.

Do đó vùng nguyên nhân chính được định vị vào:

**hard discrete routing + quá trình đồng thích nghi router–module dưới lựa chọn rời rạc.**

Tuy nhiên frozen evidence P1 không lưu hai counterfactual cần thiết:

- soft-trained → hard-eval;
- hard-trained → soft-eval.

Vì vậy audit **chưa tách được** phần nào đến từ:

- tối ưu Gumbel straight-through khi train;
- hay chính việc argmax/sparse execution khi inference.

Đây là phần còn UNRESOLVED bên trong vùng nguyên nhân đã được định vị.

## 6. Bằng chứng Grok `7e98913`

Run phát triển mới báo PASS khi chuyển:

`PureC + λ=0.15 + n_mod=4 → n_mod=6 (=N_KEYS)`

5-seed complete:

- N4 OOD = **0.44184**, trajectory = **0.4184**;
- N6 OOD = **0.3766**, trajectory = **0.4680**;
- N6/N4 OOD ≈ **0.8523**;
- trajectory tăng **+0.0496**.

Điểm có giá trị: bỏ collision `op % 4` và tạo đủ slot module cho 6 key giúp trajectory bền hơn mà mean OOD không xấu.

Nhưng kết quả này **không phải replication của P1**, vì đồng thời:

- dùng seed phát triển `[42,7,123,99,2024]`;
- có route/layout supervision `λ=0.15`;
- tăng tham số từ 77.227 lên 105.937, khoảng **1.372×**;
- eval dùng soft routing;
- task/architecture khác P1;
- P1 vốn đã có 8 module cho 8 atomic operation, nên **không có collision 8→k** kiểu Grok sửa.

Do đó collision là cơ chế thật trong nhánh Grok, nhưng **không phải nguyên nhân trực tiếp của P1 FAIL**.

## 7. Phán quyết attribution

**Kết quả audit: PASS**

**Phán quyết:**
`FAILURE_LOCALIZED_TO_HARD_DISCRETE_ROUTING_AND_SEMANTIC_SPECIALIZATION_REGION`

Mức hỗ trợ:

| Cơ chế | Quy kết |
|---|---|
| Hard discrete routing / optimization | **STRONG PRIMARY** |
| Module specialization / semantic alignment | **STRONG SECONDARY** |
| Pure label permutation identifiability | **REJECTED AS SOLE CAUSE** |
| Routing irrelevance | **REJECTED** |
| Hard training vs hard sparse inference | **UNRESOLVED, cần successor** |
| op % n_mod collision kiểu Grok | **NOT APPLICABLE TO P1** |

## 8. Quyết định chương trình

Không rerun P1. Không tune trên P1 TEST. Không mở lại verdict P1.

Nhánh SparseHardRouter hiện tại được **đóng như thiết kế đã FAIL**.

Cho phép mở một successor P1R mới, nhưng chỉ để kiểm tra nhân quả tối thiểu giữa:

1. soft-trained → soft-eval;
2. soft-trained → hard-eval;
3. hard-trained → soft-eval;
4. hard-trained → hard-eval;
5. có/không semantic route supervision với ngân sách và tham số được khóa.

P1R phải dùng **fresh confirmation namespace**, không tái sử dụng P1 TEST như tập xác nhận.