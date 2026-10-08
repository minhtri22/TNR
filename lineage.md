# TNR — Dòng dõi khoa học

File này là sổ dòng dõi khoa học chỉ nối thêm (append-only scientific lineage) của TNR.

Quy tắc:

- chỉ ghi kết quả khoa học;
- ghi cả PASS, FAIL và UNRESOLVED;
- không ghi triển khai hạ tầng, sửa lỗi mã nguồn, lỗi công cụ, lỗi thiết bị hoặc công việc vận hành;
- không sửa/xóa mục cũ; mọi hiệu chỉnh phải nối thêm mục mới;
- đăng ký trước (preregistration) và khóa thực thi (execution lock) không tự động trở thành kết quả khoa học.

---

## TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT
- Ngày: 2026-10-07
- Kết quả: **PASS**
- Phán quyết: **VALID_MECHANISM_GAP / NOVELTY_UNCLAIMED**
- Kết luận khoa học: không được tuyên bố mô-đun + định tuyến + lặp là mới; vẫn còn một khoảng trống cơ chế thực nghiệm đủ hẹp để kiểm tra định tuyến thích nghi trên nhiệm vụ có cấu trúc phụ thuộc thay đổi, dưới ngân sách và thước đo được khóa trước.
- Giới hạn: PASS này không chứng minh tính mới học thuật tổng quát, không chứng minh K8 tối ưu và không chứng minh lợi ích trên mô hình ngôn ngữ lớn (LLM).
- Tài liệu: `docs/TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT.md`

---

## Bằng chứng lịch sử nhập từ nhánh from-grok

Nguồn cố định: `from-grok@ccd0f2a342c180fd7b208ddd2e4d8eb7ee6c5ed9`

Phân loại bằng chứng: **phát triển/thăm dò (development/exploratory)**, không phải xác nhận (confirmatory), vì không có chuỗi cam kết độc lập chứng minh đăng ký trước đã được đóng băng trước từng lần chạy.

### Run 20261006_144648
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: A/B/C/D gần nhau; bộ định tuyến gần như không học được đường.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_144921
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: tăng năng lực mô hình không tạo lợi thế rõ cho định tuyến K8; khôi phục đường vẫn thấp.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_150521
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: định tuyến hard top-K không vượt đường cơ sở; F1 cạnh thấp.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_151416
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: đường cơ sở đạt R² cao nhưng định tuyến thích nghi vẫn không vượt; F1 cạnh thấp.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_154757
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: tôpô phân cấp khác nhau theo lớp tốt hơn nhẹ so với cùng tôpô, nhưng các biến thể có cổng (gated) không vượt residual phân cấp.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_155655
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: phạt entropy làm bộ định tuyến tập trung hơn nhưng không tạo lợi ích nhiệm vụ.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_160550
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: quỹ đạo tuần tự hard gần đường cơ sở residual nhưng chưa vượt; khôi phục quỹ đạo thấp.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_161628
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: xuất hiện khoảng cách ngoài phân phối (OOD) rõ; định tuyến tuần tự vẫn kém residual.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_162524
- Kết quả gốc: **FAIL**
- Kết luận quan sát: chương trình học + mất mát phụ theo hiện diện nút (curriculum + node-presence auxiliary loss) làm xấu mạnh cả ID và ngoài phân phối (OOD), dù thước đo quỹ đạo tăng nhẹ.
- Phạm vi chuẩn: FAIL thăm dò của biến thể này; không phải FAIL của giả thuyết TNR tổng quát.

### Run 20261006_164230
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: bộ nhớ khóa–giá trị cộng (additive KV) cho tín hiệu ngoài phân phối tốt hơn một số biến thể tuần tự trước.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_165236
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: additive KV tốt hơn residual trên ngoài phân phối ở 3/3 seed nhưng chưa đạt ngưỡng PASS đã nêu; tham số vẫn lệch.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_165820
- Kết quả chuẩn khi nhập: **UNRESOLVED**
- Kết luận quan sát: causal KV tốt hơn additive KV trên hai seed về MSE ngoài phân phối và khôi phục quỹ đạo.
- Lý do UNRESOLVED: so sánh thăm dò 2 seed, không có phán quyết terminal độc lập được khóa trước.
- Phạm vi chuẩn: chỉ dùng để chọn ứng viên phát triển.

### Run 20261007_kv_causal_confirm_partial
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: causal KV thắng residual 4/4 seed đã hoàn thành, nhưng chỉ có 4/5 seed, khôi phục quỹ đạo thấp và tỷ lệ tham số khoảng 1.38.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

### Run 20261006_173757
- Kết quả gốc: **UNRESOLVED**
- Kết luận quan sát: khi causal KV nhẹ hơn residual (tỷ lệ tham số khoảng 0.80), lợi thế ngoài phân phối thu hẹp còn khoảng 1.5% và thắng 4/5 seed.
- Phạm vi chuẩn: chỉ bằng chứng thăm dò.

---

## TNR_FROM_GROK_SCIENTIFIC_VALIDITY_RECONCILIATION
- Ngày: 2026-10-07
- Kết quả: **FAIL**
- Phán quyết: **CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY**
- Kết luận khoa học: các nhiệm vụ chính của `from-grok` chọn đường thật bằng biến trường hợp (case) ngẫu nhiên độc lập với đầu vào X, nhưng mô hình chỉ nhận X. Vì vậy đường cần đi không quan sát được từ thông tin mô hình có, nên chuỗi đó không thể phán quyết sạch giả thuyết “định tuyến thích nghi theo đầu vào”.
- Phát hiện bổ sung: thước đo khôi phục quỹ đạo ở nhánh đường dài so hai không gian chuỗi không hoàn toàn tương thích; một số mặt nạ tôpô chỉ hạn chế nút khả dụng chứ không thực thi chuyển tiếp cạnh.
- Hệ quả: giữ nguyên toàn bộ UNRESOLVED/FAIL lịch sử, nhưng không nâng bất kỳ kết quả nào thành bằng chứng xác nhận.
- Tài liệu: `docs/TNR_FROM_GROK_SCIENTIFIC_RECONCILIATION.md`

---

## Run 20261007_004313 — nhập từ from-grok
- Ngày: 2026-10-07
- Kết quả gốc: **PASS**
- Phân loại chuẩn: **PASS — DEVELOPMENT / EXPLORATORY ONLY**
- Kết luận quan sát: biến thể Causal–Full–Causal đạt OOD MSE trung bình 0.1760 so với HierResidual 0.1818 và vượt ngưỡng 0.97× trên 3 seed phát triển.
- Giới hạn: bộ sinh dữ liệu vẫn chọn path bằng case ngẫu nhiên độc lập với X; mô hình không nhận case, nên lỗi khả quan sát trung tâm vẫn tồn tại. Tham số Interleave cũng lớn hơn HierResidual đáng kể.
- Hệ quả: giữ kết quả PASS như bằng chứng phát triển, không nâng thành bằng chứng xác nhận và không thay đổi P1.
- Nguồn: `from-grok@43bec10e76ce20f0f408c96b314d1577cb96e501`

## TNR_FROM_GROK_SCIENTIFIC_VALIDITY_RECONCILIATION_ADDENDUM_20261007
- Ngày: 2026-10-07
- Kết quả: **FAIL**
- Phán quyết: **CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY_UNCHANGED**
- Kết luận khoa học: kết quả PASS mới của nhánh from-grok không sửa được lỗi khả quan sát đã xác định; do đó không thay đổi ranh giới giả thuyết hoặc tiêu chí P1.
- Tài liệu: `docs/TNR_FROM_GROK_RECONCILIATION_ADDENDUM_20261007.md`

---

## TNR_P1_VARIABLE_DEPENDENCY_ROUTING_CONFIRMATORY
- Ngày: 2026-10-08
- Kết quả: **FAIL**
- Phán quyết: **P1_VDR1_SPARSE_HARD_ROUTING_NOT_SUPPORTED**
- Tính hợp lệ: **PASS** — đúng 5 seed xác nhận đã khóa, không dùng seed cấm, route/split/parameter contract đã khóa trước TEST, D thực thi đúng 4 module calls/sample và TEST chỉ được materialize sau execution lock.
- Kết quả chính: `R_mean = 1.6035887437`; D thắng đường cơ sở cố định tốt nhất **0/5** world.
- Khôi phục route: độ chính xác vị trí trung bình `0.2507440476`, thấp hơn xa ngưỡng PASS `0.60`.
- Can thiệp cơ chế: `wrong/native = 2.2888673629` cho thấy lựa chọn route có ảnh hưởng mạnh đến đầu ra; nhưng `oracle/native = 1.6428756506` cho thấy ép theo route thật sau ánh xạ DEV làm xấu đáng kể, không hỗ trợ giả thuyết module/router đã học đúng ngữ nghĩa route nguyên tử.
- Điều kiện FAIL đã đăng ký trước cùng kích hoạt: `R_mean >= 1.05` và `world_wins <= 1/5`.
- Phạm vi: FAIL này chỉ áp dụng cho giả thuyết giới hạn P1-VDR1 về sparse hard routing dưới thiết kế/ngân sách đã khóa; không phủ định mọi biến thể TNR.
- Bằng chứng: `results/p1_vdr1/result.json`; QA độc lập: `results/p1_vdr1/independent_qa_and_adjudication.json`.
