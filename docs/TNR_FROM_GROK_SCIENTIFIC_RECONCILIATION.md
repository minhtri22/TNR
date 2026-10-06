# TNR — Đối chiếu khoa học nhánh from-grok

**Định danh:** `TNR_FROM_GROK_SCIENTIFIC_VALIDITY_RECONCILIATION`  
**Ngày:** 2026-10-07  
**Nguồn bằng chứng:** nhánh `from-grok`, HEAD `ccd0f2a342c180fd7b208ddd2e4d8eb7ee6c5ed9`

## 1. Mục đích

Tài liệu này đối chiếu phần nghiên cứu đã được thực hiện trên nhánh `from-grok` trong lúc dòng `main` chưa tiếp tục.

Mục tiêu là:

- bảo toàn toàn bộ kết quả khoa học đã chạy;
- không xóa hoặc viết lại FAIL / UNRESOLVED;
- phân loại đúng giá trị bằng chứng của các lần chạy;
- phát hiện các lỗi thiết kế khoa học có thể làm sai diễn giải;
- quyết định phần nào được phép dùng để thiết kế đăng ký trước (preregistration) tiếp theo.

Tài liệu này **không merge lịch sử Git** của `from-grok` vào `main`.

## 2. Trạng thái nguồn

Nhánh `from-grok` không có tổ tiên chung (common ancestor) với `main`.

Do đó nó được xem là một dòng bằng chứng độc lập và được hội tụ có chọn lọc (selective convergence), không merge/rebase trực tiếp.

Chuỗi nhánh chứa:

- thí nghiệm A/B/C/D ban đầu;
- các lần chạy v2/v3;
- tôpô phân cấp (hierarchical topology);
- định tuyến tuần tự (sequential routing);
- kiểm tra ngoài phân phối (OOD);
- chương trình học + mất mát phụ (curriculum + auxiliary loss);
- bộ nhớ khóa–giá trị trượt (sliding KV);
- bộ nhớ khóa–giá trị nhân quả (causal KV);
- nhiều seed và thử cân bằng tham số.

## 3. Phân loại đăng ký trước

Cam kết đầu tiên của nhánh `from-grok` đã chứa đồng thời ba lần chạy đầu, mã, báo cáo và kết quả.

Các lần chạy sau chủ yếu được cam kết sau khi đã có kết quả.

Vì vậy không có bằng chứng Git độc lập đủ mạnh để chứng minh rằng từng đăng ký trước (preregistration) đã được đóng băng ở một cam kết riêng trước khi thực thi.

Phân loại chuẩn:

> Toàn bộ `from-grok` được dùng như **bằng chứng phát triển/thăm dò (development/exploratory evidence)**, không phải bằng chứng xác nhận (confirmatory evidence).

Điều này không phủ nhận dữ liệu đã chạy. Nó chỉ giới hạn mức tuyên bố khoa học được phép rút ra.

## 4. Lỗi khả quan sát của nhiệm vụ

Đây là lỗi khoa học quan trọng nhất.

Trong các bộ sinh dữ liệu chính, ví dụ:

- `code/tnr_minimal_experiment.py`
- `code/tnr_experiment_v3.py`
- `code/tnr_hierarchical_A.py`
- `code/tnr_seq_ood_longdep.py`
- `code/tnr_kv_causal_parity5.py`

dữ liệu được sinh theo dạng khái quát:

~~~python
X = random_payload()
cases = random_case_independent_of_X()
path = CASE_PATHS[cases]
Y = apply_path(X, path)
~~~

nhưng mô hình chỉ nhận:

~~~python
model(X)
~~~

Biến `cases` hoặc một tín hiệu tương đương về đường cần đi **không được đưa cho mô hình**.

Do đó, theo thiết kế sinh dữ liệu:

**P(path | X) = P(path)**

Nói cách khác, đường đúng là một biến ẩn ngẫu nhiên độc lập với đầu vào mà bộ định tuyến nhìn thấy.

### Hệ quả

Một bộ định tuyến không thể suy ra chính xác đường phụ thuộc theo từng mẫu từ thông tin không tồn tại trong đầu vào.

Vì vậy các giá trị khôi phục đường thấp không chứng minh rằng định tuyến thích nghi (adaptive routing) thất bại; ngược lại, nhiệm vụ hiện tại không cung cấp đủ thông tin để phép thử đó có thể phân biệt giả thuyết trung tâm.

Các mô hình mạnh vẫn có thể đạt MSE/R² tốt bằng cách học trung bình có điều kiện hoặc xấp xỉ hỗn hợp các phép biến đổi, nhưng đó không phải bằng chứng về khôi phục đường.

## 5. Vấn đề của thước đo quỹ đạo

Ở các lần chạy tuần tự dài:

- dự đoán có dạng 3 lớp × 5 bước = 15 lựa chọn mô-đun;
- đường thật chỉ dài khoảng 3–6;
- chỉ số dự đoán được ánh xạ theo `layer * 4 + local_module`;
- đường thật chủ yếu dùng chỉ số 0–7.

Do đó thước đo chuỗi con chung dài nhất (LCS) đang so hai không gian chuỗi không hoàn toàn tương đương.

Điều này làm cho giá trị khôi phục quỹ đạo (trajectory recovery) không phải một thước đo cơ chế sạch.

P1 không được tái sử dụng trực tiếp thước đo này.

## 6. Vấn đề của tôpô trong biến thể causal KV

Trong `tnr_kv_causal_parity5.py`, mặt nạ tôpô (topology mask) dùng:

~~~python
mask_topo = (self.topo_masks[li].sum(0) > 0)
~~~

Cách này chỉ kiểm tra một nút có ít nhất một cạnh vào hay không.

Nó **không khóa chuyển tiếp theo cạnh hiện tại → nút kế tiếp**.

Nếu mọi nút đều có ít nhất một cạnh vào, bộ điều khiển vẫn có thể chọn mọi nút tại mỗi bước.

Vì vậy các nhãn “sparse / ring / complete” trong biến thể này không còn là phép thử sạch về đường đi trên graph.

## 7. Kiểm tra số học trong báo cáo KV causal

Báo cáo củng cố 4 seed ghi điều kiện:

**mean_OOD_KV < 0.97 × mean_OOD_HR**

với số liệu gần:

**0.1770 / 0.1840 ≈ 0.962**

Tỷ lệ 0.962 thực tế **nhỏ hơn** 0.97, không phải lớn hơn.

Tuy nhiên kết quả đó vẫn không thể thành PASS vì đồng thời:

- khôi phục quỹ đạo không đạt ngưỡng;
- chỉ hoàn tất 4/5 seed;
- tỷ lệ tham số khoảng 1.38, không đạt cân bằng chặt.

Do đó lỗi số học này không thay đổi phán quyết tổng thể nhưng phải được ghi nhận để tránh lặp lại.

## 8. Giá trị khoa học còn giữ được

Dù không hợp lệ để phán quyết giả thuyết trung tâm, nhánh `from-grok` vẫn cung cấp bằng chứng phát triển có giá trị:

1. Định tuyến K8 mềm/đặc không tự động thắng đường cơ sở (baseline).
2. Ép mất mát phụ (auxiliary loss) sai mục tiêu có thể làm giảm mạnh hiệu năng.
3. Mạng residual phân cấp là đường cơ sở mạnh cần giữ.
4. Bộ nhớ khóa–giá trị nhân quả (causal KV) tạo tín hiệu ngoài phân phối (OOD) nhất quán hơn một số biến thể trước, nhưng chưa được xem là bằng chứng xác nhận.
5. Cân bằng tham số làm lợi thế của causal KV mỏng đi.
6. Thiết kế nhiệm vụ phải bảo đảm đường phụ thuộc **có thể suy ra từ thông tin mà mô hình nhận được**.

## 9. Phán quyết

**FAIL — CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY**

Ý nghĩa của FAIL này:

> Chuỗi `from-grok` không phải là thất bại của giả thuyết TNR. Nó là thất bại của **tính hợp lệ của phép thử trung tâm**, do đường phụ thuộc không quan sát được từ đầu vào và thước đo quỹ đạo chưa tương thích đầy đủ.

Tất cả kết quả cũ được giữ nguyên như bằng chứng thăm dò.

Không lần chạy nào được nâng thành PASS xác nhận.

## 10. Hệ quả cho P1

P1 phải sửa ba điểm bắt buộc:

1. đường phụ thuộc phải quan sát được từ đầu vào qua một ngữ cảnh định tuyến (routing context);
2. tập kiểm tra phải giữ lại **tổ hợp đường mới**, không giữ lại thông tin cần thiết để xác định đường;
3. thước đo khôi phục đường phải dùng cùng không gian mô-đun với đường thật và xử lý hoán vị nhãn mô-đun nếu cần.

Bước được phép tiếp theo vẫn là:

**`TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION`**

Giải thích ngắn bằng Tiếng Việt:

Đăng ký trước phép thử xác nhận đầu tiên của dòng chuẩn, dùng bề mặt mới chưa bị tiêu thụ bởi `from-grok`, khóa khả quan sát của đường, ngân sách, áp lực định tuyến, thước đo và tiêu chí PASS / FAIL / UNRESOLVED trước khi viết phần triển khai (implementation).
