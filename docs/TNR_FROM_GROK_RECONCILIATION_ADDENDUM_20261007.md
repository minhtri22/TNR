# TNR — Phụ lục đối chiếu khoa học nhánh from-grok

**Định danh:** TNR_FROM_GROK_SCIENTIFIC_VALIDITY_RECONCILIATION_ADDENDUM_20261007
**Ngày:** 2026-10-07
**Nguồn mới:** from-grok@43bec10e76ce20f0f408c96b314d1577cb96e501

## 1. Phần mới được rà soát

Sau lần đối chiếu trước, nhánh from-grok có thêm một cam kết chứa:

- biến thể Causal–Full–Causal;
- run 20261007_004313;
- kết quả gốc được ghi là PASS theo ngưỡng 0.97×;
- OOD MSE trung bình: HierResidual 0.1818, KV causal thuần 0.1791, Interleave 0.1760;
- 3 seed phát triển: 42, 7, 123.

## 2. Kiểm tra tính hợp lệ

Bộ sinh dữ liệu mới vẫn dùng cấu trúc:

- sinh X độc lập;
- chọn case ngẫu nhiên;
- case xác định path;
- mô hình chỉ nhận X.

Do đó lỗi khả quan sát đã xác định trước vẫn còn:

**P(path | X) = P(path)**

Bộ định tuyến không được cung cấp thông tin đủ để suy ra đúng path theo từng mẫu.

Ngoài ra mô hình Interleave có số tham số khoảng 44.428, trong khi HierResidual khoảng 32.728, nên so sánh này cũng chưa cân bằng tham số chặt.

## 3. Phán quyết

Kết quả gốc PASS được giữ nguyên như một kết quả nghiên cứu của nhánh from-grok nhưng được phân loại chuẩn là:

**PASS — DEVELOPMENT / EXPLORATORY ONLY**

Nó không phải bằng chứng xác nhận cho giả thuyết trung tâm của TNR.

Phán quyết kiểm toán trước:

**FAIL — CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY**

không thay đổi.

## 4. Hệ quả cho P1

P1 không được đổi task, ngưỡng, seed hoặc kiến trúc dựa trên run này.

Nguồn phát triển đã tiêu thụ được đóng băng tới:

**from-grok@43bec10e76ce20f0f408c96b314d1577cb96e501**

và được đưa vào khóa thực thi P1 như provenance phát triển đã biết.
