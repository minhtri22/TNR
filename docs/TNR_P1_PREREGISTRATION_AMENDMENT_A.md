# TNR P1 — Phụ lục đăng ký trước A

**Định danh:** TNR_P1_PREREGISTRATION_AMENDMENT_A
**Ngày:** 2026-10-07
**Trạng thái:** khóa trước huấn luyện P1 và trước materialize TEST.

## Mục đích

Tài liệu này khóa các chi tiết triển khai có ảnh hưởng tới kết quả nhưng chưa được nêu đủ cụ thể trong đăng ký trước P1 ban đầu.

Không có kết quả TRAIN, DEV hoặc TEST nào được dùng để chọn các giá trị dưới đây.

## Các chi tiết được khóa

1. Thiết bị xác nhận: **CPU**.
2. PyTorch bật thuật toán xác định (deterministic algorithms) khi runner xác nhận chạy.
3. Adam:
   - learning rate = 0.001;
   - betas = (0.9, 0.999);
   - eps = 1e-8;
   - weight decay = 0.
4. Bộ định tuyến D:
   - Gumbel-softmax straight-through;
   - hard top-1;
   - temperature = **1.0 cố định**, không anneal.
5. Bộ định tuyến C:
   - softmax temperature = 1.0.
6. Early stopping:
   - đánh giá DEV MSE sau mỗi epoch;
   - một epoch được xem là cải thiện khi DEV MSE thấp hơn best trước ít nhất 1e-12;
   - dừng sau 10 epoch liên tiếp không cải thiện;
   - tối đa 60 epoch;
   - khôi phục state có DEV MSE tốt nhất.
7. Thứ tự minibatch:
   - dùng generator PyTorch riêng cho từng world;
   - seed minibatch = model_init_seed + 1;
   - cùng thứ tự index cơ sở được tạo độc lập cho từng mô hình nhưng từ cùng seed.
8. NMSE:
   - Var(Y_TEST) là phương sai population (unbiased=False);
   - tính trên toàn bộ phần tử của tensor Y_TEST sau khi flatten 4 chiều output.
9. Chia dữ liệu:
   - TRAIN/DEV được phép materialize trong tiền kiểm tra;
   - TEST chỉ được materialize sau khi runner xác minh execution lock.
10. Không có tuning siêu tham số (hyperparameter tuning) trên TRAIN/DEV trong P1 xác nhận; các giá trị trên là cố định.

## Quan hệ với đăng ký trước gốc

Phụ lục này không thay đổi:

- câu hỏi khoa học;
- route split;
- world seed;
- mô hình A/B/C/D;
- số tham số;
- ngưỡng PASS / FAIL / UNRESOLVED;
- thước đo route;
- can thiệp cơ chế.

Nó chỉ đóng các bậc tự do triển khai còn thiếu trước execution lock.
