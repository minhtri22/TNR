# Tiền đăng ký (preregistration) v1 — TNR, thực nghiệm tối thiểu đầu tiên

Khóa TRƯỚC khi chạy các seed đánh giá 0–7. Sau khi khóa, không được đổi định nghĩa dấu ">" hay ngưỡng.

## 1. Thiết lập cố định
- Nhiệm vụ: x ∈ R^8, mỗi ca có tuyến (route) là dãy 3–4 module; y = f_{r_L}(…f_{r_1}(x)), f_k(s)=tanh(Q_k s + b_k). Thế giới (Q_k, b_k) và dữ liệu đổi theo seed.
- Mô hình nhận x và bộ mô tả tuyến c (one-hot theo vị trí, 36 chiều). Không nhận trạng thái trung gian hay cạnh cần bật.
- Huấn luyện: 40 tuyến sinh cố định × 500 mẫu = 20 000 mẫu; val/test trong phân phối 100/200 mẫu mỗi tuyến; OOD: 5 tuyến mới (mọi cặp chuyển tiếp đã thấy, cả tuyến chưa thấy) × 400 mẫu.
- Optimizer Adam, lr 1e-2 (chọn trên seed hiệu chỉnh 1000), lịch cosine, 40 epoch, batch 200, không dừng sớm. Ngân sách tìm siêu tham số như nhau cho mọi mô hình: lưới lr {3e-3, 1e-2}, một seed.
- Seed đánh giá: 0,1,2,3,4,5,6,7 (thế giới, khởi tạo, dữ liệu cùng đổi theo seed).
- Mô hình (ngân sách tham số khớp D ± 5%): A = MLP 8 tầng tuyến tính (rộng 64); B = mạng dư, chiếu vào + 6 khối dư + chiếu ra (rộng 64); C = K8 tĩnh, mọi cạnh bật, T=4 (mh=49, nhận c ở nút đầu vào); D = K8 có cổng, T=4, K=2 cạnh/vòng, bộ định tuyến MLP nhận c và chỉ số vòng (mh=32), hệ số chi phí định tuyến λ=2e-4.
- Cổng cứng top-K (ngưỡng sigmoid>0.5) với gradient xuyên thẳng (straight-through). Module chỉ cập nhật khi có cạnh vào bật.
- Chi phí tính toán: proxy MAC theo cạnh/nút thực sự kích hoạt (đã gồm bộ định tuyến).
- Biến thể đối chứng: D_fixed (cổng không phụ thuộc đầu vào), D_ptr (bộ định tuyến con trỏ; mh khớp tham số), D_ring (đồ thị vòng thưa), D_rand50 (50% cạnh ngẫu nhiên), D_unshared (trọng số không chia sẻ giữa vòng lặp; nhiều tham số hơn), D_T1 (T=1, tắt lặp), D_shuffle (xáo trộn quan hệ mô tả–đích khi huấn luyện), D_oracle (cổng đúng tuyệt đối, chỉ để chẩn đoán).

## 2. Điều kiện hợp lệ (nếu vi phạm → CHƯA KẾT LUẬN)
- V1: tham số của A, B, C trong ±5% so với D.
- V2: baseline tốt nhất (A/B/C) có MSE trong phân phối trung bình < 0.5 × MSE dự đoán trung bình.
- V3: MAC trung bình của D ≤ 1.5 × MAC của A (ngưỡng ghi trước, dung sai cho chi phí bộ định tuyến).
- V4: D_oracle có MSE OOD trung bình < 0.5 × MSE OOD dự đoán trung bình (phép thử OOD có khả năng phân biệt).
- V5: ít nhất 7/8 seed hữu hạn (không NaN) cho mỗi mô hình.

## 3. Quy tắc quyết định chung (đối với chênh lệch tương đối r = (MSE_X − MSE_ref)/MSE_ref, ghép cặp theo seed)
Khoảng tin cậy 95% bootstrap phân vị của trung bình r (10 000 lần lấy mẫu lại theo seed, rng seed 0), biên δ = −10%.
- ĐẠT (PASS): cận trên < 0 và trung bình r ≤ −10%.
- KHÔNG ĐẠT (FAIL): cận dưới > −10% (đã đủ tin cậy rằng lợi ích nhỏ hơn 10% hoặc âm).
- CHƯA KẾT LUẬN (UNRESOLVED): còn lại, hoặc vi phạm điều kiện hợp lệ.
Baseline tham chiếu "tốt nhất" = trong A,B,C, mô hình có MSE trung bình thấp nhất trên chỉ số đang xét (xác định sau khi có dữ liệu nhưng theo quy tắc cố định này).

## 4. Giả thuyết
- H1 (chính): D so với baseline tốt nhất trên MSE OOD (tuyến chưa thấy), quy tắc §3, cần V1–V5.
- H2 (hiệu quả tính toán): D đạt MSE trong phân phối có cận trên 95% của r ≤ +10% so với baseline tốt nhất VÀ MAC(D) ≤ 0.5 × MAC(baseline đó) → ĐẠT; cận dưới của r > +10% hoặc MAC(D) > 0.5 × MAC → KHÔNG ĐẠT; còn lại CHƯA KẾT LUẬN.
- H3 (độ trung thực định tuyến): trên tuyến OOD, độ chính xác (precision) và độ phủ (recall) cạnh của D, sau căn chỉnh hoán vị module–phép toán ước lượng chỉ từ dữ liệu huấn luyện, đều ≥ 0.8 (trung bình seed) VÀ điều kiện chính xác của H2 (cận trên r ≤ +10%) → ĐẠT; nếu một trong precision/recall trung bình < 0.8 → KHÔNG ĐẠT.
- H4 (hiệu ứng định tuyến): D so với D_fixed trên MSE OOD, quy tắc §3 (V5 đủ).
- H5 (thăm dò đã khai báo): D_ptr so với baseline tốt nhất trên MSE OOD, quy tắc §3, cần V1–V5 (V1 áp cho D_ptr).
Ablation khác (D_ring, D_rand50, D_unshared, D_T1, D_shuffle, D_oracle) chỉ mô tả, kèm khoảng tin cậy, không có nhãn ĐẠT/KHÔNG ĐẠT; không được dùng để khẳng định cơ chế nhân quả. Tách hiệu ứng topology/định tuyến/lặp lại chỉ ở mức mô tả.

## 5. Giới hạn đã biết
Nhiệm vụ tổng hợp nhỏ; bộ mô tả tuyến được đưa vào mô hình (không suy luận tuyến từ x); bộ định tuyến chỉ phụ thuộc c và vòng lặp (không phụ thuộc trạng thái module); thế giới 8 module; 8 seed. Kết luận chỉ áp dụng trong phạm vi này.
