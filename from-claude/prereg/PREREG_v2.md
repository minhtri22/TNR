# Tiền đăng ký (preregistration) v2 — TNR

Khóa TRƯỚC khi chạy các seed 8–19. Kế thừa PREREG_v1 (không sửa v1); chỉ liệt kê khác biệt và các mục thay đổi. Mọi thứ không nêu ở đây giữ nguyên như v1 (nhiệm vụ, 40 tuyến huấn luyện, tuyến OOD, mô hình A/B/C/D, K=2, T=4, λ=2e-4, optimizer Adam, lr 1e-2, batch 200, proxy MAC, căn chỉnh hoán vị).

## 1. Thay đổi so với v1 (đều dựa trên thăm dò seed 1000, xem Lineage L011)
- Số epoch: 100 cho mọi mô hình (v1: 40). Lý do: với 100 epoch, mô hình định tuyến chuẩn (D_oracle) ghép được tuyến OOD trên seed 1000 (MSE OOD 0,0115 so với mức dự đoán trung bình 0,215), tức hy vọng thỏa V4.
- Seed đánh giá mới: 8–19 (12 seed, không trùng v1).
- Thêm hai nhánh huấn luyện cổng: "anneal" = cổng mềm dày đặc đến 30% huấn luyện, chuyển dần sang cổng cứng top-K đến 60%, nhiễu logistic có biên độ bằng hệ số trộn, λ tăng từ 0 lên 2e-4 trong giai đoạn chuyển, 40% cuối chỉ cổng cứng. Hai nhánh mới: D_anneal và D_ptr_anneal. Các nhánh D và D_ptr (cổng cứng từ đầu) giữ nguyên.
- Mô hình chạy: A, B, C, D, D_anneal, D_ptr, D_ptr_anneal, D_oracle, D_fixed. Bỏ D_ring, D_rand50, D_unshared, D_T1, D_shuffle khỏi v2 (đã có kết quả mô tả ở v1).
- V3 nới: MAC trung bình của nhánh được kiểm ≤ 2,0 × MAC của A (v1: 1,5), vì MAC thăm dò của D_anneal ≈ 1,6 × A. Ngưỡng này có thông tin từ thăm dò.
- Hiệu chỉnh đa kiểm định: các kiểm định OOD (H1, H1b, H5, H5b) dùng khoảng tin cậy bootstrap 98,75% (Bonferroni cho 4 kiểm định); dùng cận trên và cận dưới của khoảng này trong quy tắc §3 của v1.

## 2. Giả thuyết (quy tắc quyết định, biên δ = −10% như v1)
- H1: D so với baseline tốt nhất (A/B/C, xác định theo MSE OOD trung bình) trên MSE OOD. H1b: D_anneal so với cùng baseline. Cần V1–V5.
- H2: giống v1, kiểm cho D và cho D_anneal (hai nhãn riêng), so với baseline tốt nhất trong phân phối.
- H3: giống v1, kiểm riêng cho D, D_anneal, D_ptr, D_ptr_anneal (precision và recall OOD ≥ 0,8).
- H4: D so với D_fixed trên MSE OOD, khoảng tin cậy 98,75% không áp dụng (một kiểm định, dùng 95% như v1).
- H5: D_ptr so với baseline tốt nhất trên MSE OOD. H5b: D_ptr_anneal so với cùng baseline. Cần V1_ptr, V2, V4, V5.
- Điều kiện hợp lệ V1–V5 như v1 (V4: D_oracle MSE OOD trung bình < 0,5 × mức dự đoán trung bình OOD; V3 như §1).
Vi phạm điều kiện hợp lệ → CHƯA KẾT LUẬN. D_oracle chỉ để chẩn đoán.

## 3. Kỳ vọng ghi trước (để không điều chỉnh hồi tố)
Thăm dò một seed cho thấy cổng mềm→cứng cải thiện MSE trong phân phối của D (0,103 → 0,063) nhưng KHÔNG cải thiện độ phủ cạnh (recall 0,0) và làm tệ OOD (0,32 → 0,63); với D_ptr cả hai chỉ số đều tệ hơn. Do đó kỳ vọng thành thật: không nhánh nào đạt H3; H1/H1b/H5/H5b nhiều khả năng KHÔNG ĐẠT nếu V4 thỏa. Một kết quả khác sẽ là bất ngờ và cần kiểm tra lại trước khi diễn giải.

## 4. Giới hạn
Như v1; thêm: các siêu tham số của lịch anneal (30%/60%, nhiễu, λ ramp) chọn một lần, không dò; mỗi nhánh một cấu hình.
