# Tiền đăng ký (preregistration) v4 — TNR: tách đầu đọc khỏi cổng, điều kiện "tốt hơn dự đoán hằng số", ước lượng gradient tốt hơn

Khóa TRƯỚC khi chạy seed 50–79 (tiến trình sàng lọc đã bị dừng và xóa trước khi có kết quả nào được xem). Không sửa v1–v3. Kế thừa v3: nhiệm vụ (40 tuyến huấn luyện, 5 tuyến OOD), T=4, Adam, lr 1e-2, 100 epoch, batch 200, tham số khớp D_ref = 28 176 trong ±5%, bootstrap theo seed (10 000 lần, rng seed 0), biên δ = −10%, sàng lọc seed bằng oracle.

## 1. Thay đổi so với v3
1. **Điều kiện C0 ("tốt hơn dự đoán hằng số")** áp dụng cho MỌI giả thuyết OOD: nhánh được xét phải có MSE OOD trung bình ≤ 0,5 × MSE OOD của dự đoán hằng số (trung bình trên seed). Nếu quy tắc δ cho ĐẠT nhưng C0 không thỏa → nhãn là KHÔNG ĐẠT. Tương tự C0-ID cho giả thuyết về MSE trong phân phối (≤ 0,5 × dự đoán hằng số trong phân phối).
2. **Tách đầu đọc và cổng trong D_sup.** Bốn nhánh: D_sup (cổng học có giám sát + đầu đọc học, như v3); D_sup_trueo (cổng học có giám sát + đầu đọc ĐÚNG, dùng cả khi huấn luyện lẫn đánh giá); D_trueG (cổng ĐÚNG + đầu đọc học); D_oracle (cả hai đúng; dùng để sàng lọc seed). Thêm can thiệp lúc đánh giá trên D_sup đã huấn luyện (thay cổng/đầu đọc bằng bản đúng; mô tả, không nhãn).
3. **Ước lượng gradient cho cổng không giám sát.** Định tuyến phân loại: mỗi vòng chọn một nguồn (9 mục: nút vào + 8 module) và một đích (9 mục: dừng + 8 module), một cạnh mỗi vòng. Bốn nhánh chỉ khác ước lượng gradient: D_cat_st (xuyên thẳng, ST), D_cat_gs (Gumbel-Softmax ST, tau=1), D_cat_rm (ReinMax, tau=1, theo mã gốc microsoft/ReinMax; đã kiểm tra trên bài toán đồ chơi: sai số gradient hàm bậc hai 0,005 so với 0,243 của ST), D_cat_rf (REINFORCE, hạng ưu thế = tổn thất theo mẫu − trung bình lô). Lấy mẫu khi huấn luyện, argmax khi đánh giá. Không có chi phí λ (một cạnh/vòng).
4. Sàng lọc seed 50, 51, … cho đến 8 seed hợp lệ hoặc hết seed 79; cần ≥ 6 seed hợp lệ, nếu không mọi giả thuyết CHƯA KẾT LUẬN. Seed hợp lệ: loss huấn luyện epoch cuối của D_oracle < 1e-3 VÀ MSE OOD < 0,5 × dự đoán hằng số OOD.
5. Chẩn đoán D0 (mô tả, không nhãn): vùng bằng phẳng của D_oracle thuộc về thế giới/dữ liệu hay khởi tạo? Chạy D_oracle trên 6 thế giới của v3 (seed 20, 21, 22 không hợp lệ; 25, 27, 29 hợp lệ) với 3 khởi tạo mới (100, 101, 102), cùng dữ liệu; báo cáo tỷ lệ hội tụ theo thế giới.

## 2. Nhánh chạy trên seed hợp lệ
A (MLP 8 tầng, rộng 64), D_sup, D_sup_trueo, D_trueG, D_cat_st, D_cat_gs, D_cat_rm, D_cat_rf (rộng 55 cho các nhánh D_*; D_oracle đã chạy lúc sàng lọc).

## 3. Điều kiện hợp lệ
V0: ≥ 6 seed hợp lệ. V1: tham số các nhánh trong ±5% so với D_ref. V2: MSE trong phân phối của A < 0,5 × dự đoán hằng số. V5: ít nhất n−1 seed hữu hạn mỗi nhánh. Vi phạm → CHƯA KẾT LUẬN.

## 4. Giả thuyết (r = (MSE_X − MSE_ref)/MSE_ref ghép cặp theo seed; quy tắc δ: ĐẠT nếu cận trên KTC < 0 và trung bình r ≤ −10%; KHÔNG ĐẠT nếu cận dưới KTC > −10%; còn lại CHƯA KẾT LUẬN; C0/C0-ID như §1)
- H1 (đầu đọc): D_sup_trueo so với D_sup trên MSE OOD; C0 áp cho D_sup_trueo; KTC 97,5%.
- H2 (cổng): D_trueG so với D_sup trên MSE OOD; C0 áp cho D_trueG; KTC 97,5%.
  Bảng diễn giải ghi trước: H1 ĐẠT & H2 không → đầu đọc là nguyên nhân; H2 ĐẠT & H1 không → cổng học được là nguyên nhân; cả hai ĐẠT → cả hai; không cái nào ĐẠT → nguyên nhân ở nơi khác (module/dữ liệu).
- H3 (ước lượng gradient): với est ∈ {gs, rm, rf}: D_cat_est so với D_cat_st trên MSE trong phân phối; C0-ID áp cho nhánh; KTC 98,33%.
- H4 (khôi phục cạnh không giám sát): với mỗi nhánh D_cat_*: precision và recall cạnh trên tuyến OOD, sau căn chỉnh hoán vị ước lượng từ dữ liệu huấn luyện, đều ≥ 0,8 (trung bình seed) → ĐẠT, ngược lại KHÔNG ĐẠT.
- H5 (ghép tuyến mới, không giám sát): mỗi nhánh D_cat_* so với A trên MSE OOD; C0 áp cho nhánh; KTC 98,75% (4 kiểm định).
- H6 (chặn trên có giám sát, lặp lại v3 với C0): D_sup so với A trên MSE OOD; C0 áp cho D_sup; KTC 95%.

## 5. Kỳ vọng ghi trước
H2 ĐẠT (cổng đúng + đầu đọc học khôi phục khả năng ghép) và H1 không ĐẠT, tức nguyên nhân là cổng học được; H3: ReinMax cải thiện nhẹ trong phân phối nhưng không nhánh nào thỏa C0-ID (→ KHÔNG ĐẠT); H4: không nhánh nào đạt; H5: không nhánh nào đạt; H6: KHÔNG ĐẠT. Kết quả pilot một seed (1000, thế giới không hợp lệ) chỉ dùng để kiểm tra triển khai: các nhánh D_cat_* đều có MSE trong phân phối 0,11–0,13 và recall ≤ 0,23; D_sup ở seed đó khôi phục cạnh chỉ 0,05, khác v3 → khôi phục có giám sát có thể phụ thuộc thế giới; D0 nhằm làm rõ.

## 6. Giới hạn
Sàng lọc oracle hạn chế phạm vi; vẫn 8 phép toán; bộ định tuyến nhận bộ mô tả tuyến sẵn; mỗi nhánh một cấu hình siêu tham số; D0 dùng dữ liệu v3 đã xem nên chỉ mang tính mô tả; kết quả D_oracle bị chọn lọc theo thiết kế.
