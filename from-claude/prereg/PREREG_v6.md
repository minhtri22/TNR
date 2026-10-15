# Tiền đăng ký (preregistration) v6 — TNR: học đầu đọc từ khởi tạo ngẫu nhiên, ghép đầu-cuối, giảm giám sát cổng

Khóa TRƯỚC khi chạy các thế giới (seed) 90–95. Lượt chính thức v6 sẽ chạy trên Google Colab (nhiều phiên song song theo thế giới) chứ không chạy trong phiên của trợ lý. Không sửa v1–v5. Kế thừa v5: nhiệm vụ (40 tuyến huấn luyện, 5 tuyến OOD, 8 phép toán), T=4, Adam, lr 1e-2, 100 epoch, batch 200, tham số khớp D_ref = 28 176 (±6%), khởi động lại 3 lần cho mọi nhánh, lần được chọn là lần có loss huấn luyện nhỏ nhất (không theo OOD), bootstrap lấy mẫu lại theo thế giới (10 000 lần, rng seed 0), biên δ = −10%, điều kiện C0 và C0-ID (≤ 0,5 × dự đoán hằng số).

## 1. Cơ sở từ v5
D_trueG (đầu đọc softmax khởi tạo ngẫu nhiên, cổng đúng) hội tụ 0/15 lần; D_trueG_id (chỉ khác: đầu đọc khởi tạo nhọn đúng) hội tụ 5/15, ngang D_oracle, và ghép được tuyến mới (MSE OOD 0,062). Ép one-hot xuyên thẳng và phạt entropy từ khởi tạo ngẫu nhiên đều thất bại. Dừng gradient MSE vào cổng giúp cổng học có giám sát đạt recall ~1,0 nhưng đầu đọc học không hội tụ nên chưa ghép được tuyến mới. v6 hỏi: (1) có cách học đầu đọc từ khởi tạo ngẫu nhiên không (không đưa đáp án vào khởi tạo)? (2) cổng giám sát (dừng gradient MSE) + đầu đọc học đầu-cuối ghép được tuyến mới không? (3) cần bao nhiêu giám sát cổng?

## 2. Các nhánh (tham số khớp ±6%; D_* rộng 55, D_oracle rộng 32, A rộng 64)
Tham chiếu: A (MLP 8 tầng); D_oracle (cổng + đầu đọc đúng cố định); D_trueG (cổng đúng, đầu đọc softmax khởi tạo ngẫu nhiên); D_trueG_id (cổng đúng, đầu đọc khởi tạo nhọn đúng; điều khiển dương, có thông tin đáp án).
**Học đầu đọc từ khởi tạo không chứa đáp án (cổng đúng để cô lập):**
- D_trueG_bal: khởi tạo cân bằng (ma trận đầu đọc toàn 0 → softmax đồng đều ban đầu).
- D_trueG_tau: nhiệt độ giảm dần: softmax(logit/τ), τ giảm tuyến tính 1,0 → 0,1 trong 70% đầu huấn luyện rồi giữ 0,1; đánh giá bằng argmax cứng.
- D_trueG_rm: ReinMax (tau = 1, theo mã gốc microsoft/ReinMax; đã kiểm tra ở v4) cho đầu đọc phân loại: lấy mẫu một module khi huấn luyện, argmax khi đánh giá.
**Ghép đầu-cuối (cổng học bằng giám sát, gradient MSE vào cổng bị dừng, đầu đọc học):** D_e2e_bal, D_e2e_tau, D_e2e_rm (ba cách học đầu đọc ở trên) và D_e2e_id (đầu đọc khởi tạo nhọn đúng; điều khiển dương).
**Giảm giám sát cổng (đầu đọc khởi tạo nhọn đúng để cô lập):** D_e2e_id_g50, D_e2e_id_g25, D_e2e_id_g10: chỉ 50%, 25%, 10% số TUYẾN huấn luyện (20, 10, 4 trong 40) có nhãn cổng (chọn ngẫu nhiên cố định theo seed); tuyến còn lại chỉ được cổng suy ra từ chia sẻ tham số của bộ định tuyến (không có nhãn, không có gradient MSE vào cổng).

## 3. Điều kiện hợp lệ
V0: ≥ 4 thế giới hoàn tất. V1: tham số các nhánh (trừ D_oracle) trong ±6% so với D_ref. V4': D_oracle (tốt nhất trong 3 lần) hội tụ (loss huấn luyện < 1e-3 VÀ MSE OOD < 0,5 × hằng số) ở ≥ 2/3 số thế giới hoàn tất; nếu không thỏa, các giả thuyết OOD (H2, H4) là CHƯA KẾT LUẬN. Vi phạm V0/V1 → mọi giả thuyết CHƯA KẾT LUẬN.

## 4. Giả thuyết (quy tắc δ: r = (MSE_X − MSE_ref)/MSE_ref ghép cặp theo thế giới; ĐẠT nếu cận trên KTC < 0 và trung bình r ≤ −10% và C0/C0-ID thỏa; KHÔNG ĐẠT nếu cận dưới KTC > −10% hoặc quy tắc δ thỏa mà C0/C0-ID không thỏa; còn lại CHƯA KẾT LUẬN)
- H1a, H1b, H1c (học đầu đọc từ khởi tạo không chứa đáp án; MSE trong phân phối; KTC 98,33%): D_trueG_bal, D_trueG_tau, D_trueG_rm so với D_trueG; C0-ID áp cho nhánh. Diễn giải ghi trước: bất kỳ cái nào ĐẠT → có cách học đầu đọc từ đầu; không cái nào → chỉ khởi tạo chứa thông tin mới cứu được đầu đọc.
- H2a, H2b, H2c, H2d (ghép tuyến mới đầu-cuối; MSE OOD so với A; KTC 98,75%; cần V4'): D_e2e_bal, D_e2e_tau, D_e2e_rm, D_e2e_id; C0 áp cho nhánh.
- H3 (bố cục học được ở nhánh đầu-cuối; riêng cho 4 nhánh H2): precision và recall cạnh OOD (căn chỉnh hoán vị từ dữ liệu huấn luyện) đều ≥ 0,8 → ĐẠT, ngược lại KHÔNG ĐẠT.
- H4a, H4b, H4c (giảm giám sát; không thua kém; KTC 98,33%; cần V4'): D_e2e_id_g50/g25/g10 so với D_e2e_id trên MSE OOD: ĐẠT nếu cận trên KTC của r ≤ +10% và C0 thỏa; KHÔNG ĐẠT nếu cận dưới KTC > +10% hoặc cận trên ≤ +10% nhưng không thỏa C0; còn lại CHƯA KẾT LUẬN. Mô tả kèm: recall cạnh OOD.
- Mô tả (không nhãn): tỷ lệ lần chạy hội tụ mỗi nhánh (18 lần chạy nếu 6 thế giới) và kiểm định Fisher chính xác so với D_trueG; checkpoint (tham số cuối) của mọi lần chạy được lưu.

## 5. Kỳ vọng ghi trước (chưa có pilot nào cho các cách học đầu đọc mới ngoài kiểm tra chạy 1 epoch)
Không có kỳ vọng mạnh. Kỳ vọng thấp cho khởi tạo cân bằng (D_trueG_bal: đối xứng khó phá, dễ kẹt như khởi tạo ngẫu nhiên); trung bình cho nhiệt độ giảm dần và ReinMax (đã cho ước lượng gradient bậc hai tốt ở v4 nhưng không giúp định tuyến không giám sát). D_e2e_id (điều khiển dương) kỳ vọng ĐẠT H2d và H3 (dựa trên v5: H5c ĐẠT, H7 ĐẠT). H4: kỳ vọng g50 không thua D_e2e_id; g25, g10 không chắc.

## 6. Giới hạn
Tối đa 6 thế giới × 3 khởi tạo; chọn lần theo loss huấn luyện; D_trueG_id/D_e2e_id có thông tin đáp án trong khởi tạo (điều khiển dương, không phải học từ đầu); τ-lịch (1,0→0,1 trong 70%), β, tau của ReinMax và hệ số khởi tạo được chọn một lần, không dò; vẫn 8 phép toán; bộ mô tả tuyến đưa sẵn; tính lặp lại theo Colab phụ thuộc phiên bản JAX/CPU/GPU (ghi trong meta của từng phiên); giám sát cổng giảm theo tuyến, không theo mẫu.
