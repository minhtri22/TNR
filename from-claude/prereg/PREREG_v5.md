# Tiền đăng ký (preregistration) v5 — TNR: khởi động lại nhiều lần, vai trò của đầu đọc, nhiễu gradient MSE vào cổng

Khóa TRƯỚC khi chạy các thế giới (seed) 80–84. Không sửa v1–v4. Kế thừa v4: nhiệm vụ (40 tuyến huấn luyện, 5 tuyến OOD, 8 phép toán), T=4, Adam, lr 1e-2, 100 epoch, batch 200, tham số khớp D_ref = 28 176 (±6%), bootstrap lấy mẫu lại theo thế giới (10 000 lần, rng seed 0), biên δ = −10%, điều kiện C0 (MSE OOD trung bình ≤ 0,5 × dự đoán hằng số) và C0-ID (tương tự cho trong phân phối).

## 1. Thay đổi so với v4
1. **Không sàng lọc seed bằng oracle.** Thay bằng khởi động lại nhiều lần: MỌI nhánh (kể cả A và D_oracle) được huấn luyện 3 lần với 3 khởi tạo khác nhau trên cùng thế giới/dữ liệu; lần được chọn là lần có loss huấn luyện (MSE nhiệm vụ trên tập huấn luyện, không gồm hạng phụ) nhỏ nhất — không bao giờ chọn theo OOD hay kiểm tra. Đơn vị phân tích là thế giới (5 thế giới, seed 80–84). Lý do: D0 ở v4 cho thấy D_oracle hội tụ ~50% mỗi lần khi đổi khởi tạo, nên sàng lọc theo seed chọn khởi tạo may mắn.
2. **Vai trò của đầu đọc học (kiểm định trực tiếp, cổng ĐÚNG để cô lập đầu đọc):** D_trueG (đầu đọc softmax học, như v4); D_trueG_oh (đầu đọc học, ép one-hot bằng xuyên thẳng: forward argmax, backward softmax); D_trueG_ent (đầu đọc softmax học + phạt entropy trọng số 0,05 cố định); D_trueG_id (đầu đọc softmax học nhưng KHỞI TẠO nhọn đúng, ma trận 6·I; chẩn đoán xem vấn đề là bể hấp dẫn của tối ưu hóa đầu đọc khi khởi tạo ngẫu nhiên); D_oracle (cổng đúng, đầu đọc đúng cố định).
3. **Nhiễu gradient MSE vào cổng (có giám sát bố cục, đầu đọc đúng để cô lập cổng):** D_sup_trueo (cổng học bằng giám sát + gradient MSE xuyên thẳng; như v4) so với D_sup_trueo_sg (cổng chỉ nhận gradient từ tổn thất giám sát; gradient MSE vào cổng bị dừng).
4. **Nhánh kết hợp:** D_sup_sg_oh (cổng học chỉ bằng giám sát, đầu đọc học ép one-hot).
5. A (MLP 8 tầng, rộng 64) làm baseline dày; các nhánh D_* rộng 55 (D_oracle rộng 32).

## 2. Điều kiện hợp lệ
V0: ≥ 4 thế giới hoàn tất. V1: tham số các nhánh (trừ D_oracle) trong ±6% so với D_ref. V4': D_oracle (tốt nhất trong 3 lần) hội tụ (loss huấn luyện < 1e-3 VÀ MSE OOD < 0,5 × hằng số) ở ≥ 4 thế giới (thay cho sàng lọc seed); nếu không thỏa, các giả thuyết OOD (H5, H6) là CHƯA KẾT LUẬN. Vi phạm V0/V1 → mọi giả thuyết CHƯA KẾT LUẬN.

## 3. Giả thuyết
Quy tắc δ: r = (MSE_X − MSE_ref)/MSE_ref ghép cặp theo thế giới; ĐẠT nếu cận trên KTC < 0 và trung bình r ≤ −10% (và C0/C0-ID thỏa); KHÔNG ĐẠT nếu cận dưới KTC > −10% hoặc quy tắc δ thỏa mà C0/C0-ID không thỏa; còn lại CHƯA KẾT LUẬN.
- H1, H2, H3 (đầu đọc; trong phân phối, KTC 98,33%): D_trueG_oh, D_trueG_ent, D_trueG_id so với D_trueG trên MSE trong phân phối; C0-ID áp cho nhánh.
  Bảng diễn giải ghi trước: H3 ĐẠT → thủ phạm là bể hấp dẫn khi khởi tạo đầu đọc (tối ưu hóa); H1 hoặc H2 ĐẠT → thủ phạm là hỗn hợp mềm của đầu đọc; không cái nào ĐẠT → đầu đọc học không phải thủ phạm duy nhất hoặc các cách ép đều khó huấn luyện.
- H4 (nhiễu MSE qua ST): hiệu độ phủ cạnh OOD (D_sup_trueo_sg − D_sup_trueo), ghép cặp theo thế giới, KTC 95%: ĐẠT nếu cận dưới > 0 và trung bình ≥ 0,2; KHÔNG ĐẠT nếu cận trên ≤ 0,05; còn lại CHƯA KẾT LUẬN.
- H5a, H5b, H5c (ghép tuyến mới với cổng đúng, KTC 98,33%): D_trueG_oh, D_trueG_ent, D_trueG_id so với A trên MSE OOD; C0 áp cho nhánh; cần V4'.
- H6 (kết hợp): D_sup_sg_oh so với A trên MSE OOD; C0; KTC 95%; cần V4'.
- H7: D_sup_sg_oh có precision và recall cạnh OOD (căn chỉnh hoán vị từ dữ liệu huấn luyện) đều ≥ 0,8 → ĐẠT, ngược lại KHÔNG ĐẠT.
- Mô tả (không nhãn): tỷ lệ lần chạy hội tụ theo nhánh (15 lần chạy mỗi nhánh), kiểm định Fisher chính xác so với D_trueG.

## 4. Kỳ vọng ghi trước (từ pilot một seed 1000, khởi tạo đơn, chỉ để kiểm tra triển khai)
Pilot (MSE huấn luyện / OOD; recall cạnh OOD): D_trueG_oh 0,062/0,461 (1,0); D_trueG_ent 0,119/467,99 (1,0); D_sup_trueo_sg 0,0043/0,563 (0,96); D_sup_sg_oh 0,144/0,232 (1,0). Kỳ vọng: H4 ĐẠT (dừng gradient MSE giúp cổng học được: recall 0,96 so với 0,58 ở v4); H1 và H2 không ĐẠT (ép one-hot hoặc phạt entropy làm tối ưu hóa khó hơn rõ rệt ở pilot); H3: không có kỳ vọng mạnh (đây là chẩn đoán chính); H6 không ĐẠT. Kết quả trái kỳ vọng cần được kiểm tra lại trước khi diễn giải.

## 5. Giới hạn
5 thế giới × 3 khởi tạo (mẫu nhỏ); chọn lần theo loss huấn luyện; vẫn 8 phép toán; bộ mô tả tuyến được đưa sẵn; một cấu hình siêu tham số mỗi nhánh (β=0,05 và hệ số khởi tạo 6 chọn một lần, không dò); phạt entropy và one-hot xuyên thẳng là hai cách ép cụ thể, không đại diện cho mọi cách; D_oracle vẫn là nhánh đặc biệt (cổng và đầu đọc đúng cố định).
