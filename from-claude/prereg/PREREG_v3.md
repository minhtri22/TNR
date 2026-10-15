# Tiền đăng ký (preregistration) v3 — TNR: kiểm định các "điểm gãy" rút ra từ tài liệu

Khóa TRƯỚC khi chạy bất kỳ seed 20–49 nào. Không sửa v1, v2. Kế thừa v1/v2: nhiệm vụ (40 tuyến huấn luyện, 5 tuyến OOD), K=2, T=4, λ=2e-4, Adam, batch 200, proxy MAC, căn chỉnh hoán vị module–phép toán ước lượng chỉ từ dữ liệu huấn luyện, bootstrap lấy mẫu lại theo seed (10 000 lần, rng seed 0), biên δ = −10%.

## 1. Cơ sở (tài liệu và dữ liệu của chúng tôi)
Các điểm gãy giả định (xem Lineage L023): (a) học đồng thời module và bố cục khó (Rosenbaum et al. 2019); (b) gradient cổng rời rạc bị lệch/thiếu, mô hình định tuyến học thường thua mô hình dày cùng tham số (Bengio et al. 2013 qua ST; Muqeeth et al. 2023, SMEAR; Liu et al. 2023); (c) bố cục học end-to-end thường sai trong khi tổng quát hóa nhạy với bố cục (Bahdanau et al. 2019); (d) lợi ích của module phụ thuộc chuyên biệt hóa và dễ bị nhiễu bởi dễ tối ưu (Mittal et al. 2022); (e) định tuyến theo nội dung bằng attention thưa cải thiện chuyên biệt hóa (Goyal et al. 2021, RIMs). Dữ liệu v2: D_oracle hai pha (4/12 seed hội tụ về ~0 và ghép được tuyến OOD; 8/12 kẹt ở MSE huấn luyện ~0,003 và OOD 0,2–0,3).

## 2. Thiết lập v3
- Epoch 100, lr 1e-2 cho mọi mô hình (hiệu chỉnh trên seed 1000–1003, xem L024: lr 3e-3 và 1e-3 đều không làm oracle hội tụ; không dò thêm).
- Mô hình (tham số khớp D_ref = 28 176 trong ±5%): A (MLP 8 tầng, rộng 64), B (mạng dư, rộng 64), C (K8 tĩnh, mh 49), E (MoE: 8 chuyên gia dùng chung qua T=4 vòng, top-2, cân bằng tải 0,01, mh 89, bộ định tuyến thấy bộ mô tả + chỉ số vòng + trạng thái), D_attn (K8, định tuyến attention theo nội dung: khóa phụ thuộc trạng thái module, cổng cứng top-K, ST, mh 55), D_sup (như D_attn + tổn thất giám sát bố cục γ=0,1 trên entropy chéo của cạnh thật; chặn trên "bố cục học được"), D_soft (như D_attn nhưng cổng mềm dày đặc, không top-K, gradient đầy đủ; tính MAC như đồ thị đầy đủ), D_oracle (module nối đúng bố cục thật = Neural Module Network với bố cục cho trước; chẩn đoán/sàng lọc).
- Không đưa vào v3 (giới hạn đã biết): RIMs đầy đủ, mở rộng số quy tắc lên >8 (đòi hỏi đổi hằng số M); ghi nhận là giới hạn.

## 3. Sàng lọc seed bằng oracle (đo hợp lệ theo từng seed)
Chạy D_oracle trên seed 20, 21, … theo thứ tự tăng dần đến khi có đủ 10 seed hợp lệ hoặc hết seed 49. Seed hợp lệ ⇔ loss huấn luyện epoch cuối < 1e-3 VÀ MSE OOD < 0,5 × mức dự đoán trung bình OOD. Phần xác nhận chỉ chạy trên 10 seed hợp lệ đầu tiên. Nếu < 8 seed hợp lệ trong seed 20–49 → mọi giả thuyết OOD (H1, H2, H3) là CHƯA KẾT LUẬN và chỉ báo cáo mô tả. Quy tắc này không phụ thuộc kết quả của các nhánh khác (chọn theo oracle trước khi chạy chúng). Hệ quả: kết luận chỉ áp dụng cho các "thế giới" mà mô hình chuẩn học được; ghi trong báo cáo.

## 4. Điều kiện hợp lệ (vi phạm → CHƯA KẾT LUẬN cho giả thuyết liên quan)
V1: tham số của A, B, C, E, D_attn, D_sup, D_soft trong ±5% so với D_ref. V2: baseline tốt nhất trong {A,B,C,E} theo MSE trong phân phối có MSE trung bình < 0,5 × mức dự đoán trung bình. V3: MAC của D_attn và D_sup ≤ 2,0 × MAC của A (D_soft không bị ràng buộc, báo cáo MAC dày đặc). V5: ít nhất n−1 seed hữu hạn cho mỗi mô hình.

## 5. Giả thuyết và quy tắc (r = (MSE_X − MSE_ref)/MSE_ref ghép cặp theo seed)
Quy tắc δ: ĐẠT nếu cận trên KTC < 0 và trung bình r ≤ −10%; KHÔNG ĐẠT nếu cận dưới KTC > −10%; còn lại CHƯA KẾT LUẬN. KTC 97,5% cho H1, H2 (hai kiểm định OOD chính, Bonferroni), 95% cho còn lại.
- H1: D_attn so với baseline tốt nhất trong {A,B,C,E} (MSE OOD trung bình thấp nhất), trên MSE OOD.
- H2: D_sup so với cùng baseline, trên MSE OOD (bố cục giám sát có giúp tổng quát hóa?).
- H3a: D_sup — độ chính xác (precision) và độ phủ (recall) cạnh trên tuyến OOD, sau căn chỉnh, trung bình ≥ 0,8 → ĐẠT, ngược lại KHÔNG ĐẠT. H3b: như vậy cho D_attn (không giám sát). Phân biệt "bố cục học được khi có giám sát" với "học được không giám sát".
- H4: D_soft so với D_attn trên MSE trong phân phối, quy tắc δ (giả thuyết gradient: gradient đầy đủ có giúp?).
- H5 (không thua mô hình dày): E so với A trên MSE trong phân phối: ĐẠT nếu cận trên KTC ≤ +10%; KHÔNG ĐẠT nếu cận dưới > +10%; còn lại CHƯA KẾT LUẬN.
- H6 (sụp đổ/chuyên biệt hóa), riêng cho D_attn và D_sup, trên dữ liệu trong phân phối: entropy sử dụng module chuẩn hóa ≥ 0,8 (không sụp đổ) VÀ độ phủ nút (node recall) sau căn chỉnh ≥ 0,8 → ĐẠT; ngược lại KHÔNG ĐẠT.
- H7 (hiệu quả tính toán): D_attn đạt cận trên KTC của r ≤ +10% so với baseline tốt nhất trong phân phối VÀ MAC ≤ 0,5 × MAC của baseline đó → ĐẠT; cận dưới > +10% hoặc MAC > 0,5× → KHÔNG ĐẠT; còn lại CHƯA KẾT LUẬN.
- Mô tả (không nhãn): D_oracle, D_soft OOD, hoán đổi bộ định tuyến, cắt bỏ module/vòng lặp.

## 6. Kỳ vọng ghi trước
D_attn không đạt H3b; D_sup có thể đạt H3a (nếu học được bố cục khi có giám sát); H1 nhiều khả năng KHÔNG ĐẠT hoặc CHƯA KẾT LUẬN; H5 có thể KHÔNG ĐẠT (MoE thua dày cùng tham số, theo tài liệu). Một kết quả khác cần được kiểm tra lại trước khi diễn giải.

## 7. Giới hạn
Nhiệm vụ tổng hợp nhỏ (8 phép toán); sàng lọc seed bằng oracle hạn chế phạm vi suy luận; một cấu hình siêu tham số cho mỗi nhánh; bộ định tuyến attention vẫn nhận bộ mô tả tuyến sẵn; tóm tắt tài liệu dựa trên tóm tắt/đoạn trích, chưa đọc toàn văn.
