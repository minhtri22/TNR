# Báo cáo khoa học: Định tuyến nơ-ron theo topology (Topological Neural Routing, TNR) — thực nghiệm tối thiểu đầu tiên

## 1. Tóm tắt
Chúng tôi kiểm chứng một giả thuyết hẹp: trên nhiệm vụ có đồ thị phụ thuộc thay đổi theo ca, mạng K8 có cổng định tuyến (D) có lợi so với tính toán độ sâu cố định (A: MLP 8 tầng, B: mạng dư) và K8 tĩnh (C) dưới ngân sách tham số khớp nhau (~28 nghìn) hay không. Kết quả trên 8 hạt giống ngẫu nhiên (seed): **không có bằng chứng ủng hộ**. D kém hơn rõ trong phân phối (MSE 0,069 so với 0,009–0,011), bộ định tuyến gần như không khôi phục được tuyến thật (độ phủ cạnh 0,02), và phép thử tuyến chưa thấy (OOD) không đủ khả năng phân biệt vì ngay cả định tuyến chuẩn (oracle) cũng không ghép được tuyến mới trong cấu hình đã khóa. Kết luận: H1 CHƯA KẾT LUẬN, H2 KHÔNG ĐẠT, H3 KHÔNG ĐẠT, H4 CHƯA KẾT LUẬN, H5 CHƯA KẾT LUẬN. Không có giả thuyết nào ĐẠT. Kết quả chỉ áp dụng cho phạm vi nhiệm vụ tổng hợp nhỏ này.

## 2. Phương pháp
**Nhiệm vụ.** Mỗi ca có vector x ∈ R^8 và một tuyến là dãy 3–4 module từ 8 phép toán; đích y = f_{r_L}(…f_{r_1}(x)) với f_k(s) = tanh(Q_k s + b_k). Mô hình nhận x và bộ mô tả tuyến (one-hot theo vị trí, 36 chiều), không nhận trạng thái trung gian hay cạnh cần bật. Huấn luyện: 40 tuyến sinh cố định × 500 mẫu. Kiểm tra OOD: 5 tuyến mới mà mọi cặp chuyển tiếp đã thấy nhưng cả tuyến chưa thấy.

**Mô hình** (tham số khớp ±5%, D = 28 176): A (MLP 8 tầng), B (mạng dư), C (K8 tĩnh, mọi cạnh bật, nhận bộ mô tả ở nút đầu vào), D (K8 có cổng cứng top-K, K=2 cạnh/vòng, T=4 vòng, chi phí định tuyến λ=2·10⁻⁴). Cạnh tham số hóa qua "bus" chung; module không thấy bộ mô tả. Các biến thể đối chứng: D_fixed, D_ptr, D_ring, D_rand50, D_unshared, D_T1, D_shuffle, D_oracle.

**Quy trình.** Tiền đăng ký `prereg/PREREG_v1.md` (SHA-256 ghi trong Lineage) khóa giả thuyết, điều kiện hợp lệ và quy tắc quyết định: chênh lệch tương đối ghép cặp theo seed, khoảng tin cậy 95% lấy mẫu lại (bootstrap), biên −10%. Seed đánh giá 0–7; seed 1000 chỉ dùng thăm dò.

**Trung thực về thăm dò.** Trước khi khóa, thiết kế được chỉnh dựa trên seed 1000 (xem L001): bỏ bộ mô tả khỏi module, đổi sang bus chung, giảm độ khó nhiệm vụ, tăng số tuyến huấn luyện lên 40, chọn lr. Ngưỡng V3 đặt sau khi biết MAC thăm dò. Một lần chạy nền bị ngắt và khởi động lại ở seed 4 (L003, L010).

## 3. Kết quả
Trung bình ± độ lệch chuẩn trên 8 seed; mức dự đoán trung bình: 0,197 (OOD).

| Mô hình | Tham số | MSE trong phân phối | MSE OOD | MAC/mẫu |
|---|---|---|---|---|
| A | 28 360 | 0,0088 ± 0,0004 | 0,351 ± 0,029 | 28 360 |
| B | 28 360 | 0,0101 ± 0,0004 | 0,319 ± 0,042 | 28 360 |
| C | 28 112 | 0,0114 ± 0,0004 | 0,426 ± 0,072 | 145 344 |
| D | 28 176 | 0,0694 ± 0,0208 | 0,460 ± 0,312 | 38 933 |
| D_ptr | 28 010 | 0,0485 ± 0,0085 | 0,320 ± 0,104 | 16 104 |
| D_oracle (chẩn đoán) | 20 872 | 0,0051 ± 0,0007 | 0,259 ± 0,046 | 10 195 |
| D_fixed | 28 176 | 0,1264 ± 0,0073 | 0,323 ± 0,072 | 34 352 |
| D_shuffle (đối chứng) | 28 176 | 0,2076 ± 0,0124 | 0,196 ± 0,010 | 32 512 |

Hình: `figures/mse_theo_mo_hinh.png`, `figures/mac_va_ood.png`, `figures/duong_cong_huan_luyen.png`.

### 3.1 Kết luận theo giả thuyết
| Giả thuyết | Kết quả | Nhãn |
|---|---|---|
| H1: D so với B trên MSE OOD | r = +40,8% [−5,1%; +104,6%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H2: hiệu quả tính toán | MSE trong phân phối +690% so với A; MAC gấp 1,37 lần | KHÔNG ĐẠT (FAIL) |
| H3: độ trung thực định tuyến | precision 0,009; recall 0,019 (ngưỡng 0,8) | KHÔNG ĐẠT (FAIL) |
| H4: D so với D_fixed trên OOD | r = +61,0% [−12,5%; +168,6%] | CHƯA KẾT LUẬN (UNRESOLVED) |
| H5: D_ptr so với B trên OOD | r = +1,7% [−18,1%; +31,0%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |

### 3.2 Cắt bỏ và cơ chế (mô tả, không nhãn)
- Đối chứng xáo trộn (D_shuffle) không học được (MSE trong phân phối ≈ mức dự đoán trung bình), nên các mô hình khác thực sự dùng quan hệ giữa bộ mô tả và đích.
- Hoán đổi bộ mô tả của bộ định tuyến làm MSE trong phân phối của D tăng từ 0,069 lên 0,268: đầu ra phụ thuộc vào định tuyến được chọn, dù cạnh bật không trùng cạnh thật. Do đó định tuyến có tác dụng chức năng trong phân phối, nhưng không phải tuyến phụ thuộc thật.
- Trên OOD, cắt bỏ vòng lặp hay module không làm tệ thêm (độ lệch ≤ 0), tức không có tính toán có nghĩa nào bị phá vỡ: mô hình chưa tính toán được trên tuyến mới.
- Các biến thể thưa, ngẫu nhiên, không chia sẻ trọng số và T=1 khác D trên OOD với khoảng tin cậy rộng chứa 0 (xem L009); chưa thể tách hiệu ứng topology, định tuyến và lặp lại.

## 4. Thảo luận
1. **Bộ định tuyến không tự học được tuyến chỉ từ sai số nhiệm vụ.** Cổng cứng top-K với gradient xuyên thẳng cho độ phủ cạnh đúng ≈ 2% (D) và ≈ 18% (D_ptr), dù D_ptr có thiên kiến cấu trúc. Đây là kết quả âm về tối ưu hóa của chính cơ chế đã chọn, chưa phải kết luận về mọi cơ chế định tuyến.
2. **Phép thử OOD chưa hợp lệ.** Trong thăm dò (một seed, 60 epoch) định tuyến chuẩn ghép được tuyến mới; trong cấu hình khóa (40 epoch, 8 seed) thì không (MSE OOD 0,259). Chúng tôi không đoán nguyên nhân; có thể do số epoch, độ biến thiên theo seed hoặc tuyến có module lặp. Điều này khiến H1 và H5 không thể kết luận, theo đúng quy tắc đã khóa.
3. **Baseline dày không tệ.** A, B, C đạt MSE trong phân phối ≈ 0,01 với MAC bằng hoặc ít hơn D nhiều lần; chúng không ghép được tuyến mới nhưng D cũng không.
4. **Độ phân tán lớn.** Độ lệch chuẩn OOD của D (0,31) lớn so với trung bình; 8 seed cho khoảng tin cậy rộng.

## 5. Giới hạn
Nhiệm vụ tổng hợp nhỏ, chiều 8; bộ mô tả tuyến được đưa vào mô hình; bộ định tuyến chỉ phụ thuộc bộ mô tả và vòng lặp, không phụ thuộc trạng thái module; một họ siêu tham số (ngân sách tìm kiếm nhỏ: hai giá trị lr, một seed); chi phí tính toán là ước lượng MAC; không có kết luận về LLM, suy luận, hay genus. Thiết kế được chỉnh bằng thăm dò trước khi khóa.

## 6. Việc tiếp theo (đề xuất, chưa thực hiện)
Chạy v2 với tiền đăng ký mới: huấn luyện lâu hơn hoặc dùng phạt/giám sát cạnh để bộ định tuyến học được tuyến; chỉ tiếp tục kiểm tra OOD sau khi V4 đạt ở một seed hiệu chỉnh riêng. Kết quả v1 được giữ nguyên trong Lineage, không sửa hồi tố.

## 7. Tái lập
`python src/run_all.py` rồi `python src/analyze.py`; kiểm tra sổ: `python src/lineage.py verify`. Thư viện: JAX (CPU), numpy, scipy, matplotlib.

---

# Phần II — Lượt v2 (tiền đăng ký PREREG_v2, 12 seed mới)

## 8. Thay đổi so với v1
100 epoch (v1: 40); seed 8–19; thêm hai nhánh huấn luyện cổng mềm chuyển dần sang cứng (D_anneal, D_ptr_anneal); khoảng tin cậy 98,75% cho 4 kiểm định OOD; ngưỡng V3 nới còn 2,0 × MAC của A. Chi tiết và lý do: `prereg/PREREG_v2.md`, Lineage L011–L012. Kỳ vọng được ghi trước khi chạy: không nhánh nào đạt H3.

## 9. Kết quả (trung bình ± độ lệch chuẩn, 12 seed; mức dự đoán trung bình OOD = 0,203)
| Mô hình | MSE trong phân phối | MSE OOD | MAC/mẫu | Precision / recall cạnh (OOD) |
|---|---|---|---|---|
| A | 0,0077 ± 0,0003 | 0,343 ± 0,041 | 28 360 | — |
| B | 0,0089 ± 0,0005 | 0,351 ± 0,062 | 28 360 | — |
| C | 0,0094 ± 0,0004 | 0,357 ± 0,041 | 145 344 | — |
| D | 0,0767 ± 0,0165 | 0,326 ± 0,061 | 39 949 | 0,012 / 0,021 |
| D_anneal | 0,0726 ± 0,0080 | 0,525 ± 0,243 | 46 273 | 0,037 / 0,075 |
| D_ptr | 0,0455 ± 0,0317 | 0,308 ± 0,050 | 14 788 | 0,172 / 0,283 |
| D_ptr_anneal | 0,0466 ± 0,0105 | 0,458 ± 0,109 | 21 079 | 0,106 / 0,213 |
| D_oracle (chẩn đoán) | 0,0024 ± 0,0018 | 0,169 ± 0,123 | 10 195 | — |
| D_fixed | 0,1231 ± 0,0058 | 0,302 ± 0,106 | 33 429 | — |

Hình: `figures/v2_mse_theo_mo_hinh.png`.

### 9.1 Kết luận theo giả thuyết
| Giả thuyết | Kết quả | Nhãn |
|---|---|---|
| H1: D so với A, OOD | r = −4,0% [−18,0%; +10,6%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H1b: D_anneal so với A, OOD | r = +57,7% [+5,3%; +128,1%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H2: hiệu quả tính toán (D, D_anneal) | MSE trong phân phối gấp ~9–10 lần A; MAC 1,41× và 1,63× | KHÔNG ĐẠT (FAIL) |
| H3: độ trung thực định tuyến (4 nhánh) | recall tối đa 0,28 (D_ptr) so với ngưỡng 0,8 | KHÔNG ĐẠT (FAIL) |
| H4: D so với D_fixed, OOD | r = +15,6% [−1,7%; +31,4%] | KHÔNG ĐẠT (FAIL) |
| H5: D_ptr so với A, OOD | r = −9,5% [−20,4%; +2,0%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H5b: D_ptr_anneal so với A, OOD | r = +36,9% [+8,5%; +66,4%]; vi phạm V4 | CHƯA KẾT LUẬN (UNRESOLVED) |

Không giả thuyết nào ĐẠT. Điều kiện V4 không thỏa (MSE OOD trung bình của D_oracle là 0,169 > 0,102), nên không thể phát biểu gì chắc chắn về OOD cho H1, H1b, H5, H5b; riêng H1b và H5b sẽ là KHÔNG ĐẠT nếu V4 thỏa.

## 10. Thảo luận v2
1. **Thăm dò một seed đã quá lạc quan.** D_oracle đạt MSE OOD 0,0115 ở seed 1000 nhưng trung bình 0,169 (phân tán lớn, độ lệch chuẩn 0,123) trên 12 seed mới: tác dụng của việc huấn luyện lâu hơn phụ thuộc seed, chưa đủ để làm phép thử OOD ổn định.
2. **Cổng mềm chuyển dần sang cứng không giúp.** Nhánh anneal không cải thiện khôi phục tuyến và làm MSE OOD tệ hơn (D: 0,326 → D_anneal: 0,525), đồng thời dùng đủ 8 cạnh/mẫu, tức nhiều cạnh thừa (96%).
3. **Bộ định tuyến con trỏ là hướng ít tệ nhất.** D_ptr có recall cạnh cao nhất (0,28), MSE trong phân phối tốt hơn D (0,046 so với 0,077) và MAC chỉ 14,8 nghìn; nhưng vẫn xa ngưỡng 0,8 và xa baseline dày (0,008).
4. **Định tuyến phụ thuộc đầu vào có ích trong phân phối nhưng không ích trên OOD.** D tốt hơn D_fixed trong phân phối (0,077 so với 0,123) nhưng OOD không khác biệt (H4 KHÔNG ĐẠT).
5. **Dù đã chỉnh, kết luận của v1 không đổi:** trên nhiệm vụ này, định tuyến thích ứng chưa cho lợi ích đo được so với tính toán độ sâu cố định dưới ngân sách tham số tương đương.

## 11. Giới hạn bổ sung và việc tiếp theo
Phép thử OOD chưa hợp lệ (V4) nên câu hỏi tổng quát hóa sang tuyến mới vẫn mở. Các hướng chưa thử, đã nêu từ trước: cổng dừng, phạt cạnh mạnh hơn, bộ định tuyến thấy trạng thái module, giám sát yếu trên cạnh, nhiệm vụ cần suy tuyến từ x. Mỗi hướng cần tiền đăng ký riêng (v3). Cơ chế nhân quả vẫn không thể khẳng định.

---

# Phần III — Lượt v3: kiểm định các "điểm gãy" rút ra từ tài liệu (PREREG_v3)

## 12. Cơ sở và thiết kế
Tóm tắt tài liệu (dựa trên tóm tắt/đoạn trích, chưa đọc toàn văn; Lineage L023): học đồng thời module và bố cục khó; gradient cổng rời rạc lệch/thiếu, mô hình định tuyến học thường thua mô hình dày cùng tham số; bố cục học end-to-end thường sai và tổng quát hóa nhạy với bố cục; chuyên biệt hóa và định tuyến theo nội dung (attention thưa). Thiết kế v3 (`prereg/PREREG_v3.md`): thêm D_attn (định tuyến attention theo nội dung, cổng cứng), D_sup (như D_attn + giám sát bố cục), D_soft (cổng mềm dày đặc), E (MoE top-2), D_oracle (mạng module với bố cục đúng); tham số khớp ±5% (~28 nghìn); 100 epoch.
**Sàng lọc seed bằng oracle:** v2 cho thấy D_oracle hội tụ chỉ ở khoảng một phần ba seed (hai pha: hội tụ về 0 hoặc kẹt ở vùng bằng phẳng). v3 chỉ giữ seed mà oracle hội tụ (26 seed được sàng, 10 seed hợp lệ). Hệ quả: kết luận chỉ áp dụng cho "thế giới" mà mô hình chuẩn học được; số liệu D_oracle bị chọn lọc theo thiết kế.

## 13. Kết quả (10 seed hợp lệ; trung bình ± độ lệch chuẩn; mức dự đoán hằng số: 0,213 trong phân phối, 0,215 OOD)
| Mô hình | MSE trong phân phối | MSE OOD | MAC/mẫu | Precision / recall cạnh (OOD) |
|---|---|---|---|---|
| A | 0,0076 ± 0,0003 | 0,350 ± 0,041 | 28 360 | — |
| B | 0,0087 ± 0,0004 | 0,344 ± 0,050 | 28 360 | — |
| C | 0,0091 ± 0,0004 | 0,398 ± 0,071 | 145 344 | — |
| E (MoE) | 0,0098 ± 0,0005 | 0,328 ± 0,031 | 39 424 | — |
| D_attn | 0,1447 ± 0,0037 | 0,185 ± 0,010 | 40 836 | 0,089 / 0,173 |
| D_sup | 0,0125 ± 0,0168 | 0,526 ± 0,108 | 32 996 | 0,969 / 0,970 |
| D_soft | 0,0089 ± 0,0006 | 0,341 ± 0,042 | 172 832 | 0,027 / 0,007 |
| D_oracle (chẩn đoán, seed được chọn) | 0,0000 | 0,015 ± 0,003 | 10 195 | — |

Hình: `figures/v3_ket_qua.png`.

### 13.1 Kết luận theo giả thuyết
| Giả thuyết | Kết quả | Nhãn |
|---|---|---|
| H1: D_attn so với baseline tốt nhất (E), OOD | r = −43,2% [−47,4%; −39,5%] — xem cảnh báo | ĐẠT (PASS) theo chữ, không có ý nghĩa |
| H2: D_sup so với E, OOD | r = +63,4% [+36,1%; +97,8%] | KHÔNG ĐẠT (FAIL) |
| H3a: D_sup khôi phục cạnh (OOD) | precision 0,969; recall 0,970 | ĐẠT (PASS) |
| H3b: D_attn khôi phục cạnh (OOD) | precision 0,089; recall 0,173 | KHÔNG ĐẠT (FAIL) |
| H4: D_soft so với D_attn, trong phân phối | r = −93,8% [−94,0%; −93,7%] | ĐẠT (PASS) |
| H5: E không thua A, trong phân phối | r = +28,4% [+26,7%; +30,2%] | KHÔNG ĐẠT (FAIL) |
| H6: sụp đổ/chuyên biệt hóa | D_attn: entropy 0,88, độ phủ nút 0,47; D_sup: 0,99 và 0,98 | D_attn KHÔNG ĐẠT; D_sup ĐẠT |
| H7: hiệu quả tính toán của D_attn | MSE gấp ~19 lần A; MAC 1,44× | KHÔNG ĐẠT (FAIL) |

**Cảnh báo về H1.** MSE OOD của D_attn (0,185) xấp xỉ mức dự đoán hằng số (0,215), còn các baseline dày tệ hơn hằng số (0,33–0,40) vì ngoại suy sai. MSE trong phân phối của D_attn là 0,145, gấp ~19 lần A: nó học yếu và không mắc lỗi ngoại suy lớn, nên "thắng" theo quy tắc. Tiền đăng ký thiếu điều kiện "phải tốt hơn dự đoán hằng số". Nhãn giữ nguyên nhưng không có căn cứ kết luận định tuyến theo nội dung có lợi.

## 14. Thảo luận: các điểm gãy đã xác định được gì
1. **Gradient của cổng cứng (ST) là điểm gãy được ủng hộ mạnh nhất** (H4): cùng bộ định tuyến attention, cổng mềm cho MSE trong phân phối 0,0089, cổng cứng 0,145. Lưu ý D_soft là mạng trộn dày đặc, tốn khoảng 6 lần MAC và không định tuyến thưa; vậy bằng chứng nói rằng "cổng cứng khó huấn luyện", chưa nói rằng "định tuyến thưa tốt hơn".
2. **Khôi phục bố cục không phải điểm gãy khi có giám sát** (H3a ĐẠT, 0,97) nhưng là điểm gãy khi không giám sát (H3b KHÔNG ĐẠT, 0,17). Tức là tín hiệu học cho bố cục là thiếu, khớp với tài liệu (Bahdanau và cộng sự).
3. **Còn một điểm gãy thứ hai, chưa được kiểm định có tiền đăng ký:** dù bố cục đúng (D_sup), MSE OOD 0,526 và MSE trong phân phối 0,0125 (thường kẹt khoảng 0,006), trong khi D_oracle cùng bố cục (seed được chọn) đạt 0,015. Gợi ý rằng huấn luyện/biểu diễn của module (vùng bằng phẳng, học thuộc theo tuyến) là điểm gãy khi bố cục đã đúng. Chưa tách được vai trò của đầu đọc học được và cổng cứng ST; cần thí nghiệm riêng.
4. **MoE thua mô hình dày cùng tham số** (H5, +28% tương đối), phù hợp với tài liệu, dù chênh lệch tuyệt đối nhỏ (0,0098 so với 0,0076).
5. **Chuyên biệt hóa chỉ xuất hiện khi có giám sát** (H6): D_attn có độ phủ nút 0,47; D_sup 0,98. Cắt bỏ module/hoán đổi bộ định tuyến trên D_attn cho thấy đầu ra vẫn nhạy với bộ mô tả (MSE trong phân phối 0,145 → 0,292) nhưng không có tính toán đúng nghĩa.
6. **Câu hỏi gốc (lợi ích so với độ sâu cố định dưới ngân sách tương đương):** vẫn chưa có bằng chứng ủng hộ. Mô hình dày vẫn tốt nhất trong phân phối ở mọi lượt (v1, v2, v3).

## 15. Giới hạn và việc tiếp theo
Sàng lọc oracle làm hẹp phạm vi suy luận; D_oracle bị chọn lọc theo thiết kế; không mở rộng số quy tắc (M=8) và không có RIMs đầy đủ; bộ định tuyến vẫn nhận bộ mô tả tuyến sẵn; một cấu hình siêu tham số cho mỗi nhánh; tài liệu mới đọc ở mức tóm tắt. Đề xuất cho v4 (cần tiền đăng ký): (a) kiểm định điểm gãy thứ hai bằng cách thay lần lượt đầu đọc học được và cổng cứng ST trong D_sup bằng phiên bản của oracle; (b) thêm điều kiện "phải tốt hơn dự đoán hằng số" cho mọi giả thuyết OOD; (c) dùng ước lượng gradient tốt hơn ST cho cổng thưa có giám sát yếu; (d) mở rộng số quy tắc của nhiệm vụ.

---

# Phần IV — Lượt v4: tách đầu đọc khỏi cổng, điều kiện C0, ước lượng gradient tốt hơn (PREREG_v4)

## 16. Thiết kế
(1) **Tách đầu đọc/cổng** trong D_sup bằng thiết kế 2×2: D_sup (cổng học có giám sát, đầu đọc học), D_sup_trueo (cổng học, đầu đọc đúng), D_trueG (cổng đúng, đầu đọc học), D_oracle (cả hai đúng; dùng để sàng lọc seed). (2) **Điều kiện C0**: mọi giả thuyết OOD đòi hỏi MSE OOD trung bình của nhánh ≤ 0,5 × dự đoán hằng số (0,110); nếu quy tắc δ thỏa nhưng C0 không thỏa thì nhãn là KHÔNG ĐẠT. C0-ID tương tự cho MSE trong phân phối. (3) **Ước lượng gradient** cho định tuyến phân loại không giám sát (mỗi vòng chọn một nguồn và một đích): ST, Gumbel-ST, ReinMax (kiểm tra trên bài toán đồ chơi: sai số gradient bậc hai 0,005 so với 0,243 của ST), REINFORCE. Seed 50–66 sàng bằng oracle → 8 seed hợp lệ; 100 epoch, lr 1e-2; tham số khớp ±5%.

## 17. Kết quả (8 seed; trung bình ± độ lệch chuẩn; dự đoán hằng số: 0,217 trong phân phối, 0,219 OOD; ngưỡng C0 0,110)
| Mô hình | MSE trong phân phối | MSE OOD | Precision / recall cạnh (OOD) |
|---|---|---|---|
| A | 0,0075 ± 0,0003 | 0,352 ± 0,029 | — |
| D_sup | 0,0284 ± 0,0363 | 0,734 ± 0,379 | 0,985 / 0,978 |
| D_sup_trueo (đầu đọc đúng) | 0,1229 ± 0,0452 | 0,226 ± 0,133 | 0,770 / 0,584 |
| D_trueG (cổng đúng) | 0,0045 ± 0,0002 | 0,490 ± 0,081 | 1,0 / 1,0 |
| D_cat_st (ST) | 0,1362 ± 0,0044 | 0,189 ± 0,023 | 0,245 / 0,158 |
| D_cat_gs (Gumbel-ST) | 0,1251 ± 0,0069 | 0,209 ± 0,013 | 0,307 / 0,138 |
| D_cat_rm (ReinMax) | 0,1220 ± 0,0112 | 0,220 ± 0,051 | 0,059 / 0,038 |
| D_cat_rf (REINFORCE) | 0,1264 ± 0,0073 | 0,406 ± 0,237 | 0,038 / 0,024 |
| D_oracle (chẩn đoán, seed được chọn) | 0,0001 ± 0,0003 | 0,019 ± 0,013 | — |

Hình: `figures/v4_ket_qua.png`.

### 17.1 Kết luận theo giả thuyết
| Giả thuyết | Kết quả | Nhãn |
|---|---|---|
| H1: D_sup_trueo so với D_sup, OOD | r = −58,6% [−80,3%; −28,5%]; không thỏa C0 | KHÔNG ĐẠT (FAIL) |
| H2: D_trueG so với D_sup, OOD | r = −16,3% [−44,4%; +21,4%]; không thỏa C0 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H3: gradient tốt hơn ST (trong phân phối) | Gumbel-ST −8,0% [−13,5; −2,3]; ReinMax −10,5% [−15,2; −3,8] nhưng không thỏa C0-ID; REINFORCE −7,2% [−10,4; −2,8] | KHÔNG ĐẠT (FAIL) |
| H4: khôi phục cạnh không giám sát | recall 0,02–0,16 (ngưỡng 0,8) | KHÔNG ĐẠT (FAIL) |
| H5: ghép tuyến mới không giám sát | MSE OOD 0,19–0,22 ≈ hằng số, không thỏa C0 (REINFORCE chưa kết luận) | KHÔNG ĐẠT (FAIL) |
| H6: D_sup so với A, OOD, với C0 | r = +112,8% [+42,7%; +200,9%] | KHÔNG ĐẠT (FAIL) |

Không giả thuyết nào ĐẠT. Kỳ vọng ghi trước sai ở một điểm: H2 (cổng đúng + đầu đọc học) không khôi phục khả năng ghép.

## 18. Thảo luận
1. **Ước lượng gradient tốt hơn chỉ cải thiện ~10% và không đủ.** ReinMax giảm MSE trong phân phối ~10% so với ST, nhất quán, nhưng mọi nhánh không giám sát vẫn ở 0,12–0,14 (chỉ tốt hơn dự đoán hằng số 0,217 khoảng 40%), và độ phủ cạnh còn thấp hơn ST. Gradient là điểm gãy ở v3 (cổng cứng so với mềm) nhưng ở thiết kế phân loại này việc đổi bộ ước lượng không phải là chìa khóa.
2. **C0 thay đổi bức tranh.** Ba nhánh không giám sát "thắng" baseline dày A trên OOD (−38% đến −46%) chỉ vì A tệ hơn hằng số (0,352 so với 0,219), còn các nhánh này xấp xỉ hằng số (0,19–0,22). Không có nhánh nào ghép được tuyến mới; điều kiện C0 ngăn kết luận sai.
3. **Bố cục đúng không đủ.** D_sup khôi phục cạnh 0,98 nhưng MSE OOD 0,734; D_trueG có cổng đúng tuyệt đối nhưng vẫn kẹt (MSE trong phân phối 0,0045 ở cả 8 seed, độ lệch chuẩn 0,0002; MSE OOD 0,490).
4. **Gợi ý (chưa kiểm định có tiền đăng ký): đầu đọc học là thủ phạm.** D_oracle (cổng đúng + đầu đọc đúng) hội tụ về 0,0001 và ghép được tuyến mới (MSE OOD 0,019), D_trueG chỉ khác ở đầu đọc học và kẹt 8/8 seed. Can thiệp lúc đánh giá cho thấy cổng học của D_sup gần như bằng cổng đúng (thay cổng: MSE không đổi 0,0284), còn thay đầu đọc học bằng đầu đọc đúng làm MSE tăng lên 7,4: đầu ra phụ thuộc vào một hỗn hợp mềm trên nhiều module, không phải đầu ra chuẩn của module cuối. Cần thí nghiệm riêng (v5) để xác nhận.
5. **Sàng lọc bằng oracle chọn khởi tạo may mắn.** Chẩn đoán D0: D_oracle hội tụ 9/18 lần khi đổi khởi tạo (thế giới "hợp lệ" ở v3: 5/9; "không hợp lệ": 4/9). Vùng bằng phẳng không chủ yếu là tính chất của thế giới mà phần lớn là may rủi theo khởi tạo (có thể kèm hiệu ứng thế giới: thế giới 21 là 0/3). Hệ quả: số liệu D_oracle bị chọn lọc, và so sánh với các nhánh khác (khởi tạo khác) có thiên lệch. Diễn giải "kết luận chỉ áp dụng cho thế giới học được" ở v3 cần hiệu chỉnh: đúng hơn là "áp dụng cho khởi tạo may mắn của oracle".
6. **Quan sát chưa kiểm định (D_sup_trueo):** với đầu đọc đúng, cổng học có giám sát không hội tụ (recall 0,58). Giả thuyết: gradient MSE nhiễu qua ST phá giám sát cổng khi đầu ra chưa đúng.

## 19. Giới hạn và đề xuất v5
Sàng lọc oracle chọn khởi tạo may mắn; D0 chỉ có 18 lần chạy; vẫn 8 phép toán; bộ định tuyến nhận bộ mô tả tuyến sẵn; một cấu hình siêu tham số mỗi nhánh; điểm 4 và 6 là gợi ý chưa kiểm định. Đề xuất v5 (cần tiền đăng ký): (a) thay sàng lọc seed bằng khởi động lại nhiều lần (chọn khởi tạo theo loss huấn luyện, không theo OOD) cho MỌI nhánh; (b) kiểm định trực tiếp vai trò đầu đọc học: ép đầu đọc thành one-hot/sparsemax, hoặc phạt entropy; (c) kiểm định giả thuyết nhiễu MSE qua ST bằng cách dừng gradient MSE vào cổng khi giám sát; (d) mở rộng số quy tắc của nhiệm vụ.

---

# Phần V — Lượt v5: khởi động lại nhiều lần, vai trò của đầu đọc, nhiễu gradient MSE vào cổng (PREREG_v5)

## 20. Thiết kế
(1) **Khởi động lại nhiều lần thay sàng lọc seed:** mọi nhánh chạy 3 khởi tạo trên mỗi thế giới (5 thế giới, seed 80–84); lần được chọn là lần có loss huấn luyện nhỏ nhất (không theo OOD); điều kiện V4' thay cho sàng lọc: D_oracle tốt nhất trong 3 lần phải hội tụ ở ≥ 4 thế giới. (2) **Đầu đọc, cổng đúng để cô lập:** D_trueG (softmax học, khởi tạo ngẫu nhiên), D_trueG_oh (ép one-hot bằng xuyên thẳng), D_trueG_ent (phạt entropy 0,05), D_trueG_id (softmax học, khởi tạo nhọn đúng), D_oracle (đầu đọc đúng cố định). (3) **Nhiễu gradient MSE vào cổng, đầu đọc đúng để cô lập:** D_sup_trueo so với D_sup_trueo_sg (dừng gradient MSE vào cổng); nhánh kết hợp D_sup_sg_oh. Điều kiện C0/C0-ID áp dụng như v4.

## 21. Kết quả (5 thế giới, lần tốt nhất trong 3; trung bình ± độ lệch chuẩn; hằng số: 0,210 trong phân phối, 0,209 OOD; ngưỡng C0 0,105)
| Mô hình | MSE trong phân phối | MSE OOD | Lần hội tụ (trên 15) | Precision / recall cạnh (OOD) |
|---|---|---|---|---|
| A | 0,0077 ± 0,0002 | 0,351 ± 0,034 | 0 | — |
| D_oracle (cổng + đầu đọc đúng) | 0,0008 ± 0,0015 | 0,055 ± 0,086 | 5 | — |
| D_trueG (cổng đúng, đầu đọc ngẫu nhiên) | 0,0045 ± 0,0002 | 0,426 ± 0,031 | 0 | 1,0 / 1,0 |
| D_trueG_oh (one-hot) | 0,0410 ± 0,0123 | 0,265 ± 0,052 | 0 | 1,0 / 1,0 |
| D_trueG_ent (phạt entropy) | 0,0067 ± 0,0004 | 0,487 ± 0,211 | 0 | 1,0 / 1,0 |
| D_trueG_id (đầu đọc khởi tạo đúng) | 0,0005 ± 0,0009 | 0,062 ± 0,062 | 5 | 1,0 / 1,0 |
| D_sup_trueo (đầu đọc đúng, cổng ST) | 0,1031 ± 0,0492 | 0,276 ± 0,108 | 0 | 0,943 / 0,832 |
| D_sup_trueo_sg (dừng gradient MSE) | 0,0901 ± 0,0461 | 0,275 ± 0,118 | 0 | 0,978 / 0,989 |
| D_sup_sg_oh (cổng giám sát + one-hot) | 0,1085 ± 0,0228 | 0,256 ± 0,052 | 0 | 0,975 / 1,000 |

Hình: `figures/v5_ket_qua.png`.

### 21.1 Kết luận theo giả thuyết
| Giả thuyết | Kết quả | Nhãn |
|---|---|---|
| H1: D_trueG_oh so với D_trueG (trong phân phối) | r = +812% | KHÔNG ĐẠT (FAIL) |
| H2: D_trueG_ent so với D_trueG | r = +48,6% | KHÔNG ĐẠT (FAIL) |
| H3: D_trueG_id so với D_trueG | r = −88,8% [−99,9%; −69,0%]; hội tụ 5/15 so với 0/15 | ĐẠT (PASS) |
| H4: dừng gradient MSE, hiệu recall OOD | +0,157 [+0,029; +0,382] (ngưỡng 0,2) | CHƯA KẾT LUẬN (UNRESOLVED) |
| H5c: D_trueG_id so với A, OOD | r = −80,4%; thỏa C0 | ĐẠT (PASS) |
| H5a, H5b: one-hot, entropy so với A, OOD | không thỏa C0 | CHƯA KẾT LUẬN (UNRESOLVED) |
| H6: D_sup_sg_oh so với A, OOD | quy tắc δ thỏa (−25,3%), C0 không thỏa | KHÔNG ĐẠT (FAIL) |
| H7: D_sup_sg_oh khôi phục cạnh OOD | precision 0,975; recall 1,000 | ĐẠT (PASS) |

Điều kiện hợp lệ V0, V1, V4' thỏa. Kỳ vọng ghi trước sai ở H4 (không đạt ngưỡng).

## 22. Thảo luận
1. **Khi cổng đúng và đầu đọc được khởi tạo đúng, mô hình ghép được tuyến mới** (H3, H5c ĐẠT): MSE OOD 0,062 so với 0,351 của baseline dày và 0,426 của D_trueG. Đây là bằng chứng đầu tiên trong chuỗi v1–v5 rằng kiến trúc module định tuyến có thể tổng quát hóa sang tuyến chưa thấy tốt hơn mô hình dày, với điều kiện cổng đúng được cho trước.
2. **Thủ phạm của lỗi ở v4 không phải khả năng biểu diễn của đầu đọc học, mà là tối ưu hóa từ khởi tạo ngẫu nhiên.** D_trueG khởi tạo ngẫu nhiên hội tụ 0/15 lần, D_trueG_id (chỉ khác khởi tạo đầu đọc) hội tụ 5/15, bằng D_oracle; Fisher p = 0,042 (mô tả, chưa hiệu chỉnh; các lần chạy không hoàn toàn độc lập). Hai cách ép đầu đọc thành thưa từ khởi tạo ngẫu nhiên (one-hot xuyên thẳng, phạt entropy) đều tệ hơn, nên chưa có cách học đầu đọc từ đầu nào hoạt động trong thiết kế này. Khởi tạo đúng đã đưa sẵn đáp án vào đầu đọc nên không chứng minh đầu đọc học được từ đầu.
3. **Vùng bằng phẳng là hiện tượng chung:** ngay cả D_oracle chỉ hội tụ 5/15 lần. Khởi động lại 3 lần cứu được 4/5 thế giới cho D_oracle và D_trueG_id (lần tốt nhất hội tụ); thế giới 83 không có lần nào hội tụ cho D_trueG_id.
4. **Dừng gradient MSE vào cổng giúp một phần** (H4 CHƯA KẾT LUẬN): recall OOD 0,989 so với 0,832; riêng nhánh D_sup_sg_oh đạt precision 0,975 và recall 1,000 (H7 ĐẠT), tức cổng học bằng giám sát đạt bố cục gần đúng tuyệt đối, kể cả tuyến OOD.
5. **Nhưng cổng học + đầu đọc học chưa ghép được tuyến mới** (H6 KHÔNG ĐẠT): D_sup_sg_oh có cổng gần hoàn hảo mà MSE OOD 0,256 vì đầu đọc one-hot không hội tụ. Điểm gãy còn lại là học đầu đọc từ khởi tạo ngẫu nhiên.
6. **Câu hỏi gốc:** trong phạm vi này, chưa có nhánh nào tự học được cả bố cục (không giám sát) lẫn đầu đọc mà ghép được tuyến mới. Có một đường đi đã chứng minh được (cổng đúng + đầu đọc khởi tạo đúng: H5c); con đường còn lại phụ thuộc vào cách khởi tạo/học đầu đọc.

## 23. Giới hạn và đề xuất v6
5 thế giới × 3 khởi tạo (mẫu nhỏ; 15 lần chạy không độc lập hoàn toàn); chọn lần theo loss huấn luyện; D_trueG_id dùng khởi tạo đúng (thông tin đáp án); β và hệ số khởi tạo chọn một lần không dò; vẫn 8 phép toán; bộ mô tả tuyến đưa sẵn. Đề xuất v6 (cần tiền đăng ký): (a) học đầu đọc từ khởi tạo ngẫu nhiên bằng cách khác (khởi tạo cân bằng/đối xứng, nhiệt độ giảm dần, hoặc ReinMax cho đầu đọc phân loại); (b) kết hợp cổng giám sát có dừng gradient MSE với đầu đọc khởi tạo đúng/học được để kiểm tra khả năng ghép tuyến mới đầu-cuối; (c) giảm bớt giám sát cổng (giám sát yếu, hoặc chỉ một phần tuyến); (d) mở rộng số quy tắc và thế giới.
