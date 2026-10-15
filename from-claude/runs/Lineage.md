# Lineage — sổ dòng dõi thực nghiệm TNR (chỉ ghi thêm)

Quy ước: mỗi mục chỉ được THÊM vào cuối tệp. Băm của mục = SHA-256(băm_trước + nội_dung_mục). Kiểm tra: `python src/lineage.py verify`.
Trạng thái hợp lệ: ĐẠT (PASS) · KHÔNG ĐẠT (FAIL) · CHƯA KẾT LUẬN (UNRESOLVED) · THĂM DÒ (không mang tính bằng chứng).

## [L001] Pha thăm dò và hiệu chỉnh thiết kế (seed 1000)
- Thời điểm: 2026-10-07 09:42 UTC
- Trạng thái: **THĂM DÒ**

Dữ liệu dùng: chỉ seed hiệu chỉnh 1000, tách biệt với seed đánh giá chính thức (0–7). Kết quả pha này KHÔNG phải bằng chứng.

Các thay đổi thiết kế và lý do (đều xảy ra TRƯỚC khi khóa tiền đăng ký):
1. Module không được thấy bộ mô tả tuyến c (tránh rò rỉ thông tin tuyến vào module); riêng C (tĩnh) nhận c ở nút đầu vào để không bị tước điều kiện.
2. Tham số hóa cạnh đổi từ W_ij riêng từng cạnh sang "bus" chung (msg_j = A_j · Σ_i g_ij B_i h_i), giảm khả năng học thuộc theo từng cạnh.
3. Nhiệm vụ nhẹ hơn: chiều 8, độ mạnh phi tuyến 1.0 (ban đầu 16 chiều, 1.5: ngay cả oracle cũng không học nổi trong phân phối).
4. Số tuyến huấn luyện: với 9 và 15 tuyến, oracle (định tuyến đúng tuyệt đối) KHÔNG ghép được tuyến OOD; với 40 tuyến, oracle ghép được (MSE OOD ≈ 0 trên 4/5 tuyến; O4 có module lặp nên 0.049). Do đó chọn 40 tuyến huấn luyện (sinh cố định, phủ mọi cặp chuyển tiếp của tuyến OOD).
5. Chọn tốc độ học bằng MSE val trong phân phối trên seed 1000, lưới {3e-3, 1e-2}: cả A, B, C, D, D_ptr đều chọn 1e-2.

Số liệu pilot đáng ghi (seed 1000, 40 epoch, lr 1e-2, MSE; mức dự đoán trung bình = 0.214): A 0.0087 (OOD 0.299); B 0.0098 (0.265); C 0.0112 (0.358); D 0.0397 (0.310); D_ptr 0.0188 (0.349). Độ phủ cạnh đúng của D ≈ 0.05, của D_ptr ≈ 0.40.
Nhận xét thăm dò: bộ định tuyến chưa tự học được tuyến chỉ từ MSE + ràng buộc K cạnh; điều này được ghi nhận, KHÔNG được dùng để chỉnh thêm siêu tham số sau khi khóa.
<!-- prev:0000000000000000000000000000000000000000000000000000000000000000 hash:c3d2af933fe352a20bbc2ebee12de1345762b98c6b14244c4b7607c199904d34 -->

## [L002] Khóa tiền đăng ký PREREG_v1
- Thời điểm: 2026-10-07 09:43 UTC
- Trạng thái: **THĂM DÒ**

Tệp: prereg/PREREG_v1.md, SHA-256 = 27cac6bef70bb3de131d47ea895f2cdcdd5d6f2e5f03c7934fb1d9297c9bfa7c.
Khóa: H1–H5, điều kiện hợp lệ V1–V5, quy tắc quyết định (biên δ = −10%, bootstrap 95%), seed 0–7, lr 1e-2, 40 epoch.
Ghi chú trung thực: tiến trình chạy chính thức (src/run_all.py) được khởi động vài giây TRƯỚC khi mục này được ghi; chưa có kết quả nào được xem tại thời điểm đó. Mục V3 (ngưỡng 1.5× MAC của A) được đặt sau khi biết MAC pilot của D (~1.4× A), nên đây là ngưỡng có thông tin từ pilot.
<!-- prev:c3d2af933fe352a20bbc2ebee12de1345762b98c6b14244c4b7607c199904d34 hash:f97bf8118a955a7d08fd7d42adcaa8a0063a3a0fb541d97aa078da357b0c0857 -->

## [L003] Chạy chính thức run_001_main hoàn tất
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **THĂM DÒ**

8 seed (0–7) × 12 cấu hình = 96 tệp kết quả trong runs/run_001_main/ (meta.json ghi bề rộng đã khớp tham số, lr 1e-2, 40 epoch, batch 200).
Ghi chú gián đoạn: tiến trình nền bị môi trường dừng ở seed 4 (sau khi A, B, C của seed 4 xong) và được khởi động lại; script bỏ qua tệp đã có, không chạy lại, không đổi cấu hình. Sau đó cần thêm một lần khởi động lại do cùng nguyên nhân (bảo vệ bằng kiểm tra tiến trình). Không có tệp kết quả nào bị ghi đè.
Phân tích: src/analyze.py, đúng theo PREREG_v1; đầu ra results/results.json.
<!-- prev:f97bf8118a955a7d08fd7d42adcaa8a0063a3a0fb541d97aa078da357b0c0857 hash:2c60f66441c6a56d14125614b0fedf0c60f837b8f9958ba9f96fdf670712c4f3 -->

## [L004] H1 — D so với baseline tốt nhất trên MSE OOD
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

Baseline tham chiếu: B (MSE OOD thấp nhất trong A,B,C). Chênh lệch tương đối trung bình r = +40,8%, KTC 95% [−5,1%; +104,6%].
Vi phạm điều kiện hợp lệ V4: D_oracle có MSE OOD trung bình 0,259, không < 0,5 × 0,197 = 0,098; tức phép thử OOD không phân biệt được ngay cả với định tuyến chuẩn. (V1, V2, V3, V5 thỏa.)
Theo §3 của tiền đăng ký, vi phạm điều kiện hợp lệ → CHƯA KẾT LUẬN. Ghi chú: ngay cả không xét V4, D không có lợi thế (r trung bình dương).
<!-- prev:2c60f66441c6a56d14125614b0fedf0c60f837b8f9958ba9f96fdf670712c4f3 hash:13faeeb661f2d356ade86c047538845b9ef0d9ba04bd98f459956da14e799195 -->

## [L005] H2 — hiệu quả tính toán tại độ chính xác tương đương (trong phân phối)
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Baseline tham chiếu: A. MSE test của D trung bình 0,069 so với A 0,0088: r = +690%, KTC 95% [+528%; +879%] (cận dưới > +10%). MAC(D)/MAC(A) = 1,37 (> 0,5). Cả hai vế đều không đạt.
<!-- prev:13faeeb661f2d356ade86c047538845b9ef0d9ba04bd98f459956da14e799195 hash:5f781b4600df248baa4cc2d9c490338a3bf353e1bb415b29356c2cbc1234caf4 -->

## [L006] H3 — độ trung thực định tuyến trên tuyến OOD
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Sau căn chỉnh hoán vị module–phép toán (ước lượng chỉ từ dữ liệu huấn luyện): độ chính xác cạnh (precision) 0,009; độ phủ cạnh (recall) 0,019, xa ngưỡng 0,8. Trong phân phối: precision 0,020; recall 0,032. Bộ định tuyến MLP không khôi phục được tuyến phụ thuộc thật.
<!-- prev:5f781b4600df248baa4cc2d9c490338a3bf353e1bb415b29356c2cbc1234caf4 hash:8c2bfdaa03d7542d391b45ac47cdf29db7e0a5f27e8a3c2fa94da7db6532451d -->

## [L007] H4 — hiệu ứng định tuyến: D so với D_fixed trên MSE OOD
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r trung bình = +61,0%, KTC 95% [−12,5%; +168,6%]. MSE OOD: D 0,460 (độ lệch chuẩn 0,31), D_fixed 0,323. Không có bằng chứng định tuyến phụ thuộc đầu vào giúp ích trên OOD; độ phân tán giữa các seed của D rất lớn. (Trong phân phối D_fixed kém hơn rõ: 0,126 so với 0,069.)
<!-- prev:8c2bfdaa03d7542d391b45ac47cdf29db7e0a5f27e8a3c2fa94da7db6532451d hash:cee31e2830af0e9260fc62df3a72df666e0c0e71bf7ad72b5094221766361ef8 -->

## [L008] H5 — D_ptr (bộ định tuyến con trỏ) so với baseline tốt nhất trên MSE OOD
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

Tham chiếu B. r trung bình = +1,7%, KTC 95% [−18,1%; +31,0%]. Vi phạm V4 (như H1) nên CHƯA KẾT LUẬN; ngoài ra khoảng tin cậy chứa 0 và không đạt biên −10%. D_ptr: MSE test 0,0485; recall OOD 0,18.
<!-- prev:cee31e2830af0e9260fc62df3a72df666e0c0e71bf7ad72b5094221766361ef8 hash:fbb05c586e37da4880d11c0b19a630289b647948648d5e80eb6e91cdbace1f56 -->

## [L009] Cắt bỏ (ablation) mô tả — không mang nhãn ĐẠT/KHÔNG ĐẠT
- Thời điểm: 2026-10-07 17:48 UTC
- Trạng thái: **THĂM DÒ**

MSE OOD tương đối so với D (trung bình, KTC 95%): D_oracle −25,7% [−49,2; +1,5]; D_shuffle −44,1% [−59,9; −29,1] (mô hình không học được: MSE test 0,208 ≈ mức dự đoán trung bình, đúng vai trò đối chứng); D_fixed −4,5%; D_ring −4,8%; D_rand50 +10,5%; D_unshared −8,2% (nhiều tham số hơn: 66 192); D_T1 +13,6%; D_ptr −12,1%. Hầu hết khoảng tin cậy chứa 0.
Can thiệp định tuyến: hoán đổi bộ mô tả của bộ định tuyến làm MSE test của D tăng 0,069 → 0,268, nên đầu ra phụ thuộc vào tuyến được chọn trong phân phối dù cạnh bật không trùng cạnh thật. Trên OOD, cắt bỏ vòng lặp hoặc module không làm xấu thêm (độ lệch ≤ 0), tức mô hình không dùng tính toán có nghĩa trên OOD.
<!-- prev:fbb05c586e37da4880d11c0b19a630289b647948648d5e80eb6e91cdbace1f56 hash:58eb6f4ac4dbb941e8bdba1eebad2877aa482cafbe9a78ba2c63584c4fa82b2c -->

## [L010] Đính chính mục L003
- Thời điểm: 2026-10-07 17:49 UTC
- Trạng thái: **THĂM DÒ**

Mục L003 nói có thêm một lần khởi động lại sau lần ở seed 4; mình không có bằng chứng cho việc đó. Chỉ chắc chắn có một lần khởi động lại (ở seed 4). Lệnh kiểm tra-tiến-trình được chạy ở các lượt chờ sau đó nhưng không xác nhận được có kích hoạt hay không. Dữ liệu và kết luận không bị ảnh hưởng.
<!-- prev:58eb6f4ac4dbb941e8bdba1eebad2877aa482cafbe9a78ba2c63584c4fa82b2c hash:2cef14b92290948447dd0a4dad228800eb59cb03c62735363f8e1aff2325bd6e -->

## [L011] Pha thăm dò v2 (điểm 1–2) trên seed 1000
- Thời điểm: 2026-10-07 23:31 UTC
- Trạng thái: **THĂM DÒ**

Mã thêm: lịch cổng mềm→cứng (mix), nhiễu logistic, λ tăng dần (nhánh "anneal"). Chỉ seed 1000; không phải bằng chứng.
Kết quả (lr 1e-2, batch 200; mức dự đoán trung bình OOD = 0,215):
- D_oracle, 100 epoch: MSE test 0,0000; MSE OOD 0,0115. 160 epoch: OOD 0,0121. => Điểm 1 (huấn luyện lâu hơn) làm phép thử OOD hợp lệ trên seed này (V4 kỳ vọng thỏa).
- D cổng cứng, 100 epoch: test 0,1031; OOD 0,3213; recall OOD 0,10; entropy cổng 0,08.
- D anneal: test 0,0630; OOD 0,6342; recall 0,00; entropy 0,09; 8,0 cạnh/mẫu.
- D_ptr cổng cứng: test 0,0396; OOD 0,4755; recall 0,20.
- D_ptr anneal: test 0,0605; OOD 0,5621; recall 0,15.
Kết luận thăm dò: điểm 1 có tác dụng; điểm 2 (cổng mềm→cứng + nhiễu + λ tăng) chỉ cải thiện MSE trong phân phối của D, không cải thiện khôi phục tuyến, làm tệ OOD, và các nhánh đều dùng gần đủ ngân sách K×T = 8 cạnh/mẫu (cạnh thừa nhiều). Một seed, nên các chênh lệch nhỏ không đáng tin.
<!-- prev:2cef14b92290948447dd0a4dad228800eb59cb03c62735363f8e1aff2325bd6e hash:91a41aef31e1a2f19b123d0b4d92f0d5339d3c5025a56c1ffc44cde2d9b94b28 -->

## [L012] Khóa tiền đăng ký PREREG_v2
- Thời điểm: 2026-10-07 23:31 UTC
- Trạng thái: **THĂM DÒ**

Tệp: prereg/PREREG_v2.md, SHA-256 = c58c37978ddc6988cc90e775fdf0c3d105e14ac319fef2c7eb601ec8cdba0f73.
Khóa: 100 epoch; seed 8–19; các nhánh D, D_anneal, D_ptr, D_ptr_anneal, A, B, C, D_oracle, D_fixed; giả thuyết H1, H1b, H2, H3, H4, H5, H5b; V3 nới còn 2,0× MAC của A; khoảng tin cậy 98,75% cho 4 kiểm định OOD. Kỳ vọng ghi trước: không nhánh nào đạt H3, H1/H5 nhiều khả năng KHÔNG ĐẠT.
Chưa chạy lượt chính thức v2 tại thời điểm khóa (không có kết quả seed 8–19 nào được xem).
<!-- prev:91a41aef31e1a2f19b123d0b4d92f0d5339d3c5025a56c1ffc44cde2d9b94b28 hash:2cd51c012d547473e643da1a8508e99b43b74c0145a3e9c69b46fdfd03e2720d -->

## [L013] Bắt đầu lượt chính thức v2 (run_002_main)
- Thời điểm: 2026-10-07 23:53 UTC
- Trạng thái: **THĂM DÒ**

Khởi chạy src/run_v2.py theo PREREG_v2 (seed 8–19, 100 epoch, 9 cấu hình), có vòng giám sát tự chạy lại nếu tiến trình bị dừng; script bỏ qua tệp đã có. Tại thời điểm ghi mục này chưa có kết quả nào.
<!-- prev:2cd51c012d547473e643da1a8508e99b43b74c0145a3e9c69b46fdfd03e2720d hash:06884b1c50281d3e7666f6b39ca7756c5e434cfc4b764456e385455acdc04dad -->

## [L014] Lượt chính thức v2 (run_002_main) hoàn tất
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **THĂM DÒ**

12 seed (8–19) × 9 cấu hình = 108 tệp trong runs/run_002_main/ (100 epoch, lr 1e-2, batch 200). Phân tích: src/analyze_v2.py theo PREREG_v2; đầu ra results/results_v2.json.
Gián đoạn: tiến trình nền bị dừng nhiều lần giữa các lượt làm việc và được khởi động lại thủ công; script bỏ qua tệp đã có, không đổi cấu hình, không ghi đè. (Vòng giám sát tự chạy lại không hoạt động như dự định vì cách kiểm tra tiến trình sai; chỉ khởi động lại thủ công mới có tác dụng.)
Điều kiện hợp lệ: V1, V1_ptr, V2, V3 (mọi nhánh), V5 thỏa; V4 KHÔNG thỏa: MSE OOD trung bình của D_oracle = 0,169 (độ lệch chuẩn 0,123), không < 0,5 × 0,203 = 0,102.
<!-- prev:06884b1c50281d3e7666f6b39ca7756c5e434cfc4b764456e385455acdc04dad hash:e000d303ad519fb6f3c275f41a39b944ba1631923d3b1565cd42454ebb6b827f -->

## [L015] H1 — D so với baseline tốt nhất (A) trên MSE OOD
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r trung bình = −4,0%, KTC 98,75% [−18,0%; +10,6%]. Vi phạm V4. Dù không xét V4, khoảng tin cậy chứa 0 và vượt biên −10% nên cũng không thể ĐẠT.
<!-- prev:e000d303ad519fb6f3c275f41a39b944ba1631923d3b1565cd42454ebb6b827f hash:31d3fc07329d1e7c55b735123b838beebc0d319cc9807684ac1cdc69a0ff0644 -->

## [L016] H1b — D_anneal so với A trên MSE OOD
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r = +57,7%, KTC 98,75% [+5,3%; +128,1%]. Vi phạm V4. Nếu V4 thỏa thì kết quả sẽ là KHÔNG ĐẠT (cận dưới > −10%): D_anneal tệ hơn baseline trên OOD.
<!-- prev:31d3fc07329d1e7c55b735123b838beebc0d319cc9807684ac1cdc69a0ff0644 hash:54775e0829e6d790dadb6e5e20dffb14a86b86ec59a0814ba6f513b078dd4cb1 -->

## [L017] H2 — hiệu quả tính toán (D và D_anneal so với A, trong phân phối)
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Baseline: A (MSE 0,0077). D: r = +891% [+773%; +1010%], MAC gấp 1,41 lần A. D_anneal: r = +839% [+782%; +901%], MAC gấp 1,63 lần A. Cả hai nhãn đều KHÔNG ĐẠT (cận dưới > +10% và MAC > 0,5 × MAC của A).
<!-- prev:54775e0829e6d790dadb6e5e20dffb14a86b86ec59a0814ba6f513b078dd4cb1 hash:ad0d5d5a96a4edc93e5189e517835927dceeb62f1e29152890f492fcf1fec51c -->

## [L018] H3 — độ trung thực định tuyến trên tuyến OOD (bốn nhánh)
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Precision/recall (trung bình 12 seed): D 0,012/0,021; D_anneal 0,037/0,075; D_ptr 0,172/0,283; D_ptr_anneal 0,106/0,213; đều xa ngưỡng 0,8. Cả bốn nhánh KHÔNG ĐẠT. Tỷ lệ cạnh thừa 79–96%. Nhánh anneal dùng đủ 8 cạnh/mẫu (K×T).
<!-- prev:ad0d5d5a96a4edc93e5189e517835927dceeb62f1e29152890f492fcf1fec51c hash:65983205d65483b54d0b21ecc79d0579c3fec80a118e62e961d3824f4c7b87c3 -->

## [L019] H4 — D so với D_fixed trên MSE OOD
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +15,6%, KTC 95% [−1,7%; +31,4%] (cận dưới > −10%). MSE OOD: D 0,326, D_fixed 0,302: định tuyến phụ thuộc đầu vào không mang lại lợi ích OOD. Ngược lại trong phân phối D tốt hơn D_fixed rõ (0,077 so với 0,123).
<!-- prev:65983205d65483b54d0b21ecc79d0579c3fec80a118e62e961d3824f4c7b87c3 hash:fc969666fa8eef6d76295daef497b0a9ea092d1f38f40c41479fa90c45f40c7a -->

## [L020] H5 — D_ptr so với A trên MSE OOD
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r = −9,5%, KTC 98,75% [−20,4%; +2,0%]. Vi phạm V4; ngoài ra cận trên > 0 nên không thể ĐẠT.
<!-- prev:fc969666fa8eef6d76295daef497b0a9ea092d1f38f40c41479fa90c45f40c7a hash:9c014c0142441044fe1d3d1b4bd0628b6977eb081e9d53d3d3890f1a80ebc58b -->

## [L021] H5b — D_ptr_anneal so với A trên MSE OOD
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r = +36,9%, KTC 98,75% [+8,5%; +66,4%]. Vi phạm V4. Nếu V4 thỏa thì kết quả sẽ là KHÔNG ĐẠT.
<!-- prev:9c014c0142441044fe1d3d1b4bd0628b6977eb081e9d53d3d3890f1a80ebc58b hash:24edbd96f762ef8e59646da130f64025f1da53fc1c7d0ba794a2857551eadf6a -->

## [L022] Đối chiếu kỳ vọng ghi trước ở PREREG_v2 §3
- Thời điểm: 2026-10-08 06:36 UTC
- Trạng thái: **THĂM DÒ**

Kỳ vọng ghi trước: không nhánh nào đạt H3 — ĐÚNG. H1/H5 nhiều khả năng KHÔNG ĐẠT nếu V4 thỏa — V4 không thỏa nên là CHƯA KẾT LUẬN; H1b/H5b sẽ là KHÔNG ĐẠT nếu V4 thỏa. Tác dụng của 100 epoch lên D_oracle ở seed 1000 (OOD 0,0115) không lặp lại ở 12 seed mới (trung bình 0,169, phân tán lớn): thăm dò một seed đã quá lạc quan.
<!-- prev:24edbd96f762ef8e59646da130f64025f1da53fc1c7d0ba794a2857551eadf6a hash:fc3265cfb47022563cdce80f4e9e26c4245a3825e4624a3d0cad51a56537be4b -->

## [L023] Tổng quan tài liệu và ánh xạ điểm gãy
- Thời điểm: 2026-10-08 23:23 UTC
- Trạng thái: **THĂM DÒ**

Tìm và đọc tóm tắt/đoạn trích (chưa đọc toàn văn): Rosenbaum et al. 2019 (Routing Networks: học đồng thời module và bố cục khó); Cases et al. 2019 (Recursive Routing Networks); Mittal, Bengio, Lajoie 2022 (Is a Modular Architecture Enough?: sụp đổ và chuyên biệt hóa; lợi ích phụ thuộc số quy tắc; hệ module học end-to-end kém tiềm năng); Bahdanau et al. 2019 (tổng quát hóa có hệ thống nhạy với bố cục; bố cục học end-to-end thường sai); Muqeeth et al. 2023 (SMEAR: định tuyến học rời rạc thường thua mô hình dày cùng tham số; gradient ước lượng là nghi phạm); Liu et al. 2023, SparseMixer/ReinMax (ST chỉ là xấp xỉ bậc nhất; bộ định tuyến thiếu gradient từ chuyên gia không kích hoạt); Goyal et al. 2021 (RIMs: attention thưa, cạnh tranh cải thiện chuyên biệt hóa).
Ánh xạ sang số liệu của chúng tôi: entropy cổng ~0,02 (cổng chốt sớm); recall cạnh 2–28%; D thua A ~10 lần trong phân phối; bộ định tuyến chỉ thấy bộ mô tả, không thấy trạng thái module. Đây là giả thuyết cần kiểm định (v3), chưa phải kết luận.
<!-- prev:fc3265cfb47022563cdce80f4e9e26c4245a3825e4624a3d0cad51a56537be4b hash:aa7aed56a6e4e4b2015a707d26fb5785e968697e86ef9324a61c5cd7f99ba7a4 -->

## [L024] Phát hiện v2: D_oracle hai pha; hiệu chỉnh lr
- Thời điểm: 2026-10-08 23:23 UTC
- Trạng thái: **THĂM DÒ**

Phân tích lại 12 seed v2: D_oracle có hai pha — 4/12 seed hội tụ về loss ~0 và MSE OOD 0,008–0,015; 8/12 seed kẹt ở loss ~0,003 và MSE OOD 0,11–0,31. Nên V4 hỏng do tối ưu hóa của mô hình chuẩn, không phải thiếu năng lực.
Hiệu chỉnh trên seed 1000–1003 (100 epoch, lr khác nhau; không dùng seed đánh giá): lr 3e-3: 0/4 seed hội tụ (MSE OOD 0,34–0,45); lr 1e-3: 0/3 seed hội tụ (0,35–0,46; tiến trình bị dừng trước khi xong seed 1003); cấu hình 3e-3 với 200 epoch chưa chạy do tiến trình bị dừng. Kết luận: giữ lr 1e-2 như v2, không dò thêm; xử lý bằng sàng lọc seed theo oracle (v3 §3).
<!-- prev:aa7aed56a6e4e4b2015a707d26fb5785e968697e86ef9324a61c5cd7f99ba7a4 hash:c4884a15eee11a23aaa417dd43fda52a479e5990679181493ac8386c2937d4d4 -->

## [L025] Khóa tiền đăng ký PREREG_v3
- Thời điểm: 2026-10-08 23:23 UTC
- Trạng thái: **THĂM DÒ**

Tệp prereg/PREREG_v3.md, SHA-256 = 3b5fb23eb312df41ae52706c90f04211cc26cd4e21e46a959810d02bf6d1f44b.
Khóa: sàng lọc seed bằng D_oracle (seed 20–49, cần ≥ 8, lấy 10 hợp lệ); các nhánh A, B, C, E (MoE), D_attn, D_sup, D_soft; giả thuyết H1, H2, H3a, H3b, H4, H5, H6 (hai nhánh), H7; KTC 97,5% cho hai kiểm định OOD chính. Kỳ vọng ghi trước ở §6. Chưa có kết quả seed 20–49 nào được xem tại thời điểm khóa.
<!-- prev:c4884a15eee11a23aaa417dd43fda52a479e5990679181493ac8386c2937d4d4 hash:9229136c95e6851f5547efab585aea4ed021956e7d3fbed8f7b1b27053ce172a -->

## [L026] Lượt chính thức v3 (run_003_main) hoàn tất
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **THĂM DÒ**

Sàng lọc: 26 seed (20–45) → 10 seed hợp lệ (25, 27, 29, 31, 35, 36, 40, 42, 43, 45) theo D_oracle; đủ ≥ 8 nên kết luận OOD được phép. 7 nhánh × 10 seed = 70 tệp + 26 tệp oracle trong runs/run_003_main/. Phân tích: src/analyze_v3.py theo PREREG_v3 → results/results_v3.json.
Điều kiện hợp lệ V0, V1, V2, V3, V5 thỏa. Gián đoạn: tiến trình nền bị dừng nhiều lần giữa các lượt; khởi động lại thủ công, script bỏ qua tệp đã có; không ghi đè. Cách kiểm tra tiến trình ban đầu khớp nhầm shell bọc ngoài nên có các khoảng chết không đáng có; không ảnh hưởng dữ liệu.
Lưu ý chọn mẫu: D_oracle chỉ được báo cáo trên các seed đã được CHỌN vì oracle hội tụ, nên số liệu D_oracle (MSE OOD 0,015) bị chọn lọc theo thiết kế, không phải ước lượng không chệch.
<!-- prev:9229136c95e6851f5547efab585aea4ed021956e7d3fbed8f7b1b27053ce172a hash:71555914108efc4e7d166d5f20804c73a89f2c8557f5eab26f9ac6266866cd0d -->

## [L027] H1 — D_attn so với baseline tốt nhất (E) trên MSE OOD
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **ĐẠT (PASS)**

r trung bình = −43,2%, KTC 97,5% [−47,4%; −39,5%]; tham chiếu E (MSE OOD 0,328). THEO ĐÚNG QUY TẮC ĐÃ KHÓA đây là ĐẠT.
CẢNH BÁO diễn giải (không đổi nhãn): MSE OOD của D_attn là 0,185 ≈ mức dự đoán trung bình (0,215), còn các baseline dày có MSE OOD 0,33–0,40, tức tệ hơn dự đoán hằng số. MSE trong phân phối của D_attn là 0,145 (gấp ~19 lần A), nên D_attn không "ghép được tuyến mới" mà chỉ học yếu, nên không tạo lỗi ngoại suy lớn. Tiền đăng ký thiếu điều kiện "phải tốt hơn dự đoán hằng số"; đây là khiếm khuyết của thiết kế kiểm định, ghi nhận để sửa ở v4, không sửa hồi tố. Không có căn cứ để tuyên bố định tuyến theo nội dung có lợi.
<!-- prev:71555914108efc4e7d166d5f20804c73a89f2c8557f5eab26f9ac6266866cd0d hash:ae95fc984571427f04d44daa40101e195a4e483d229615bcffd7a6aff02fc1f3 -->

## [L028] H2 — D_sup (bố cục giám sát) so với baseline tốt nhất (E) trên MSE OOD
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +63,4%, KTC 97,5% [+36,1%; +97,8%]. MSE OOD của D_sup 0,526 (0,414–0,766), trên mức dự đoán hằng số, dù tuyến được khôi phục tốt (H3a).
<!-- prev:ae95fc984571427f04d44daa40101e195a4e483d229615bcffd7a6aff02fc1f3 hash:4497ce43aafe828a46d81b65f0d1fcd9cb0bb03574547c44a6f667048e8c9c95 -->

## [L029] H3a — D_sup: khôi phục cạnh trên tuyến OOD
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **ĐẠT (PASS)**

Precision 0,969; recall 0,970 (ngưỡng 0,8); độ phủ nút 0,971. Khi có giám sát bố cục, bộ định tuyến attention khôi phục được tuyến, kể cả tuyến chưa thấy.
<!-- prev:4497ce43aafe828a46d81b65f0d1fcd9cb0bb03574547c44a6f667048e8c9c95 hash:04dd840b288297c8fd04944cf5643908d9e6aa103f005c9ffe77e0fae68f4ccd -->

## [L030] H3b — D_attn: khôi phục cạnh không giám sát
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Precision 0,089; recall 0,173; tỷ lệ cạnh thừa 91%; 7,7 cạnh/mẫu (gần đủ K×T = 8). Đúng kỳ vọng ghi trước.
<!-- prev:04dd840b288297c8fd04944cf5643908d9e6aa103f005c9ffe77e0fae68f4ccd hash:e9568a1b623cadf8062b9f232e6d79b188778404bdfa63a032991dcd698fed8a -->

## [L031] H4 — D_soft so với D_attn trên MSE trong phân phối (giả thuyết gradient)
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **ĐẠT (PASS)**

r = −93,8%, KTC 95% [−94,0%; −93,7%]. MSE trong phân phối: D_soft 0,0089; D_attn 0,145. Ủng hộ mạnh giả thuyết rằng cổng cứng với ước lượng xuyên thẳng (ST) là điểm gãy khi tối ưu.
Chú ý: D_soft thực chất là mạng trộn mềm dày đặc (ở ngưỡng 0,5 gần như không cạnh nào "bật": 0,08 cạnh/mẫu, độ phủ nút 0,004), MAC gấp ~6 lần (172 832); MSE OOD 0,341 như các baseline dày. Không có bằng chứng nó định tuyến thưa hay ghép tuyến.
<!-- prev:e9568a1b623cadf8062b9f232e6d79b188778404bdfa63a032991dcd698fed8a hash:4d6895c998cb853e3ad9ef688697ecc7091475010e0fa9fd849f9ab90b8adca6 -->

## [L032] H5 — MoE (E) không thua mô hình dày (A) trong phân phối
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +28,4%, KTC 95% [+26,7%; +30,2%] (cận dưới > +10%). MSE trong phân phối: E 0,0098; A 0,0076. Phù hợp với tài liệu (định tuyến học thua mô hình dày cùng tham số), tuy chênh lệch tuyệt đối nhỏ.
<!-- prev:4d6895c998cb853e3ad9ef688697ecc7091475010e0fa9fd849f9ab90b8adca6 hash:edbefb68bd3aed2ecf931331a6993acde540fe8d36c490a8f2b62113dbd87128 -->

## [L033] H6 — sụp đổ và chuyên biệt hóa
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

D_attn: entropy sử dụng module 0,879 (không sụp đổ) nhưng độ phủ nút 0,465 → KHÔNG ĐẠT. D_sup: entropy 0,989; độ phủ nút 0,983 → ĐẠT. Nhãn ghi KHÔNG ĐẠT cho nhánh không giám sát (D_attn); nhánh D_sup riêng: ĐẠT.
<!-- prev:edbefb68bd3aed2ecf931331a6993acde540fe8d36c490a8f2b62113dbd87128 hash:b1246a4af86d3ae7127b1e01c685f22f41904b20c41e3113c7de9ade746a59b3 -->

## [L034] H7 — hiệu quả tính toán của D_attn
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

MSE trong phân phối của D_attn gấp ~19 lần A (r = +1802%, KTC [+1758%; +1846%]); MAC gấp 1,44 lần A.
<!-- prev:b1246a4af86d3ae7127b1e01c685f22f41904b20c41e3113c7de9ade746a59b3 hash:ca1fe95b8dbb0a9902563300502252860a99ca3366079b0151c2420db8c63139 -->

## [L035] Đối chiếu kỳ vọng ghi trước ở PREREG_v3 §6 và các quan sát chưa kiểm định
- Thời điểm: 2026-10-10 03:45 UTC
- Trạng thái: **THĂM DÒ**

Đúng kỳ vọng: D_attn không đạt H3b; D_sup đạt H3a; H5 KHÔNG ĐẠT. Sai kỳ vọng: H1 ra ĐẠT theo chữ nhưng không có ý nghĩa (xem L028).
Quan sát CHƯA được kiểm định có tiền đăng ký: (1) D_sup khôi phục tuyến 97% nhưng MSE trong phân phối trung bình 0,0125 (độ lệch chuẩn 0,017; thường kẹt ở mức ~0,006, tương tự vùng bằng phẳng của oracle ở seed không hợp lệ) và MSE OOD 0,526, trong khi D_oracle cùng bố cục đúng đạt 0,015 trên các seed được chọn. Gợi ý: khi bố cục đã đúng, điểm gãy còn lại nằm ở huấn luyện/biểu diễn của module (kẹt vùng bằng phẳng, học thuộc theo tuyến), không phải ở khôi phục bố cục. Chưa tách được vai trò của đầu đọc học được và cổng cứng ST. (2) Trong D_sup, hoán đổi bộ mô tả của bộ định tuyến làm MSE trong phân phối tăng 0,0125 → 0,425: đầu ra phụ thuộc mạnh vào bố cục.
<!-- prev:ca1fe95b8dbb0a9902563300502252860a99ca3366079b0151c2420db8c63139 hash:77c457bfe1c4e2671b04eae8cf24ac99fec4e0403857ea959012917fe86550d5 -->

## [L036] Đính chính tham chiếu mục
- Thời điểm: 2026-10-10 03:46 UTC
- Trạng thái: **THĂM DÒ**

Mục đối chiếu kỳ vọng (L035) viết 'xem L028' khi nhắc cảnh báo về H1; mục đúng là L027 (H1). Ngoài ra mục H6 (L033) gộp hai nhãn: D_attn KHÔNG ĐẠT, D_sup ĐẠT. Hai điểm này chỉ là sửa tham chiếu, không đổi dữ liệu hay kết luận.
<!-- prev:77c457bfe1c4e2671b04eae8cf24ac99fec4e0403857ea959012917fe86550d5 hash:ee6a57febce6e401074bb02e1a4fd9bf73e4426b32df5581ee1008945aeb321f -->

## [L037] Phân tích lại kết quả v3 theo điều kiện C0 (mô tả, không đổi nhãn v3)
- Thời điểm: 2026-10-10 11:31 UTC
- Trạng thái: **THĂM DÒ**

Áp C0 (MSE OOD trung bình ≤ 0,5 × dự đoán hằng số OOD ≈ 0,108) lên v3: D_attn có MSE OOD 0,185 → không thỏa C0; do đó H1 của v3 (ĐẠT theo chữ) sẽ là KHÔNG ĐẠT theo quy tắc mới. D_sup 0,526, D_soft 0,341 và các baseline dày (0,33–0,40) cũng không thỏa C0; chỉ D_oracle (0,015, seed được chọn) thỏa. Nhãn v3 giữ nguyên (không sửa hồi tố); đây là cơ sở của C0 trong v4.
<!-- prev:ee6a57febce6e401074bb02e1a4fd9bf73e4426b32df5581ee1008945aeb321f hash:16784da3758c21758ff4f027ac05ebf72661156bae8001b0490a52e8d8aaa0a4 -->

## [L038] Pilot v4 trên seed 1000 và kiểm tra ước lượng gradient
- Thời điểm: 2026-10-10 11:31 UTC
- Trạng thái: **THĂM DÒ**

Seed 1000 (không phải seed đánh giá). Kiểm tra triển khai trên bài toán đồ chơi (40 000 mẫu): hàm tuyến tính — ST, ReinMax, REINFORCE đều sai số gradient < 0,002; hàm bậc hai — ST sai số 0,243, ReinMax 0,005 (ReinMax chính xác bậc hai như lý thuyết).
Pilot 100 epoch (MSE trong phân phối / OOD; dự đoán hằng số OOD 0,215; recall cạnh OOD): D_cat_st 0,134/0,178 (0,23); D_cat_gs 0,128/0,332 (0,15); D_cat_rm 0,113/0,437 (0,15); D_cat_rf 0,124/0,230 (0,00); D_trueG 0,0048/0,458 (1,0); D_sup_trueo 0,149/0,170 (0,16); D_sup 0,134/0,227 (0,05).
Chỉ dùng để kiểm tra triển khai; không dò siêu tham số. Quan sát: D_sup ở seed 1000 (thế giới oracle không hội tụ) khôi phục cạnh chỉ 0,05, trái với 0,97 ở v3; D_trueG kẹt vùng bằng phẳng (0,0048/0,458) như oracle ở thế giới không hợp lệ.
<!-- prev:16784da3758c21758ff4f027ac05ebf72661156bae8001b0490a52e8d8aaa0a4 hash:f2656ad8727e2096514c00cff1db14df3ebe300828fa6e5d982566a89a544365 -->

## [L039] Khóa tiền đăng ký PREREG_v4
- Thời điểm: 2026-10-10 11:31 UTC
- Trạng thái: **THĂM DÒ**

Tệp prereg/PREREG_v4.md, SHA-256 = f2c35f14800470783c4cbdd0c1af24d2a3ca42dec9db7d36a329821218bf8188.
Khóa: điều kiện C0/C0-ID; tách đầu đọc/cổng (D_sup, D_sup_trueo, D_trueG, D_oracle); bốn ước lượng gradient (ST, Gumbel-ST, ReinMax, REINFORCE) trên định tuyến phân loại; sàng lọc seed 50–79 (cần ≥ 6, lấy 8); H1–H6; chẩn đoán D0; kỳ vọng ghi trước ở §5.
Ghi chú trung thực: tiến trình run_v4.py bị khởi động nhầm vài giây trước khi khóa; đã dừng và xóa toàn bộ đầu ra (không có kết quả nào được xem) trước khi khóa. Tại thời điểm khóa chưa có kết quả seed 50–79 nào.
<!-- prev:f2656ad8727e2096514c00cff1db14df3ebe300828fa6e5d982566a89a544365 hash:fb7c68d534b8134a2c8121fb4dcaf41891ae64ddd2f33e2f495d3e014c904488 -->

## [L040] Lượt chính thức v4 (run_004_main) hoàn tất
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **THĂM DÒ**

Sàng lọc seed 50–66 bằng D_oracle → 8 seed hợp lệ (50, 54, 58, 61, 62, 63, 65, 66; đủ ≥ 6). 8 nhánh × 8 seed + chẩn đoán D0 (6 thế giới × 3 khởi tạo mới) trong runs/run_004_main/. Phân tích: src/analyze_v4.py theo PREREG_v4 → results/results_v4.json. V0, V1, V2, V5 thỏa. Ngưỡng C0 (OOD) = 0,110; C0-ID = 0,109.
Gián đoạn: tiến trình nền bị dừng nhiều lần giữa các lượt, khởi động lại thủ công; không có tệp bị ghi đè.
<!-- prev:fb7c68d534b8134a2c8121fb4dcaf41891ae64ddd2f33e2f495d3e014c904488 hash:270f76ae561fa650b7cc364dbedb29ad2b3e4917585b6015c60596a16aced6a6 -->

## [L041] H1 — đầu đọc: D_sup_trueo so với D_sup trên MSE OOD
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = −58,6%, KTC 97,5% [−80,3%; −28,5%]: quy tắc δ thỏa, nhưng D_sup_trueo không thỏa C0 (MSE OOD 0,226 ≈ dự đoán hằng số 0,219; MSE trong phân phối 0,123). Theo PREREG_v4 §1 nhãn là KHÔNG ĐẠT. Diễn giải: nó 'tốt hơn' D_sup chỉ vì D_sup tệ hơn hằng số (0,734), không vì ghép được tuyến mới.
<!-- prev:270f76ae561fa650b7cc364dbedb29ad2b3e4917585b6015c60596a16aced6a6 hash:697af3b564677476647f28ae94f5c844361cdf7845684c165577533a251f8097 -->

## [L042] H2 — cổng: D_trueG so với D_sup trên MSE OOD
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

r = −16,3%, KTC 97,5% [−44,4%; +21,4%]; D_trueG không thỏa C0 (MSE OOD 0,490). Theo bảng diễn giải ghi trước: không cái nào trong H1/H2 ĐẠT → 'nguyên nhân ở nơi khác hoặc chưa kết luận'. Kỳ vọng ghi trước (H2 ĐẠT) SAI.
<!-- prev:697af3b564677476647f28ae94f5c844361cdf7845684c165577533a251f8097 hash:3873789454a754196d768b19afde8045a6fb0a381c8bbd11e16e164e2e7ce5c4 -->

## [L043] H3 — ước lượng gradient (so với ST, MSE trong phân phối)
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

Gumbel-ST: r = −8,0% [−13,5%; −2,3%] → CHƯA KẾT LUẬN. ReinMax: r = −10,5% [−15,2%; −3,8%], quy tắc δ thỏa nhưng không thỏa C0-ID (MSE trong phân phối 0,122 > 0,109) → KHÔNG ĐẠT. REINFORCE: r = −7,2% [−10,4%; −2,8%] → CHƯA KẾT LUẬN. Nhãn tổng hợp ghi KHÔNG ĐẠT vì nhánh tốt nhất (ReinMax) không thỏa C0-ID; ReinMax cải thiện ~10% nhất quán so với ST nhưng mọi nhánh vẫn ở MSE trong phân phối 0,12–0,14, xa mức 0,0075 của A.
<!-- prev:3873789454a754196d768b19afde8045a6fb0a381c8bbd11e16e164e2e7ce5c4 hash:5589317e11e48f18f868958ad4168654c8d9730326b3dd48ee126276c13be360 -->

## [L044] H4 — khôi phục cạnh không giám sát (precision / recall trên tuyến OOD)
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

ST 0,245/0,158; Gumbel-ST 0,307/0,138; ReinMax 0,059/0,038; REINFORCE 0,038/0,024; ngưỡng 0,8. Cả bốn nhánh KHÔNG ĐẠT. Ước lượng gradient tốt hơn không giúp khôi phục bố cục.
<!-- prev:5589317e11e48f18f868958ad4168654c8d9730326b3dd48ee126276c13be360 hash:6676f6bf35443ebf231c979cc670e0aa1adc2bda8364f421d2462e61b279f638 -->

## [L045] H5 — ghép tuyến mới không giám sát (so với A, MSE OOD)
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

ST r = −46,1% [−52,4%; −38,9%]; Gumbel-ST −40,3% [−46,2%; −33,8%]; ReinMax −37,6% [−46,2%; −25,7%]: ba nhánh thỏa quy tắc δ nhưng đều không thỏa C0 (MSE OOD 0,19–0,22 ≈ hằng số 0,219) → KHÔNG ĐẠT. REINFORCE: +13,0% [−34,3%; +69,0%] → CHƯA KẾT LUẬN (MSE OOD 0,406, độ lệch chuẩn lớn). Nhãn tổng hợp: KHÔNG ĐẠT. Các nhánh 'thắng' A (0,352) chỉ vì A tệ hơn hằng số.
<!-- prev:6676f6bf35443ebf231c979cc670e0aa1adc2bda8364f421d2462e61b279f638 hash:9cf7104b549e7a16189435d2b45a94298c398d901c467dada60e9f288f27e780 -->

## [L046] H6 — D_sup so với A trên MSE OOD, với C0
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +112,8%, KTC 95% [+42,7%; +200,9%]; MSE OOD D_sup 0,734 (độ lệch chuẩn 0,379), trong khi khôi phục cạnh OOD 0,985/0,978. Lặp lại kết quả v3 với seed mới: bố cục đúng ≠ ghép được tuyến mới.
<!-- prev:9cf7104b549e7a16189435d2b45a94298c398d901c467dada60e9f288f27e780 hash:2954dddbd3c7755ec1c20f337915aa379ec2026069615d63589cee4e2e37c901 -->

## [L047] Can thiệp lúc đánh giá trên D_sup (mô tả)
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **THĂM DÒ**

MSE trung bình 8 seed. Thay cổng học bằng cổng đúng: trong phân phối 0,0284 (cơ sở 0,0284), OOD 0,740 (cơ sở 0,734) → cổng học đã gần như bằng cổng đúng. Thay đầu đọc học bằng đầu đọc đúng (one-hot): trong phân phối 7,37, OOD 8,95 → phá hủy. Gợi ý (chưa kiểm định): đầu đọc học là một hỗn hợp mềm trên nhiều module và đầu ra D_sup phụ thuộc vào hỗn hợp đó, không phải đầu ra chuẩn của module cuối.
<!-- prev:2954dddbd3c7755ec1c20f337915aa379ec2026069615d63589cee4e2e37c901 hash:0238e7a3fb24076d866224f540137d499a46432d683ea0a15e62f6be0d679053 -->

## [L048] Chẩn đoán D0 — vùng bằng phẳng của D_oracle thuộc về thế giới hay khởi tạo?
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **THĂM DÒ**

D_oracle, 6 thế giới v3 × 3 khởi tạo mới (18 lần). Hội tụ: thế giới 20: 3/3; 21: 0/3; 22: 1/3; 25: 3/3; 27: 1/3; 29: 1/3 → 9/18 (50%). Thế giới 'hợp lệ' ở v3 (25, 27, 29): 5/9; thế giới 'không hợp lệ' (20, 21, 22): 4/9. Kết luận mô tả: khả năng hội tụ của D_oracle gần như không dự đoán được từ nhãn hợp lệ của v3 (khác biệt nhỏ, n nhỏ); phần lớn là may rủi theo khởi tạo, có thể thêm hiệu ứng thế giới (thế giới 21: 0/3). Hệ quả: sàng lọc seed bằng oracle chọn CHÍNH khởi tạo may mắn của oracle; số liệu D_oracle (MSE OOD 0,019) bị chọn lọc, và so sánh nhánh khác (khởi tạo khác) với oracle có thiên lệch.
<!-- prev:0238e7a3fb24076d866224f540137d499a46432d683ea0a15e62f6be0d679053 hash:d44caab27b7df299873d174c32697315462900b30f377fd58d7f4c276f91e9df -->

## [L049] Đối chiếu kỳ vọng ghi trước và diễn giải v4
- Thời điểm: 2026-10-11 23:05 UTC
- Trạng thái: **THĂM DÒ**

Kỳ vọng §5: H2 ĐẠT — SAI (UNRESOLVED); H3 không nhánh nào thỏa C0-ID — ĐÚNG; H4 không nhánh nào đạt — ĐÚNG; H5 không nhánh nào đạt — ĐÚNG; H6 KHÔNG ĐẠT — ĐÚNG.
Quan sát chưa có tiền đăng ký: D_trueG (cổng đúng + đầu đọc học) kẹt ở MSE trong phân phối 0,0045 (độ lệch chuẩn 0,0002) ở cả 8 seed, trong khi D_oracle (cổng đúng + đầu đọc đúng) hội tụ về 0,0001 trên các seed được chọn. Chỉ khác nhau ở đầu đọc, nhưng seed của D_oracle được chọn vì khởi tạo may mắn (D0: ~50% hội tụ); nếu D_trueG hội tụ độc lập 50%, xác suất kẹt 8/8 ≈ 0,4% — gợi ý mạnh nhưng không có tiền đăng ký rằng đầu đọc học (hỗn hợp mềm) cản hội tụ/chuẩn hóa. D_sup_trueo (đầu đọc đúng, cổng học) không hội tụ (recall cạnh 0,58): gợi ý gradient MSE nhiễu qua ST phá vỡ giám sát cổng khi đầu ra chưa đúng. Các gợi ý này cần thí nghiệm riêng (v5).
<!-- prev:d44caab27b7df299873d174c32697315462900b30f377fd58d7f4c276f91e9df hash:b8b18afe9c6d928b04f8430cbedec51c1aff7eb6ad251d1cbbba41b6725901c8 -->

## [L050] Pilot v5 trên seed 1000 (chỉ kiểm tra triển khai)
- Thời điểm: 2026-10-12 04:08 UTC
- Trạng thái: **THĂM DÒ**

Một khởi tạo, seed 1000 (thế giới oracle không hội tụ), 100 epoch; không dò siêu tham số. MSE huấn luyện / OOD (dự đoán hằng số OOD 0,215); precision-recall cạnh OOD: D_trueG_oh 0,0616/0,461 (1,0/1,0); D_trueG_ent 0,119/467,99 (1,0/1,0) — OOD nổ lớn; D_sup_trueo_sg 0,0043/0,563 (0,92/0,96); D_sup_sg_oh 0,144/0,232 (1,0/1,0).
Quan sát: dừng gradient MSE vào cổng giúp cổng học được với đầu đọc đúng (recall 0,96 so với 0,58 ở v4); ép one-hot hoặc phạt entropy làm tối ưu hóa đầu đọc khó hơn. Ghi chú: pilot bị chạy lặp một lần do tiến trình bị dừng và khởi động lại; không ảnh hưởng.
<!-- prev:b8b18afe9c6d928b04f8430cbedec51c1aff7eb6ad251d1cbbba41b6725901c8 hash:baf94e710ee366ac9e280d2454586af8cdeab1bcf663d6cbd6a3fb21140ef02b -->

## [L051] Khóa tiền đăng ký PREREG_v5
- Thời điểm: 2026-10-12 04:08 UTC
- Trạng thái: **THĂM DÒ**

Tệp prereg/PREREG_v5.md, SHA-256 = 542c976a765081e90599b01acfea200e905de945b178832267f2101b4ae4a997.
Khóa: khởi động lại 3 lần cho mọi nhánh (thay sàng lọc seed), 5 thế giới (seed 80–84); các nhánh A, D_oracle, D_trueG, D_trueG_oh, D_trueG_ent, D_trueG_id, D_sup_trueo, D_sup_trueo_sg, D_sup_sg_oh; H1–H7; V4' thay cho sàng lọc; kỳ vọng ghi trước ở §4. Chưa có kết quả seed 80–84 nào tại thời điểm khóa.
<!-- prev:baf94e710ee366ac9e280d2454586af8cdeab1bcf663d6cbd6a3fb21140ef02b hash:f22fa9da2035051c8ae813c96f921c3989c879434ecd82c09481301aacf54e24 -->

## [L052] Lượt chính thức v5 (run_005_main) hoàn tất
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **THĂM DÒ**

5 thế giới (seed 80–84) × 3 khởi tạo × 9 nhánh = 135 tệp trong runs/run_005_main/. Phân tích: src/analyze_v5.py theo PREREG_v5 → results/results_v5.json. Lần được chọn theo loss huấn luyện nhỏ nhất (không theo OOD). V0, V1, V4' thỏa (D_oracle tốt nhất trong 3 lần hội tụ ở 4/5 thế giới). Gián đoạn: tiến trình nền bị dừng nhiều lần giữa các lượt, khởi động lại thủ công; không có tệp nào bị ghi đè.
<!-- prev:f22fa9da2035051c8ae813c96f921c3989c879434ecd82c09481301aacf54e24 hash:933fa9c9475af9a4eb585546cfeffdca62d032db974392a09b8b965d5f1080a2 -->

## [L053] H1 — D_trueG_oh (đầu đọc ép one-hot) so với D_trueG, MSE trong phân phối
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +812%, KTC 98,33% [+523%; +1109%]. Ép one-hot bằng xuyên thẳng làm tối ưu hóa đầu đọc tệ hơn nhiều (MSE trong phân phối 0,041 so với 0,0045).
<!-- prev:933fa9c9475af9a4eb585546cfeffdca62d032db974392a09b8b965d5f1080a2 hash:d2b763297716cca359f51f54575070a5c085ecac1d9bdce353ae1de736624159 -->

## [L054] H2 — D_trueG_ent (phạt entropy đầu đọc) so với D_trueG, MSE trong phân phối
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = +48,6%, KTC 98,33% [+44,7%; +51,6%]. Phạt entropy (β = 0,05) làm tệ hơn (0,0067 so với 0,0045) và không ổn định (MSE OOD lên đến hàng trăm ở vài lần chạy).
<!-- prev:d2b763297716cca359f51f54575070a5c085ecac1d9bdce353ae1de736624159 hash:ab4164be822076cd92949c451610434c0a3f10f25baff19d241425d022fd8fc9 -->

## [L055] H3 — D_trueG_id (đầu đọc học, khởi tạo nhọn đúng) so với D_trueG, MSE trong phân phối
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **ĐẠT (PASS)**

r = −88,8%, KTC 98,33% [−99,9%; −69,0%]; thỏa C0-ID. MSE trong phân phối (tốt nhất trong 3 lần): D_trueG_id 0,0005; D_trueG 0,0045. Tỷ lệ lần chạy hội tụ (loss huấn luyện < 1e-3): D_trueG_id 5/15, D_oracle 5/15, D_trueG 0/15 (Fisher chính xác so với D_trueG: p = 0,042 cho hai nhánh đầu; mô tả, chưa hiệu chỉnh đa kiểm định, 15 lần chạy không độc lập hoàn toàn vì 3 lần/thế giới).
Theo bảng diễn giải ghi trước: H3 ĐẠT → thủ phạm là bể hấp dẫn khi khởi tạo đầu đọc (tối ưu hóa), không phải khả năng biểu diễn của đầu đọc học. Lưu ý: khởi tạo đúng đã đưa sẵn đáp án vào đầu đọc; kết quả nói rằng khởi tạo đầu đọc quan trọng, chưa nói đầu đọc học được từ khởi tạo ngẫu nhiên (H1, H2 là hai cách thử đều thất bại).
<!-- prev:ab4164be822076cd92949c451610434c0a3f10f25baff19d241425d022fd8fc9 hash:63aae62ca37b7618498fc8318cdd97e72999176d4b659c355678364a16e1c56e -->

## [L056] H4 — dừng gradient MSE vào cổng: hiệu độ phủ cạnh OOD (D_sup_trueo_sg − D_sup_trueo)
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **CHƯA KẾT LUẬN (UNRESOLVED)**

Hiệu trung bình +0,157, KTC 95% [+0,029; +0,382] (cận dưới > 0 nhưng trung bình < 0,2 theo ngưỡng đã khóa). Độ phủ cạnh OOD: có dừng gradient 0,989; không dừng 0,832 (cao hơn 0,58 của v4, có thể do chọn lần theo loss huấn luyện). Cải thiện nhỏ, chưa thỏa ngưỡng; kỳ vọng ghi trước (ĐẠT) không đạt.
<!-- prev:63aae62ca37b7618498fc8318cdd97e72999176d4b659c355678364a16e1c56e hash:9835d04fbc723d5ee02f0fb29a05c0df0ad34d0af94749db799e8ca5c6474509 -->

## [L057] H5 — ghép tuyến mới với cổng đúng (so với A, MSE OOD)
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **ĐẠT (PASS)**

H5c: D_trueG_id r = −80,4%, KTC 98,33% [−93,7%; −57,3%]; thỏa C0 (MSE OOD 0,062, hằng số 0,209) → ĐẠT. H5a: D_trueG_oh r = −22,4% [−41,8%; +3,4%], không thỏa C0 (0,265) → CHƯA KẾT LUẬN. H5b: D_trueG_ent r = +37,1% [−10,9%; +98,0%], không thỏa C0 (0,487) → CHƯA KẾT LUẬN.
Nhãn tổng hợp: ĐẠT chỉ cho H5c (D_trueG_id), với tuyến cổng ĐÚNG cho trước và đầu đọc được khởi tạo đúng; H5a, H5b CHƯA KẾT LUẬN. D_oracle (chặn trên): MSE OOD 0,055.
<!-- prev:9835d04fbc723d5ee02f0fb29a05c0df0ad34d0af94749db799e8ca5c6474509 hash:487af0a5a0931487b7c7adf93c56f749ac76dd00f80b7b8632c630f3c58ee124 -->

## [L058] H6 — D_sup_sg_oh (cổng giám sát không MSE + đầu đọc one-hot) so với A, MSE OOD
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **KHÔNG ĐẠT (FAIL)**

r = −25,3%, KTC 95% [−42,2%; −5,6%]: quy tắc δ không thỏa (trung bình > −10%? trung bình −25% thỏa, nhưng) C0 không thỏa (MSE OOD 0,256 vs ngưỡng 0,105) → KHÔNG ĐẠT. MSE trong phân phối 0,1085: đầu đọc one-hot học bằng xuyên thẳng không hội tụ.
<!-- prev:487af0a5a0931487b7c7adf93c56f749ac76dd00f80b7b8632c630f3c58ee124 hash:53e9320f4e150f38bd63bcc916a748f28529fb0dc4f3706a2a4b81e4a5f0f14c -->

## [L059] H7 — D_sup_sg_oh: khôi phục cạnh OOD
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **ĐẠT (PASS)**

Precision 0,975; recall 1,000 (ngưỡng 0,8). Khi dừng gradient MSE vào cổng, cổng học bằng giám sát khôi phục tuyến gần như hoàn hảo (kể cả tuyến OOD), nhưng đầu ra vẫn không đúng vì đầu đọc one-hot học không hội tụ.
<!-- prev:53e9320f4e150f38bd63bcc916a748f28529fb0dc4f3706a2a4b81e4a5f0f14c hash:eb2d286e039b561501778fefc19c6a01f4eb0591daab9432290d0e25fc3ff5f0 -->

## [L060] Đối chiếu kỳ vọng ghi trước và quan sát v5
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **THĂM DÒ**

Kỳ vọng §4 PREREG_v5: H4 ĐẠT — SAI (CHƯA KẾT LUẬN, cải thiện nhỏ); H1, H2 không ĐẠT — ĐÚNG; H3 không kỳ vọng mạnh — ĐẠT; H6 không ĐẠT — ĐÚNG.
Quan sát chính: (1) Vùng bằng phẳng là hiện tượng chung của thiết kế: D_oracle (cổng và đầu đọc đúng cố định) chỉ hội tụ 5/15 lần; khởi động lại 3 lần giúp D_oracle và D_trueG_id có lần tốt nhất hội tụ ở 4/5 thế giới. (2) Đầu đọc học từ khởi tạo ngẫu nhiên không bao giờ hội tụ (0/15) trong khi khởi tạo đúng hội tụ 5/15 (bằng D_oracle). (3) Cổng có giám sát học tốt khi dừng gradient MSE (recall 0,99–1,0) nhưng chưa có đầu đọc học được → chưa ghép được tuyến mới trong nhánh có cổng học.
<!-- prev:eb2d286e039b561501778fefc19c6a01f4eb0591daab9432290d0e25fc3ff5f0 hash:1c6213dfeeeb10daeb85c86900e387be1f3fd634aa0e2f879c919835a4933bf8 -->

## [L061] Đính chính mục H6 của v5
- Thời điểm: 2026-10-14 23:06 UTC
- Trạng thái: **THĂM DÒ**

Mục H6 (L058) có câu diễn đạt rối. Nói đúng: quy tắc δ THỎA (trung bình r = −25,3%, cận trên KTC −5,6% < 0, trung bình ≤ −10%) nhưng C0 KHÔNG thỏa (MSE OOD 0,256 > ngưỡng 0,105), nên theo PREREG_v5 nhãn là KHÔNG ĐẠT. Nhãn không đổi; chỉ sửa diễn đạt.
<!-- prev:1c6213dfeeeb10daeb85c86900e387be1f3fd634aa0e2f879c919835a4933bf8 hash:120ede374d759fa6f88308668db14cd8d1861283a348796effb47c32ad510aa0 -->

## [L062] Thiết kế v6 và kiểm tra mã (chưa có kết quả nghiên cứu)
- Thời điểm: 2026-10-15 11:43 UTC
- Trạng thái: **THĂM DÒ**

Yêu cầu: học đầu đọc từ khởi tạo ngẫu nhiên bằng cách khác (khởi tạo cân bằng, nhiệt độ giảm dần, ReinMax), ghép cổng giám sát có dừng gradient MSE với đầu đọc học (đầu-cuối), và giảm giám sát cổng; chạy trên Google Colab thay vì trong phiên trợ lý để tránh gián đoạn.
Mã thêm: các cấu hình D_trueG_bal/tau/rm, D_e2e_id/bal/tau/rm, D_e2e_id_g50/g25/g10 (train.py, models_v3.py); src/run_v6.py (tự tiếp tục, lưu kết quả và checkpoint .npz, tạo MANIFEST.sha256; tham số --worlds để chia phiên Colab); src/analyze_v6.py (gộp nhiều thư mục); colab/TNR_v6_colab.ipynb.
Kiểm tra mã: mỗi nhánh mới chạy được 1 epoch; run_v6 lưu checkpoint đúng; analyze_v6 chạy được trên dữ liệu GIẢ LẬP mượn từ tệp v5 (chỉ để thử đường ống, đã xóa, không có ý nghĩa khoa học). KHÔNG có pilot huấn luyện đủ nào cho các nhánh v6, nên không có kết quả hay kỳ vọng định lượng nào từ dữ liệu v6.
Lưu ý: từ v6 có checkpoint; v1–v5 không lưu trọng số (chỉ số liệu đánh giá và đường cong), nên repo không có checkpoint của v1–v5.
<!-- prev:120ede374d759fa6f88308668db14cd8d1861283a348796effb47c32ad510aa0 hash:52b345b80d9b657c0ffc5d52be110a0925662504a52d0b4d1be845084dc4278c -->

## [L063] Khóa tiền đăng ký PREREG_v6
- Thời điểm: 2026-10-15 11:43 UTC
- Trạng thái: **THĂM DÒ**

Tệp prereg/PREREG_v6.md, SHA-256 = ca2a78c9adb4533e96bfe88948c446607c49b821c6bf61c6425d090aa5736a2c.
Khóa: thế giới 90–95, 3 khởi tạo mỗi nhánh, 14 nhánh; H1a–c (học đầu đọc từ khởi tạo không chứa đáp án), H2a–d (ghép đầu-cuối so với A, cần V4'), H3 (bố cục đầu-cuối), H4a–c (giảm giám sát; không thua kém); V0 (≥ 4 thế giới), V4' (D_oracle hội tụ ở ≥ 2/3 thế giới); kỳ vọng ghi trước ở §5; mọi quy tắc C0/C0-ID/chọn lần theo loss huấn luyện như v5.
Lượt chính thức sẽ chạy trên Colab bởi người dùng; kết quả được đưa vào runs/run_006_main/ cùng MANIFEST.sha256 rồi phân tích bằng src/analyze_v6.py. Tại thời điểm khóa chưa có kết quả thế giới 90–95 nào.
<!-- prev:52b345b80d9b657c0ffc5d52be110a0925662504a52d0b4d1be845084dc4278c hash:ae0836c629c86178ebe8b01b15830de6e4699b3b8c4d614add070a70d3c89d40 -->

## [L064] Cam kết Git độc lập cho tiền đăng ký v6
- Thời điểm: 2026-10-15 11:44 UTC
- Trạng thái: **THĂM DÒ**

Đã đẩy lên https://github.com/minhtri22/tnr nhánh `from-claude` (từ `main`, tại 131874d): commit 2c3664d (gói v1–v5, Lineage L001–L061) rồi commit a2b61e2 (PREREG_v6.md SHA-256 ca2a78c9adb4533e96bfe88948c446607c49b821c6bf61c6425d090aa5736a2c, mã v6, notebook Colab, Lineage L062–L063). Commit a2b61e2 có trước mọi lần chạy v6, nên v6 là lượt đầu tiên có cam kết Git độc lập chứng minh tiền đăng ký được khóa trước khi chạy.
Không sửa `lineage.md` gốc và các tài liệu có sẵn của repo; gói nằm trong thư mục `from-claude/`. v1–v5 không có checkpoint (không lưu trọng số); v6 sẽ có.
<!-- prev:ae0836c629c86178ebe8b01b15830de6e4699b3b8c4d614add070a70d3c89d40 hash:3076d9ab9b647e4e034e6e3a84d892a0807c8ac1d2f2ff12597093cfb9446e65 -->

