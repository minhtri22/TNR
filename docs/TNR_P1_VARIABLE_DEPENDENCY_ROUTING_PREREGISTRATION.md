# TNR P1 — Đăng ký trước định tuyến phụ thuộc biến thiên

**Định danh bước:** TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION  
**Ngày khóa:** 2026-10-07  
**Trạng thái:** PREREGISTERED — chưa triển khai, chưa huấn luyện, chưa materialize TEST.

## 1. Cơ sở mở P1

P1 được mở vì hai kết quả khoa học đã có:

- TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT: **PASS — VALID_MECHANISM_GAP / NOVELTY_UNCLAIMED**.
- TNR_FROM_GROK_SCIENTIFIC_VALIDITY_RECONCILIATION: **FAIL — CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY**.

Toàn bộ nhánh from-grok được xem là bằng chứng phát triển/thăm dò (development/exploratory evidence), không phải bằng chứng xác nhận (confirmatory evidence).

## 2. Câu hỏi khoa học

Câu hỏi P1:

> Khi đường tính toán cần thiết có thể suy ra từ ngữ cảnh đầu vào, nhưng tổ hợp đường đầy đủ ở TEST chưa từng xuất hiện trong TRAIN, định tuyến mô-đun thưa thích nghi (adaptive sparse modular routing) có khái quát hợp thành ngoài phân phối tốt hơn mạng độ sâu cố định có ngân sách tương đương hay không?

Claim được phép:

Adaptive sparse routing > matched fixed-depth computation

chỉ trong thế giới đồ chơi P1-VDR1 được khóa dưới đây.

P1 không kiểm tra tính mới của K8, genus 3 như cơ chế ML, reasoning nói chung, LLM, hay khả năng áp dụng trực tiếp sang ARN/CQG/SIX/ArcLLM.

## 3. Sửa lỗi khả quan sát từ from-grok

Trong P1, đường thật bắt buộc phải quan sát được từ thông tin mô hình nhận.

Mỗi mẫu gồm:

- payload x có 16 chiều;
- bốn token ngữ cảnh định tuyến (routing context);
- mỗi token chỉ ra, qua một mã liên tục, phép biến đổi nguyên tử cần thực hiện tại bước tương ứng.

Tất cả mô hình nhận cùng lượng thông tin đầu vào.

P1 không cho phép biến trường hợp (case) chọn đường một cách ngẫu nhiên rồi giấu hoàn toàn biến đó khỏi mô hình.

## 4. Thế giới đồ chơi P1-VDR1

### 4.1. Ngân hàng biến đổi

Có 8 phép biến đổi nguyên tử T0 đến T7.

Mỗi phép biến đổi tác động trên trạng thái 16 chiều theo dạng residual:

T_i(h) = h + 0.5 * tanh(h @ W_i + b_i)

Cách sinh W_i và b_i được khóa:

- dùng NumPy PCG64 với seed biến đổi của từng world;
- mỗi phần tử W_base được lấy từ Normal(0, 0.12);
- d1 = i mod 16;
- d2 = (3*i + 1) mod 16;
- cộng 1.0 vào W_i[d1,d1];
- cộng 0.85 vào W_i[d2,d2];
- b_i ban đầu bằng 0;
- b_i[d1] = 0.25 * ((i mod 5) - 2).

Không thêm nhiễu vào nhãn.

### 4.2. Đường phụ thuộc

Mỗi route là một hoán vị dài 4, không lặp mô-đun, lấy từ 8 mô-đun.

Tổng số route là 8P4 = 1680.

Target được tạo bằng cách áp dụng bốn phép biến đổi theo đúng thứ tự route và lấy 4 chiều đầu làm y.

### 4.3. Mã ngữ cảnh

Mỗi mô-đun có một vector mã 8 chiều.

Bảng mã 8 x 8 được sinh như sau:

- dùng NumPy PCG64 với seed ngữ cảnh của world;
- lấy ma trận 8 x 8 từ Normal(0,1);
- thực hiện QR decomposition;
- chuẩn hóa dấu để đường chéo của R dương;
- dùng các hàng của Q làm mã cho 8 mô-đun.

Input đầy đủ gồm:

- payload 16 chiều;
- bốn mã ngữ cảnh, mỗi mã 8 chiều.

Tổng kích thước input là 48 chiều.

## 5. Chia TRAIN / DEV / TEST

Liệt kê toàn bộ 1680 route theo thứ tự từ điển trước khi băm.

Mỗi route được serialize chính xác thành chuỗi:

TNR_P1_VDR1|r1-r2-r3-r4

Ví dụ:

TNR_P1_VDR1|0-3-6-2

Tính SHA256 của chuỗi này và sắp route tăng dần theo chuỗi hex SHA256.

Chia cố định:

- TRAIN: 1176 route đầu;
- DEV: 168 route tiếp theo;
- TEST: 336 route cuối.

Không có route đầy đủ trùng giữa ba tập.

Đã kiểm tra trước bằng phép đếm tĩnh:

- TRAIN có đủ 8 node và đủ 56 cạnh có hướng;
- DEV có đủ 8 node và đủ 56 cạnh có hướng;
- TEST có đủ 8 node và đủ 56 cạnh có hướng.

Như vậy TEST giữ lại tổ hợp bậc cao chưa thấy, nhưng không giữ lại mô-đun nguyên tử hoặc cạnh nguyên tử.

Hash của danh sách route từng split được tính bằng cách nối các dòng theo dạng:

r1,r2,r3,r4

dùng LF giữa các dòng, không có dòng trống chen giữa.

Hash chuẩn:

- TRAIN: 54a3e16e19a46354bd77928cd2f02713f80c8166416cf87fbe5c56ec1eba91e3
- DEV: b6bb4d6be8f4f86d28d6c6cf75961a10b11f66d35d7ae37301848f3719273cf7
- TEST: 75fa374ea83fb8c185f759324d0d3f5727ea6808f68e9f358adae0910ca3bfca

## 6. Số mẫu

Cho mỗi world:

- TRAIN: 4 payload cho mỗi route = 4704 mẫu;
- DEV: 4 payload cho mỗi route = 672 mẫu;
- TEST: 4 payload cho mỗi route = 1344 mẫu.

Payload được lấy từ Normal(0, 0.7).

Thứ tự sinh payload phải đi theo thứ tự route chuẩn của từng split, bốn payload liên tiếp trên mỗi route.

## 7. Năm world xác nhận mới

P1 dùng đúng 5 seed mới:

- 314159
- 271828
- 161803
- 141421
- 173205

Các seed đã xuất hiện trong from-grok bị cấm cho kết quả xác nhận:

- 42
- 7
- 123
- 99
- 2024

Với mỗi world seed s:

- seed biến đổi = s + 1000;
- seed ngữ cảnh = s + 2000;
- seed payload = s + 3000;
- seed khởi tạo mô hình = s + 4000.

TEST không được materialize trước khóa thực thi (execution lock).

## 8. Các mô hình so sánh

### A — Đường sâu cố định (FixedMLP)

Kiến trúc khóa trước:

- input 48;
- width 72;
- 4 biến đổi ẩn tổng cộng: 48→72, sau đó 3 lớp 72→72;
- tanh sau mỗi biến đổi ẩn;
- output 72→4.

Số tham số trainable: **19.588**.

### B — Ngăn xếp residual cố định (FixedResidual)

Kiến trúc khóa trước:

- input projection 48→46;
- 4 residual blocks;
- mỗi block gồm 46→46→46 với tanh ở giữa;
- output 46→4.

Số tham số trainable: **19.738**.

Đây là đường cơ sở cố định mạnh chính.

### C — Bộ định tuyến dày mềm (DenseSoftRouter)

Kiến trúc khóa trước:

- input projection 48→32;
- 8 mô-đun;
- mỗi mô-đun 32→32→32 với tanh ở giữa;
- 4 bước tuần tự;
- controller nhận trạng thái 32 chiều + token ngữ cảnh 8 chiều;
- controller 40→24→8;
- mỗi bước tính cả 8 mô-đun;
- softmax trộn đầu ra các mô-đun;
- output 32→4.

Số tham số trainable: **19.780**.

C chỉ dùng để tách ảnh hưởng của định tuyến thưa (sparse routing) khỏi lợi ích của bộ điều khiển thích nghi.

### D — Bộ định tuyến thưa cứng TNR (SparseHardRouter)

D có cùng kiến trúc học được và cùng **19.780 tham số trainable** như C.

Khác biệt cơ chế:

- topology khả dụng là K8 có hướng;
- route không cho lặp node nên self-loop không cần dùng;
- 4 bước tuần tự;
- controller 40→24→8;
- huấn luyện dùng Gumbel-softmax straight-through hard top-1;
- đánh giá dùng argmax;
- tại TEST chỉ mô-đun được chọn mới được thực thi.

Không dùng:

- ép đường thật (teacher forcing);
- mất mát phụ route (auxiliary route loss);
- phạt entropy;
- chương trình học (curriculum);
- tuning bằng dữ liệu from-grok.

## 9. Cân bằng ngân sách

Tỷ lệ tham số đã khóa:

- P_A / P_D = 0.9903
- P_B / P_D = 0.9979
- P_C / P_D = 1.0000

Tất cả nằm trong cửa sổ ±5%.

Không được thay width sau khi P1 được khóa. Nếu phát hiện sai số đếm tham số, phải mở đăng ký trước kế nhiệm thay vì sửa hồi tố.

### 9.1. Tính toán suy luận

Bỏ qua bias và phép phi tuyến nhỏ, số phép nhân–cộng (multiply-accumulate, MAC) tĩnh xấp xỉ:

- MAC_A ≈ 19.296
- MAC_B ≈ 19.320
- MAC_D ≈ 14.464

D phải có đúng 4 lần gọi mô-đun hoạt động trên mỗi mẫu tại TEST.

Không được báo cáo độ thưa logic nếu phần đánh giá vẫn tính cả 8 mô-đun.

Điều kiện hiệu quả:

MAC_D <= 1.20 * min(MAC_A, MAC_B)

C được phép tốn compute cao hơn và không tham gia claim hiệu quả.

## 10. Áp lực định tuyến

Áp lực định tuyến (routing pressure) của D là ràng buộc kiến trúc hard top-1.

Mỗi bước phải chọn đúng một mô-đun.

Không dùng route-cost lambda vì số lần gọi đã cố định bằng 4.

Loss nhiệm vụ duy nhất là MSE.

## 11. Huấn luyện

Khóa trước cho mọi mô hình:

- optimizer: Adam;
- learning rate: 0.001;
- batch size: 128;
- tối đa 60 epoch;
- early stopping theo DEV MSE;
- patience: 10 epoch;
- cùng preprocessing;
- cùng world seed tương ứng;
- không điều chỉnh hyperparameter theo TEST.

Mỗi world huấn luyện A, B, C và D từ đầu.

## 12. Thước đo chính

MSE chuẩn hóa ngoài phân phối:

NMSE = MSE / Var(Y_TEST)

Cho mỗi world s:

R_s = NMSE_D / min(NMSE_A, NMSE_B)

Thước đo chính:

R_mean = trung bình của 5 giá trị R_s.

Một world được tính là thắng nếu:

NMSE_D < min(NMSE_A, NMSE_B)

## 13. Khôi phục route

Nhãn mô-đun học được có thể bị hoán vị.

Do đó mapping chỉ được xác định trên DEV:

1. dựng ma trận nhầm lẫn giữa token thật và mô-đun được chọn;
2. dùng ghép cặp Hungary (Hungarian assignment) để tìm hoán vị tối ưu;
3. đóng băng mapping;
4. áp dụng nguyên vẹn mapping đó lên TEST.

Thước đo:

- độ chính xác vị trí route (route position accuracy);
- độ chính xác toàn route (exact route accuracy), chỉ là thước đo phụ.

Mức ngẫu nhiên cho độ chính xác vị trí là khoảng 0.125.

## 14. Can thiệp cơ chế

Trên TEST của D, giữ nguyên input và chạy ba chế độ:

- native: controller tự chọn route;
- oracle-forced: ép module theo route thật sau mapping đã khóa từ DEV;
- wrong-forced: ép route sai bằng phép dịch vòng cố định i → (i+1) mod 8.

Đo MSE_native, MSE_oracle và MSE_wrong.

Mục đích là kiểm tra route có quan hệ nhân quả với output, không chỉ tương quan.

## 15. Điều kiện hợp lệ trước phán quyết

Một execution chỉ được adjudicate khi đồng thời:

1. ba hash route split khớp;
2. TRAIN/DEV/TEST không trùng route đầy đủ;
3. TRAIN và TEST đều có đủ 8 node và 56 cạnh có hướng;
4. đúng 5 world seed đã khóa;
5. không dùng seed from-grok;
6. số tham số A/B/C/D khớp số đã khóa hoặc sai số chỉ do cách đếm bias đã được giải trình trước TEST;
7. D có đúng 4 active module calls/sample tại TEST;
8. TEST không được dùng để chọn epoch hoặc hyperparameter;
9. mã nguồn và cấu hình đã được khóa hash trước lần materialize TEST đầu tiên.

Nếu validity gate không đạt thì execution **không có phán quyết khoa học** và không ghi PASS/FAIL vào lineage.md.

## 16. Tiêu chí PASS

P1 chỉ PASS nếu đồng thời đạt cả bốn nhóm:

### 16.1. Hiệu năng

R_mean <= 0.95

Tức D giảm ít nhất 5% NMSE so với đường cơ sở cố định tốt nhất, trung bình trên 5 world.

### 16.2. Độ nhất quán

D thắng ít nhất 4/5 world.

### 16.3. Cơ chế định tuyến

Route position accuracy trên TEST sau mapping DEV >= 0.60.

### 16.4. Can thiệp nhân quả

MSE_wrong >= 1.10 * MSE_native

và

MSE_oracle <= 1.02 * MSE_native

PASS chỉ áp dụng cho đúng P1-VDR1.

## 17. Tiêu chí FAIL

P1 FAIL nếu, sau khi mọi validity gate đạt:

- R_mean >= 1.05; hoặc
- D chỉ thắng tối đa 1/5 world.

FAIL chỉ phủ định ứng viên P1 trong phạm vi P1-VDR1, không phủ định mọi kiến trúc TNR có thể có.

## 18. Tiêu chí UNRESOLVED

Mọi trường hợp hợp lệ còn lại là **UNRESOLVED**.

Ví dụ:

- D tốt hơn 1–4% nhưng chưa đạt 5%;
- route recovery cao nhưng hiệu năng nhiệm vụ không đạt;
- hiệu năng đạt nhưng can thiệp cơ chế không đạt;
- D thắng 2/5 hoặc 3/5 world;
- kết quả hiệu năng nằm giữa ngưỡng PASS và FAIL.

Không được đổi threshold sau khi nhìn TEST.

## 19. Tường lửa với from-grok

Toàn bộ code, run và báo cáo trên from-grok được xem là đã tiêu thụ cho phát triển.

Được phép dùng để:

- phát hiện lỗi thiết kế;
- chọn đường cơ sở (baseline);
- xác định failure mode;
- thiết kế P1.

Không được dùng để:

- tính metric P1;
- điều chỉnh threshold P1;
- thay world seed sau khi xem kết quả;
- tuyên bố bằng chứng xác nhận.

## 20. Trạng thái sau đăng ký trước

P1 đã được đăng ký trước.

**Chưa được chạy TEST. Chưa được materialize TEST. Chưa có kết quả P1.**

Bước hợp lệ tiếp theo:

**TNR_P1_IMPLEMENTATION_STATIC_PREFLIGHT_AND_EXECUTION_LOCK**

Giải thích ngắn bằng Tiếng Việt:

Triển khai bộ sinh dữ liệu và bốn mô hình đúng đặc tả P1; kiểm tra bằng fixture và DEV; xác minh hash route, tham số, đường suy luận thưa, thước đo và can thiệp; sau đó khóa hash mã nguồn/cấu hình. Bước này **chưa được materialize hoặc đánh giá TEST**.
