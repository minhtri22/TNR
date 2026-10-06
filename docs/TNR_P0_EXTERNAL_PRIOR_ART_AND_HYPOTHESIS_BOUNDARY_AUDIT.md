# TNR P0 — Rà soát công trình có trước và ranh giới giả thuyết

**Định danh bước:** `TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT`

**Ngày khóa:** 2026-10-07

**Loại bước:** rà soát khoa học bên ngoài (external scientific audit), không huấn luyện mô hình, không mở kết quả thí nghiệm.

## 1. Câu hỏi của P0

P0 trả lời câu hỏi:

> Sau khi đối chiếu các họ công trình đã có về tính toán thích nghi, mạng mô-đun và định tuyến động, TNR còn một khoảng trống nghiên cứu (research gap) đủ rõ để đăng ký trước một thí nghiệm cơ chế hay không?

P0 **không** nhằm chứng minh:

- TNR là kiến trúc hoàn toàn mới;
- K8 là tôpô tối ưu;
- genus 3 có cơ chế học máy trực tiếp;
- có tính mới pháp lý hoặc khả năng cấp bằng sáng chế;
- có lợi ích trên mô hình ngôn ngữ lớn (LLM).

## 2. Phạm vi rà soát

Rà soát tập trung vào các họ gần nhất với giả thuyết TNR:

1. mạng định tuyến (routing networks);
2. mạng mô-đun nơ-ron (neural module networks);
3. tính toán có điều kiện (conditional computation);
4. hỗn hợp chuyên gia (mixture-of-experts);
5. thời gian/độ sâu tính toán thích nghi (adaptive computation time/depth);
6. mô-đun hồi quy và cơ chế độc lập (recurrent modular mechanisms);
7. mạng nơ-ron động (dynamic neural networks);
8. định tuyến mô-đun cho khái quát hóa hợp thành (modular routing for compositional generalization).

Đây là rà soát ranh giới có mục tiêu, không phải tổng quan hệ thống toàn bộ văn liệu (systematic literature review).

## 3. Các công trình gần nhất

### 3.1. Routing Networks — Rosenbaum, Klinger, Riemer

Công trình **Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning** đã mô tả một bộ định tuyến chọn các khối hàm và áp dụng chúng đệ quy theo từng đầu vào.

Điều này trực tiếp bác bỏ bất kỳ tuyên bố nào rằng ý tưởng chung:

> “bộ định tuyến chọn mô-đun khác nhau cho từng đầu vào và tái sử dụng mô-đun qua nhiều bước”

là mới.

Nguồn: https://arxiv.org/abs/1711.01239

### 3.2. Neural Module Networks — Andreas và cộng sự

Mạng mô-đun nơ-ron (Neural Module Networks) đã ghép các mô-đun học được thành mạng phụ thuộc cấu trúc câu hỏi, đặc biệt cho bài toán suy luận hợp thành.

Do đó việc:

> “dùng các mô-đun tái sử dụng để thực hiện các chuỗi tính toán khác nhau”

cũng đã có tiền lệ rõ ràng.

Nguồn: https://arxiv.org/abs/1511.02799

### 3.3. Adaptive Computation Time — Graves

Thời gian tính toán thích nghi (Adaptive Computation Time) cho phép mạng hồi quy học số bước tính toán khác nhau tùy đầu vào.

Do đó việc thay độ sâu cố định bằng số vòng lặp thích nghi không phải một ý tưởng mới riêng của TNR.

Nguồn: https://arxiv.org/abs/1603.08983

### 3.4. Recurrent Independent Mechanisms — Goyal và cộng sự

Cơ chế độc lập hồi quy (Recurrent Independent Mechanisms) sử dụng nhiều nhóm trạng thái có động lực riêng, chỉ cập nhật khi liên quan và giao tiếp thưa qua cơ chế chú ý.

Điều này đã bao phủ đáng kể trực giác:

- mô-đun chuyên môn hóa;
- chỉ kích hoạt một phần hệ;
- giao tiếp chọn lọc;
- cải thiện khả năng khái quát khi các yếu tố thay đổi.

Nguồn: https://arxiv.org/abs/1909.10893

### 3.5. Dynamic Routing Networks — Cai và cộng sự

Mạng định tuyến động (Dynamic Routing Networks) chọn các nhánh biến đổi theo từng mẫu đầu vào để giảm chi phí suy luận.

Điều này cho thấy định tuyến theo từng trường hợp (instance-aware routing) và áp lực chi phí tính toán đã là một dòng nghiên cứu rõ ràng.

Nguồn: https://arxiv.org/abs/1905.04849

### 3.6. Dynamic Neural Networks: A Survey — Han và cộng sự

Khảo sát mạng nơ-ron động (Dynamic Neural Networks) phân loại các mô hình có đồ thị tính toán hoặc tham số thay đổi theo đầu vào, không gian hoặc thời gian.

Vì vậy TNR không được tự định vị bằng câu “đồ thị tính toán thay đổi theo đầu vào” như một điểm mới.

Nguồn: https://arxiv.org/abs/2102.04906

### 3.7. Block-Operations — Dietz và Klakow

Công trình **Block-Operations: Using Modular Routing to Improve Compositional Generalization** trực tiếp kiểm tra giả thuyết rằng khó khăn khái quát hóa hợp thành có liên quan tới định tuyến mô-đun, và báo cáo lợi ích trên nhiệm vụ tổng hợp lẫn thực tế.

Đây là công trình gần nhất với hướng P1 của TNR và bắt buộc phải được xem là đường cơ sở khái niệm (conceptual baseline).

Nguồn: https://arxiv.org/abs/2408.00508

### 3.8. Các hướng mới hơn

Các nghiên cứu gần đây tiếp tục xem mô-đun tái sử dụng và cơ chế định tuyến là yếu tố của khái quát hóa hợp thành, kể cả trong suy luận của mô hình ngôn ngữ.

Ví dụ:

- **From Reasoning Traces to Reusable Modules**: https://arxiv.org/abs/2606.18089
- **Universal Transformers Need Memory**: https://arxiv.org/abs/2604.21999

Các kết quả này làm yếu thêm mọi tuyên bố rộng rằng “mô-đun + định tuyến + lặp” tự thân là mới.

## 4. Những tuyên bố bị loại sau P0

Sau rà soát, TNR **không được phép** dùng các tuyên bố sau làm đóng góp mới:

1. “Mỗi đầu vào có thể đi qua các mô-đun khác nhau.”
2. “Mạng có thể chọn đường tính toán động.”
3. “Các mô-đun có thể chuyên môn hóa.”
4. “Cùng một mô-đun có thể được tái sử dụng qua nhiều vòng.”
5. “Tính toán động có thể giảm chi phí.”
6. “Định tuyến mô-đun có thể hỗ trợ khái quát hóa hợp thành.”
7. “Một ngân hàng mô-đun dày kết nối là kiến trúc mới chỉ vì nó có dạng K8.”

Các ý trên đều có tiền lệ đủ gần để không được dùng như tuyên bố tính mới độc lập.

## 5. Khoảng trống còn hợp lệ

P0 không tìm thấy trong tập công trình gần nhất một phép thử được cấu trúc **đúng theo tổ hợp ràng buộc sau**:

1. nhiệm vụ tổng hợp có **đồ thị phụ thuộc tính toán thay đổi theo từng mẫu** và có đường phụ thuộc chuẩn biết trước;
2. cùng một ngân hàng mô-đun được so sánh giữa:
   - thứ tự thực thi cố định;
   - kết nối dày nhưng không định tuyến thưa;
   - định tuyến thích nghi thưa;
3. đường cơ sở chính và mô hình định tuyến được cân bằng đồng thời về:
   - tổng số tham số trong phép so sánh chính;
   - số lần gọi mô-đun hoạt động trên mỗi mẫu;
4. tập kiểm tra ngoài phân phối (OOD) giữ lại **các chuyển tiếp hợp thành chưa từng thấy**, trong khi mọi phép biến đổi nguyên tử đều đã xuất hiện khi huấn luyện;
5. đánh giá không chỉ độ chính xác nhiệm vụ mà còn có:
   - độ khớp đường định tuyến với cấu trúc tiềm ẩn;
   - can thiệp nhân quả lên định tuyến;
   - phép tách ảnh hưởng của thứ tự cố định so với lựa chọn mô-đun;
6. tiêu chí PASS / FAIL / UNRESOLVED được khóa trước khi triển khai.

Khoảng trống này là **khoảng trống thực nghiệm về cơ chế (empirical mechanism gap)**.

Nó không đồng nghĩa với một kiến trúc nền tảng hoàn toàn mới.

## 6. Ranh giới giả thuyết được phép chuyển sang P1

Giả thuyết được phép là:

> Trên một họ nhiệm vụ có cấu trúc phụ thuộc tính toán thay đổi theo từng mẫu, một ngân hàng mô-đun tái sử dụng với định tuyến thích nghi thưa có thể khái quát sang các tổ hợp chuyển tiếp chưa thấy tốt hơn một ngân hàng mô-đun cùng tổng tham số và cùng số lần gọi mô-đun nhưng thực thi theo thứ tự cố định.

Dạng ký hiệu:

**Adaptive routed modular bank > fixed-order matched modular bank**

Dấu “>” phải được định nghĩa bằng thước đo chính trong đăng ký trước (preregistration), không được đổi sau khi xem kết quả.

## 7. Những gì P1 chưa được phép tuyên bố

Ngay cả nếu P1 PASS, chưa được suy ra:

- K8 tốt hơn mọi tôpô khác;
- topology + iterations tốt hơn Transformer hoặc GNN nói chung;
- cơ chế có tác dụng trên ngôn ngữ tự nhiên;
- cơ chế tạo ra suy luận (reasoning) nói chung;
- genus 3 là nguyên nhân;
- lợi ích mở rộng sang ARN/CQG/SIX/ArcLLM;
- tính mới học thuật tổng quát đã được chứng minh.

## 8. Phán quyết P0

**PASS — VALID_MECHANISM_GAP / NOVELTY_UNCLAIMED**

Giải thích:

P0 xác nhận còn một **khoảng trống cơ chế có thể kiểm chứng** đủ hẹp để mở đăng ký trước P1. PASS ở đây chỉ có nghĩa:

> đủ cơ sở để chạy một thí nghiệm prospective có ranh giới rõ.

PASS **không** có nghĩa TNR đã chứng minh một kiến trúc mới hoặc một đóng góp khoa học mới.

## 9. Chuyển bước được phép

Bước hợp lệ tiếp theo:

**`TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION`**

Giải thích ngắn bằng Tiếng Việt:

Khóa trước thế giới đồ chơi (toy-world), cấu trúc phụ thuộc thay đổi, các đường cơ sở (baselines), cân bằng ngân sách, áp lực định tuyến, thước đo chính và tiêu chí PASS / FAIL / UNRESOLVED. Chưa được huấn luyện mô hình hoặc mở dữ liệu kết quả kiểm tra.
