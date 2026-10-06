# TNR — Quy tắc quản trị chuẩn

## 1. Mục đích

Tài liệu này xác lập quy tắc quản trị chuẩn (canonical governance) cho cách TNR được trao đổi, mô tả và chuyển bước khoa học.

Quy tắc này không thay đổi kết quả khoa học đã có. Nó điều chỉnh cách trình bày, cách gọi tên và cách giải thích từ thời điểm được ghi nhận trở đi.

---

## 2. Quy tắc ngôn ngữ khi trao đổi với người dùng

Mọi trao đổi với người dùng phải dùng **Tiếng Việt làm ngôn ngữ chính và đầy đủ**.

Khi cần dùng thuật ngữ chuyên môn Tiếng Anh, phải theo mẫu:

**Tiếng Việt (English technical term)**

Ví dụ đúng:

- định tuyến động (dynamic routing)
- đường cơ sở (baseline)
- bằng chứng (evidence)
- không gian biểu diễn (representation space)
- tính toán có điều kiện (conditional computation)
- đăng ký trước (preregistration)

Ví dụ không đúng:

- “dynamic routing sẽ tốt hơn baseline”
- “evidence chưa đủ”
- “implementation đã pass”

Cách viết đúng tương ứng:

- “định tuyến động (dynamic routing) sẽ được so sánh với đường cơ sở (baseline)”
- “bằng chứng (evidence) chưa đủ”
- “phần triển khai (implementation) đã đạt yêu cầu”

---

## 3. Ngoại lệ cho định danh kỹ thuật nguyên văn

Các đối tượng sau được phép giữ nguyên Tiếng Anh khi việc dịch có thể làm mất tính chính xác:

- tên bước nghiên cứu;
- mã định danh;
- tên file;
- tên thư mục;
- tên nhánh Git;
- tên hàm, lớp, biến;
- câu lệnh;
- công thức;
- tên mô hình;
- tên kiến trúc hoặc tên riêng đã được chuẩn hóa.

Ví dụ:

`TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT`

được phép giữ nguyên.

Tuy nhiên ngay sau đó phải có giải thích ngắn bằng Tiếng Việt, ví dụ:

> **TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT** — rà soát công trình có trước (prior art) và khóa ranh giới giả thuyết trước khi thiết kế thí nghiệm.

---

## 4. Quy tắc cho bước khoa học tiếp theo

Mỗi khi nêu **bước khoa học tiếp theo**, phải có đủ hai phần:

1. **Tên bước chuẩn** — có thể bằng Tiếng Anh để giữ định danh nhất quán.
2. **Giải thích ngắn bằng Tiếng Việt** — phải nói rõ:
   - bước này làm gì;
   - tạo ra đầu ra gì;
   - chưa được phép làm gì nếu còn khóa khoa học.

Mẫu chuẩn:

> **TÊN_BƯỚC_CHUẨN**  
> Giải thích ngắn bằng Tiếng Việt về mục tiêu, đầu ra và giới hạn của bước.

Không được chỉ trả về tên bước Tiếng Anh mà không giải thích.

---

## 5. Quy tắc cho tài liệu mới

Tài liệu nghiên cứu mới phải ưu tiên:

- Tiếng Việt cho phần diễn giải;
- thuật ngữ Tiếng Anh trong ngoặc đơn sau thuật ngữ Tiếng Việt;
- công thức, mã, định danh và tên chuẩn được giữ nguyên khi cần;
- không để các đoạn giải thích dài hoàn toàn bằng Tiếng Anh nếu không có yêu cầu rõ ràng.

---

## 6. Quy tắc cho tài liệu cũ

Tài liệu cũ không cần sửa hồi tố chỉ để thay đổi ngôn ngữ.

Tuy nhiên:

- nếu một tài liệu cũ được sửa nội dung;
- hoặc một phần mới được thêm vào;

thì phần được sửa hoặc thêm mới phải tuân thủ quy tắc quản trị chuẩn này.

---

## 7. Sổ dòng dõi khoa học (lineage.md)

TNR sử dụng file gốc `lineage.md` làm **sổ dòng dõi khoa học chỉ nối thêm (append-only scientific lineage)**.

Quy tắc chuẩn:

1. `lineage.md` chỉ ghi **kết quả khoa học** và các phán quyết khoa học đã được xác lập.
2. Phải ghi nhận cả kết quả **PASS** và **FAIL**; khi cần có thể ghi **UNRESOLVED** nếu tiêu chí khoa học đã định nghĩa trạng thái này.
3. Một mục đã ghi không được sửa, xóa, đổi nghĩa hoặc viết đè để làm đẹp lịch sử. Nếu cần hiệu chỉnh, phải **nối thêm một mục mới** giải thích hiệu chỉnh và tham chiếu mục cũ.
4. Không ghi vào `lineage.md` các sự kiện thuần kỹ thuật hoặc vận hành, bao gồm:
   - triển khai hạ tầng (infrastructure implementation);
   - lỗi và sửa lỗi mã nguồn (code bug/fix);
   - lỗi công cụ, kết nối, quyền truy cập hoặc thiết bị;
   - cài đặt môi trường, tải mô hình, build, cache;
   - thử lại do lỗi hạ tầng;
   - các bước tiền kiểm tra kỹ thuật (technical preflight) không tạo ra kết quả khoa học.
5. Việc đăng ký trước (preregistration), khóa thực thi (execution lock) hoặc chuẩn bị triển khai chỉ được ghi vào tài liệu tương ứng; **không tự động trở thành một mục kết quả trong `lineage.md`**.
6. Chỉ khi một bước tạo ra kết luận khoa học có thể phân loại theo tiêu chí đã khóa thì mới được nối thêm vào `lineage.md`.
7. `lineage.md` là nguồn chuẩn (canonical source) cho lịch sử PASS / FAIL / UNRESOLVED của chương trình TNR.

---

## 8. Ưu tiên khi có xung đột

Nếu có xung đột giữa:

- cách viết trong tài liệu cũ;
- cách viết trong trao đổi cũ;
- và tài liệu quản trị này;

thì **tài liệu quản trị này có ưu tiên cao hơn** cho mọi công việc TNR tiếp theo.

---

## 9. Áp dụng ngay cho bước khoa học hiện tại

Bước khoa học tiếp theo hiện tại là:

**TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT**

Giải thích ngắn bằng Tiếng Việt:

Rà soát công trình có trước (prior art), xác định phần nào của ý tưởng đã tồn tại, thu hẹp ranh giới giả thuyết có thể kiểm chứng và khóa các đường cơ sở (baselines) bắt buộc. Bước này **chưa được huấn luyện mô hình (train model)** và chưa được mở kết quả thí nghiệm.

Nếu P0 xác nhận còn khoảng trống nghiên cứu (research gap) hợp lệ, bước sau mới là:

**TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION**

Giải thích ngắn bằng Tiếng Việt:

Đăng ký trước (preregistration) thí nghiệm về định tuyến phụ thuộc biến thiên (variable dependency routing), khóa nhiệm vụ, cấu trúc phụ thuộc, ngân sách, thước đo và tiêu chí PASS / FAIL / UNRESOLVED trước khi triển khai.
