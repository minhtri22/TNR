# Ứng viên mục nhập cho `lineage.md` gốc (CHƯA được áp dụng)

Tài liệu này đề xuất các mục theo đúng quy tắc của `docs/TNR_CANONICAL_GOVERNANCE.md` §7 (chỉ kết quả khoa học; PASS / FAIL / UNRESOLVED; không ghi hạ tầng, sửa lỗi mã, đăng ký trước hay khóa thực thi). Chưa sửa `lineage.md` gốc; chủ repo xem xét rồi tự quyết định có nối vào hay không. Sổ chi tiết của gói này (gồm cả đăng ký trước, khóa, đính chính, gián đoạn vận hành) nằm riêng ở `from-claude/runs/Lineage.md` và theo quy ước khác (có chuỗi băm).

**Phân loại bằng chứng:** phát triển/thăm dò (development/exploratory) theo tiêu chuẩn của repo, vì lúc v1–v5 chạy chưa có chuỗi cam kết Git độc lập chứng minh đăng ký trước đã khóa trước từng lần chạy (các khóa chỉ được ghi trong sổ nội bộ). Lượt v6 là lượt đầu tiên có cam kết Git độc lập: tiền đăng ký v6 được đẩy lên nhánh `from-claude` trước khi chạy.

**Phạm vi chung:** nhiệm vụ tổng hợp nhỏ (8 phép toán, tuyến 3–4 module), tham số khớp ±5–6%, chỉ kết luận trong phạm vi này; không suy ra gì về mô hình ngôn ngữ lớn (LLM).

## v1 (`from-claude/prereg/PREREG_v1.md`) — K8 có cổng (D) so với MLP, mạng dư, K8 tĩnh
- H1 (D so với baseline tốt nhất, MSE trên tuyến chưa thấy): **UNRESOLVED** (vi phạm điều kiện hợp lệ V4: mô hình định tuyến chuẩn không ghép được tuyến mới).
- H2 (hiệu quả tính toán): **FAIL**. H3 (độ trung thực định tuyến, recall cạnh 0,02): **FAIL**. H4 (D so với cổng cố định): **UNRESOLVED**. H5 (con trỏ): **UNRESOLVED**.
## v2 (`PREREG_v2.md`) — 100 epoch, thêm cổng mềm→cứng
- H1, H1b, H5, H5b: **UNRESOLVED** (V4 không thỏa). H2 (hai nhãn), H3 (4 nhánh, recall tối đa 0,28), H4: **FAIL**.
## v3 (`PREREG_v3.md`) — định tuyến attention, giám sát bố cục, cổng mềm, MoE; sàng lọc seed bằng oracle
- H1 (D_attn so với E, OOD): **PASS theo chữ nhưng không có ý nghĩa** (MSE OOD ≈ dự đoán hằng số; xem điều kiện C0 từ v4). H2 (D_sup): **FAIL**.
- H3a (D_sup khôi phục cạnh 0,97): **PASS**. H3b (D_attn, không giám sát, recall 0,17): **FAIL**.
- H4 (cổng mềm so với cổng cứng, trong phân phối, −93,8%): **PASS**. H5 (MoE không thua mô hình dày): **FAIL** (+28%). H6: D_attn **FAIL**, D_sup **PASS**. H7: **FAIL**.
## v4 (`PREREG_v4.md`) — tách đầu đọc/cổng; điều kiện C0; ước lượng gradient
- H1 (đầu đọc đúng): **FAIL** (không thỏa C0). H2 (cổng đúng, đầu đọc học): **UNRESOLVED**. H3 (gradient ReinMax/Gumbel-ST/REINFORCE so với ST): **FAIL**. H4 (khôi phục cạnh không giám sát): **FAIL**. H5 (ghép tuyến mới không giám sát): **FAIL**. H6 (D_sup, C0): **FAIL**.
## v5 (`PREREG_v5.md`) — khởi động lại nhiều lần; vai trò đầu đọc; dừng gradient MSE vào cổng
- H1 (đầu đọc one-hot): **FAIL**. H2 (phạt entropy): **FAIL**. H3 (đầu đọc khởi tạo đúng, MSE trong phân phối −88,8%): **PASS**.
- H4 (dừng gradient MSE, hiệu recall 0,157): **UNRESOLVED**. H5c (D_trueG_id so với A, MSE OOD −80,4%, thỏa C0): **PASS**; H5a, H5b: **UNRESOLVED**.
- H6 (D_sup_sg_oh): **FAIL** (không thỏa C0). H7 (khôi phục cạnh 0,975/1,000): **PASS**.

**Kết luận khoa học tổng hợp:** chưa có bằng chứng định tuyến thích nghi tự học (không giám sát) vượt đường cơ sở dày dưới ngân sách tham số tương đương; có bằng chứng thăm dò rằng khi cổng đúng được cho trước và đầu đọc khởi tạo đúng thì ghép được tuyến mới (v5 H5c); các điểm gãy đã xác định: gradient của cổng cứng, tín hiệu học bố cục, khởi tạo đầu đọc và vùng bằng phẳng của tối ưu hóa. Giới hạn: không chứng minh lợi ích trên LLM, không chứng minh K8 tối ưu.
