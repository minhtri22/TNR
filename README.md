# TNR — Topological Neural Routing

> Nghiên cứu kiến trúc mạng nơ-ron theo hướng **định tuyến tôpô thích nghi** (adaptive topological routing), thay vì chỉ dựa vào các tầng tính toán xếp chồng theo chiều sâu.

## Trạng thái

**Research foundation / hypothesis formation.**

TNR hiện **chưa tuyên bố** rằng một topology cụ thể tốt hơn mạng nơ-ron phân tầng truyền thống. Repo này được tạo để biến trực giác kiến trúc thành các giả thuyết có thể kiểm chứng, so sánh với baseline công bằng và giữ lineage khoa học rõ ràng.

## Câu hỏi trung tâm

Mạng nơ-ron truyền thống thường tổ chức computation gần dạng:

~~~text
Input → L1 → L2 → L3 → ... → Ln → Output
~~~

Depth là tài nguyên tính toán chính. ResNet thêm skip connections, DenseNet tăng kết nối chéo, Transformer cho tương tác rộng trong từng layer, GNN cho message passing trên graph và Mixture-of-Experts dùng router để chọn expert.

TNR đặt câu hỏi:

> **Liệu adaptive topological routing có thể thay thế một phần fixed depth trên các bài toán có computational dependency graph thay đổi theo từng input hay không?**

Thay vì xem computation là một chuỗi layer cố định, TNR xem nó như một tập module có thể trao đổi thông tin qua một topology và được tái sử dụng qua nhiều vòng.

## Cảm hứng ban đầu

Nguồn cảm hứng là một cấu trúc đa diện 8 mặt genus 3 có đồng thời:

- cấu trúc cục bộ bên trong từng mặt;
- global adjacency rất dày giữa các mặt.

TNR **không** nhằm “uốn mạng nơ-ron thành một vật thể 3D”. Ý tưởng được trừu tượng hóa thành hai cấp:

~~~text
LOCAL COMPUTATION
M1: internal computation
M2: internal computation
...
M8: internal computation

        ↓

GLOBAL TOPOLOGY
M1 ↔ M2
M1 ↔ M3
...
M7 ↔ M8
~~~

Trường hợp thử nghiệm ban đầu có thể dùng complete module graph K8, nhưng **không giả định mọi cạnh phải hoạt động ở mọi bước**.

## Mô hình tối thiểu

Trạng thái module i tại bước t+1:

$$
h_i^{t+1}
=
f_i
\left(
h_i^t,
\sum_{j\neq i}
g_{ij}^t W_{ij}h_j^t
\right)
$$

Trong đó $g_{ij}^t$ là gate quyết định cạnh nào được kích hoạt. Gate có thể mềm:

$$
g_{ij}^t \in [0,1]
$$

hoặc cứng:

$$
g_{ij}^t \in \{0,1\}.
$$

Có thể áp dụng compute constraint:

$$
\sum_{ij} g_{ij}^t \le K
$$

để buộc router lựa chọn thay vì bật toàn bộ graph.

## Depth so với topology + iterations

Layered network thường dùng:

$$
\text{computation} \approx \text{depth}
$$

TNR nghiên cứu:

$$
\boxed{\text{topology} + \text{routing} + \text{iterations}}
$$

với cùng một tập module được tái sử dụng qua nhiều bước:

$$
H^0 \rightarrow H^1 \rightarrow \cdots \rightarrow H^T.
$$

Do đó TNR gần một recurrent computational graph có dynamic routing hơn là feed-forward stack thuần túy.

## Các giả thuyết ban đầu

### H1 — Short communication path

Trong complete module graph, bất kỳ hai module nào cũng có đường trực tiếp. Cần kiểm tra liệu short path có thực sự giảm information degradation hay chỉ làm tăng interference/compute.

### H2 — Variable computational order

Các input khác nhau có thể cần path khác nhau:

~~~text
Task A: 1 → 4 → 7
Task B: 1 → 6 → 3
Task C: 2 → 8 → 5 → 3
~~~

Đây là hypothesis trung tâm: router có học được computational trajectory thích nghi tốt hơn fixed pipeline hay không.

### H3 — Module specialization

Các module có thể học hoặc được gán vai trò như perception, memory, spatial, symbolic, causal, retrieval, planning và decision. Specialization phải được đo, không được giả định.

### H4 — Iterative computation

Topology hai chiều cho phép trajectory như:

~~~text
causal → memory → causal → decision
~~~

mà không cần mã hóa mọi bước thành layer mới.

### H5 — Structured representation và interference

TNR đặt câu hỏi liệu representation được chia thành các module/interface rõ ràng có giảm interference so với một state monolithic hay không. Đây hiện chỉ là hypothesis.

## Baseline đầu tiên

Thí nghiệm đầu tiên dự kiến so sánh dưới **parameter budget và compute budget tương đương**:

- **A — 8-layer MLP**
- **B — residual/dense modular baseline**
- **C — static K8 modular network**
- **D — gated/adaptive K8 modular network**

Claim hẹp cần kiểm chứng:

$$
\boxed{
\text{Adaptive topological routing}
>
\text{fixed depth}
}
$$

trên các task có **variable computational dependency graph**, nếu và chỉ nếu prospective experiment hỗ trợ claim đó.

## Điều TNR không tuyên bố

TNR hiện không tuyên bố:

- complete graph là topology tối ưu;
- genus 3 có ý nghĩa ML trực tiếp;
- K8 là kiến trúc mới;
- TNR tốt hơn Transformer, GNN, MoE, ResNet hoặc DenseNet;
- dynamic routing tự động tạo reasoning;
- toy-task result sẽ tổng quát sang LLM.

Transformer attention, DenseNet, GNN, MoE, recurrent computation và conditional computation đã bao phủ nhiều phần của không gian ý tưởng này. Novelty, nếu tồn tại, phải được xác định bằng **mechanism + evidence**, không bằng hình thức graph.

## Nguyên tắc nghiên cứu

1. Đăng ký trước giả thuyết (preregister hypothesis) và thước đo (metric) trước khi thực thi (execution).
2. Tách kết quả triển khai (implementation result) khỏi kết quả khoa học (scientific result).
3. Đường cơ sở (baseline) phải được cân bằng ngân sách (budget-match).
4. Ghi nhận cả PASS và FAIL.
5. Không sửa hồi tố phán quyết (retro-edit verdict).
6. Không mở rộng tuyên bố (claim) vượt ngoài nhiệm vụ/hình học (task/geometry) đã kiểm chứng.
7. Chỉ đưa cơ chế (mechanism) sang dự án khác sau khi có bằng chứng (evidence) đủ mạnh.

## Quy tắc quản trị chuẩn (canonical governance)

Quy tắc này có hiệu lực chuẩn (canonical) cho mọi trao đổi và tài liệu TNR được tạo hoặc sửa từ thời điểm ghi nhận.

1. **Trao đổi với người dùng phải hoàn toàn bằng Tiếng Việt.**
2. Khi cần dùng thuật ngữ chuyên môn Tiếng Anh, phải viết **Tiếng Việt trước, Tiếng Anh trong ngoặc đơn ngay sau đó**. Ví dụ: định tuyến động (dynamic routing), đường cơ sở (baseline), bằng chứng (evidence).
3. Không dùng một câu hoặc đoạn giải thích thuần Tiếng Anh trong phần trao đổi với người dùng.
4. Tên bước nghiên cứu, định danh kỹ thuật, tên file, tên nhánh, mã nguồn, lệnh, công thức và tên riêng kỹ thuật có thể giữ nguyên Tiếng Anh để bảo toàn tính chính xác.
5. Mỗi **bước tiếp theo** có tên Tiếng Anh phải luôn có **một giải thích ngắn bằng Tiếng Việt ngay sau tên bước**, nêu rõ bước đó làm gì và chưa được phép làm gì nếu có giới hạn khoa học.
6. Trong tài liệu nghiên cứu mới, ưu tiên Tiếng Việt là ngôn ngữ chính; thuật ngữ chuyên môn Tiếng Anh đặt trong ngoặc đơn sau Tiếng Việt khi xuất hiện trong phần diễn giải.
7. Khi có xung đột giữa cách viết cũ và quy tắc này, **quy tắc quản trị chuẩn này có ưu tiên cao hơn**. Tài liệu cũ không bị sửa hồi tố chỉ để đổi ngôn ngữ, nhưng mọi lần sửa nội dung tiếp theo phải tuân thủ quy tắc mới.

Tài liệu chuẩn chi tiết: [Quản trị chuẩn TNR](docs/TNR_CANONICAL_GOVERNANCE.md).

## Quan hệ với các dự án khác

TNR là research program độc lập.

Nếu một mechanism được chứng minh:

- **ARN:** có thể kế thừa representation topology / routing substrate.
- **CQG:** có thể kế thừa cost-aware edge activation và value-of-computation.
- **ArcLLM / NEXUS:** có thể kế thừa runtime/execution substrate.
- **SIX:** chỉ liên quan nếu về sau có evidence về representation/causal transfer.

Nguyên tắc:

$$
\boxed{\text{TNR discovers the mechanism}}
$$

$$
\boxed{\text{Downstream systems inherit a frozen, validated mechanism}}
$$

## Tài liệu

- [Nền tảng nghiên cứu TNR](docs/TNR_RESEARCH_FOUNDATION.md)
- [Quản trị chuẩn TNR](docs/TNR_CANONICAL_GOVERNANCE.md)
- [Rà soát công trình có trước P0](docs/TNR_P0_EXTERNAL_PRIOR_ART_AND_HYPOTHESIS_BOUNDARY_AUDIT.md)
- [Đối chiếu khoa học nhánh from-grok](docs/TNR_FROM_GROK_SCIENTIFIC_RECONCILIATION.md)
- [Đăng ký trước P1](docs/TNR_P1_VARIABLE_DEPENDENCY_ROUTING_PREREGISTRATION.md)
- [Dòng dõi khoa học chỉ nối thêm](lineage.md)

## Trạng thái khoa học hiện tại

- P0: **PASS — VALID_MECHANISM_GAP / NOVELTY_UNCLAIMED**.
- Kiểm toán tính hợp lệ của chuỗi `from-grok`: **FAIL — CENTRAL_CLAIM_EXPERIMENTAL_VALIDITY**. Các kết quả cũ được giữ như bằng chứng phát triển/thăm dò (development/exploratory evidence), không dùng làm bằng chứng xác nhận (confirmatory evidence).
- P1: **đã đăng ký trước (preregistered)**.
- Triển khai P1 + tiền kiểm tra tĩnh (static preflight): **PASS**.
- Khóa thực thi (execution lock): **đã tạo cục bộ và QA PASS, TEST chưa materialize**.
- Bước xác nhận P1 chưa được phép chạy cho tới khi đúng khóa này được cam kết (commit) và đẩy lên nhánh chuẩn `research/tnr-p1-vdr1`.

## Bước khoa học kế tiếp

**`TNR_P1_CONFIRMATORY_ONE_SHOT_EXECUTION_AND_ADJUDICATION`**

Giải thích ngắn bằng Tiếng Việt:

Sau khi khóa thực thi được công bố chuẩn trên GitHub, chạy đúng 5 thế giới xác nhận đã khóa một lần duy nhất, khi đó mới được materialize TEST, đóng băng bằng chứng thô (raw evidence), tính các tiêu chí đã đăng ký trước và đưa ra PASS / FAIL / UNRESOLVED. **Chưa được chạy bước này trước khi khóa Git chuẩn được xác nhận.**
