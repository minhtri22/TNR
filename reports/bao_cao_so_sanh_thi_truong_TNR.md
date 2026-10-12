# Báo cáo so sánh thị trường và định vị nghiên cứu TNR / Agent ladder

**Ngày:** 2026-10-12  
**Đối tượng:** Chuỗi lab program→execute (Topological Neural Routing / agent ladder) so với sản phẩm và nền tảng agent đang có trên thị trường  
**Ngôn ngữ:** Tiếng Việt; thuật ngữ tiếng Anh có **bảng giải thích riêng** (mục 8)

---

## 1. Tóm tắt điều hành

Nghiên cứu lab không phải sản phẩm thương mại. Nó đo **inductive bias** (thiên kiến quy nạp) cố định:

> Câu lệnh ngôn ngữ tự nhiên → **program** (chương trình/slot công cụ) tường minh → **execute** (thực thi) trên thế giới có quy tắc.

Sản phẩm thị trường 2025–2026 chủ yếu đi hướng:

- **LLM lớn + tool calling** (gọi công cụ) + orchestration (điều phối), hoặc  
- **Computer use** (điều khiển giao diện máy tính), hoặc  
- **Agent CX/CRM** đóng gói (hỗ trợ khách hàng / vé việc).

| Chiều | Lab TNR / agent ladder | Sản phẩm thị trường điển hình |
|-------|------------------------|--------------------------------|
| Mục tiêu | Chứng minh OOD có cấu trúc, PASS/FAIL khoa học | Giải quyết ticket, doanh thu, UX |
| Program tường minh | Bắt buộc, đo exact | Thường ẩn trong chuỗi suy luận LLM |
| Metric | Exact / cell-acc, preregister | Resolution rate, CSAT, $/resolution |
| Quy mô | Toy → semi-real → domain hẹp | Production multi-tenant |
| Điểm mạnh | Audit được, composition OOD | Phủ việc thật, data/API sẵn |
| Điểm yếu | Chưa live API / log user raw | Khó tách bias vs scale model |

**Định vị:** lab là **lớp phương pháp + protocol đo** có thể **cắm dưới** hoặc **kiểm chứng** agent production, không thay ChatGPT agent / Salesforce Agentforce / LangGraph.

---

## 2. Bản đồ sản phẩm thị trường (nhóm)

### 2.1 Nền tảng mô hình + agent đa năng

| Sản phẩm / hướng | Đặc trưng | Liên hệ lab |
|------------------|-----------|-------------|
| **OpenAI** Agents SDK, tool calling, ChatGPT agent / Operator | Gọi hàm, code interpreter, computer-ish workflows | Cùng ý “tool”; lab tách program/execute để đo OOD |
| **Anthropic** Claude + **Computer use** + Agent SDK | Click/GUI, MCP tools, long-running | Lab không GUI; lab mạnh schema + fault/confirm |
| **Google** Gemini Agent Mode / Chrome agent | Agent trong hệ sinh thái Google | Tương tự: phủ rộng, không public systematic OOD composition |

### 2.2 Framework lập trình agent

| Sản phẩm | Đặc trưng | Liên hệ lab |
|----------|-----------|-------------|
| **LangChain / LangGraph** | Chain, graph state, multi-provider | Lab = protocol đo; LangGraph = hạ tầng production |
| **CrewAI**, AutoGen/AG2 | Multi-agent vai trò | Lab single-pipeline program→execute |
| **OpenAI / Anthropic Agent SDK** | Vòng lặp agent vendor-lock | Lab độc lập model nhỏ (BiLSTM) để cô lập bias |

### 2.3 Agent doanh nghiệp CX / CRM / ticket

| Sản phẩm | Đặc trưng | Liên hệ lab |
|----------|-----------|-------------|
| **Salesforce Agentforce** (+ Fin/Intercom) | Agent trên CRM, $/conversation hoặc credit | Gần **R3 Ticket/CRM** lab; lab có exact state + confirm/fault |
| **Intercom Fin** | Resolution-based pricing, knowledge + procedures | Procedures ≈ program; lab đo composition procedure OOD |
| **Decagon, Sierra, Ada, Maven AGI, Fini** | Agent hỗ trợ trên Zendesk/Intercom… | Cùng lớp ứng dụng đề xuất; lab chưa connector live |
| Helpdesk **MCP** (Zendesk, Intercom…) | Tool chuẩn cho LLM | Khớp hướng tools bank lab |

### 2.4 Agent web / automation

| Sản phẩm | Đặc trưng | Liên hệ lab |
|----------|-----------|-------------|
| **MultiOn** và tương tự | Agent thao tác web | Gần computer use; lab đóng rãnh continuous/GUI |
| Browser-use agents | DOM/GUI | Ngoài scope lab hiện tại |

### 2.5 Nghiên cứu / benchmark (không phải “sản phẩm bán”)

SCAN, SAIL, VLN (R2R, RxR), gSCAN: **cùng họ** với HumanNL-SAIL lab.  
Sản phẩm thị trường **hiếm khi** công bố systematic compositional OOD như lab.

---

## 3. So sánh theo tiêu chí kỹ thuật

### 3.1 Cách biểu diễn hành động

| | Lab | Thị trường |
|--|-----|------------|
| Biểu diễn | **Slots / tool-id sequence** cố định schema | Thường **free-form plan** trong token LLM + function call JSON |
| Ưu | Replay, audit, exact match | Linh hoạt ngôn ngữ mở |
| Nhược | Schema phải thiết kế | Khó chứng minh composition systematic |

### 3.2 Tổng quát hóa (generalization)

| | Lab | Thị trường |
|--|-----|------------|
| OOD structure (tổ hợp tool/map mới) | **Đã đo** (B5, B1, A1, A4…) | Ít benchmark public cùng protocol |
| OOD language (paraphrase) | ProdNL / SAIL human | LLM lớn mạnh hơn rõ (pretrain) |
| Dual OOD | Absolute khó (đã ghi UNRESOLVED) | Thường “trông ổn” trên traffic thật, ít tách factorial |

### 3.3 Độ tin cậy vận hành

| | Lab | Thị trường |
|--|-----|------------|
| Fault / retry | R2d, B6 — **aligned metric** | Production retry/idempotent tùy vendor |
| Confirm trước side-effect | B4 PASS | Policies / human-in-the-loop trên Agentforce, Fin Tasks… |
| Control-flow | CF if/branch PASS | Procedures, Agent Operating Procedures (Decagon…) |

### 3.4 Quy mô và chi phí

| | Lab | Thị trường |
|--|-----|------------|
| Model | Nhỏ (BiLSTM), train từ scratch trên task | Frontier LLM, $/resolution hoặc seat |
| Data | Synthetic + SAIL human + mock API | Knowledge base + ticket history + connectors |
| Mục tiêu tối ưu | PASS scientific | CSAT, AHT, resolution rate, margin |

---

## 4. Bảng định vị một trang

```text
                    Audit / exact program
                           ▲
                           │
              Lab TNR      │
              agent ladder │
                           │
     Composition ──────────┼─────────── Scale & UX
     OOD measured          │            Market agents
                           │
                           │     Computer use / GUI
                           │
                           ▼
                    Continuous / open world
```

- **Góc trên-trái (lab):** đo được, hẹp, khoa học.  
- **Góc phải (thị trường):** rộng, sản phẩm, metric kinh doanh.  
- **Hợp lực:** dùng protocol lab để **regression-test** procedure composition trước khi ship agent CRM/CX.

---

## 5. Bài học từ thị trường cho lab

1. **Procedures / Agent Operating Procedures** của Fin, Decagon ≈ program slots — lab đã có nền exact + OOD structure.  
2. **MCP / OpenAPI connectors** là cầu production; lab mới mock HTTP/CRM.  
3. **Computer use** là rãnh khác (GUI); lineage đã đóng continuous/GUI có chủ đích.  
4. **Pricing per resolution** đòi hỏi abstain/confidence — lab mới đề xuất, chưa implement.  
5. Framework (LangGraph) giải **vận hành**; lab giải **giả thuyết inductive bias** — không thay nhau.

---

## 6. Bài học từ lab cho sản phẩm

1. **Metric lệch** (target có noise, exec không) tạo ảo giác fail — production cần cùng fault realization khi đánh giá.  
2. **Alias/schema** phải có trong train/distribution hoặc lớp normalize — nếu không lexical OOD sập.  
3. **Length-aware program** tránh over-execute multi-step.  
4. **Structure OOD** (hold-out tool triple) nên có trong QA agent, không chỉ paraphrase test.  
5. **Confirm gate** trước CLOSE/refund là constraint đo được, không chỉ prompt “hãy cẩn thận”.

---

## 7. Kết luận định vị

| Câu hỏi | Trả lời ngắn |
|---------|----------------|
| Lab có thay Agentforce/Fin không? | **Không.** |
| Lab có thay LangGraph không? | **Không** — lab là protocol + bias, không phải runtime đa model. |
| Giá trị khác biệt? | **Đo composition / fault / confirm** theo PASS/FAIL preregister trên program→execute. |
| Ứng dụng gần? | Lớp kiểm thử + thiết kế schema agent CRM/ticket/file trước khi gắn frontier LLM. |
| Không làm gì? | Không claim AGI; không vá robot continuous để “bằng” computer use. |

---

## 8. Thuật ngữ tiếng Anh — giải thích riêng

| Thuật ngữ | Giải thích tiếng Việt |
|-----------|----------------------|
| **Agent** | Hệ thống nhận mục tiêu/ngôn ngữ, chọn hành động, tương tác công cụ hoặc môi trường (không chỉ chatbot trả lời). |
| **Tool calling / function calling** | LLM chọn và gọi hàm/API theo schema JSON thay vì chỉ sinh chữ. |
| **Orchestration** | Điều phối nhiều bước/model/tool (chuỗi, đồ thị, multi-agent). |
| **Computer use** | Agent thao tác GUI (chuột, bàn phím, ảnh màn hình), không chỉ API. |
| **MCP (Model Context Protocol)** | Giao thức kết nối tool/nguồn dữ liệu chuẩn cho model/agent. |
| **Inductive bias** | Giả định cấu trúc đưa vào model (ở lab: phải qua program rồi mới execute). |
| **OOD (out-of-distribution)** | Dữ liệu khác phân phối train (tổ hợp tool mới, paraphrase, map khác…). |
| **Compositional generalization** | Ghép phần đã học thành tổ hợp mới chưa thấy khi train. |
| **Program→execute** | Tách “hiểu thành chương trình/slot” và “chạy chương trình trên thế giới”. |
| **Slot** | Ô rời rạc trong program (tool-id, hướng, điều kiện if…). |
| **Executor** | Bộ thực thi deterministic theo program và quy tắc thế giới. |
| **Flat / residual baseline** | Model học trực tiếp NL→kết quả, không program tường minh. |
| **Exact success / cell-acc** | Đúng tuyệt đối state hoặc ô lưới (không chỉ MSE gần đúng). |
| **Preregister** | Khóa ngưỡng PASS/FAIL trước khi chạy thí nghiệm. |
| **Fault injection** | Cố ý gắn lỗi mạng/API để thử chịu lỗi. |
| **Retry / idempotent** | Thử lại; thao tác lặp không làm hỏng state. |
| **Human-in-the-loop** | Người xác nhận bước nhạy cảm. |
| **Resolution rate** | Tỷ lệ ticket/hội thoại agent tự xử lý xong (metric CX). |
| **CSAT** | Mức hài lòng khách hàng. |
| **Multi-tenant** | Nhiều khách hàng/công ty trên một hệ thống. |
| **Schema** | Khuôn dữ liệu/tool (tên trường, tham số API). |
| **Alias** | Tên thay thế cùng một tool/trường (legacy_status ≈ status). |
| **Control-flow** | Nhánh if/while trong program (không chỉ chuỗi thẳng). |
| **Grounding** | Gắn ngôn ngữ với thực thể/vị trí trong môi trường. |
| **VLN (vision-and-language navigation)** | Điều hướng theo ngôn ngữ kèm quan sát thị giác. |
| **SAIL** | Corpus lệnh điều hướng người viết (nghiên cứu cổ điển). |
| **SCAN** | Benchmark composition lệnh điều hướng synthetic. |
| **BiLSTM** | Mạng LSTM hai chiều — encoder nhỏ dùng trong lab. |
| **Frontier LLM** | Mô hình ngôn ngữ lớn hàng đầu thương mại. |
| **Runtime** | Môi trường chạy agent thật (khác protocol thí nghiệm). |
| **Regression-test** | Bộ kiểm thử lặp lại để bắt lỗi khi đổi hệ thống. |

---

## 9. Tài liệu liên quan trong repo

- `reports/bao_cao_tong_hop_TNR_agent_ladder.md` — tổng hợp chuỗi nghiên cứu & kiến trúc  
- `reports/bao_cao_agent_ladder_4_lop.md` — claim 4 lớp  
- `Lineage.md` — PASS/FAIL append-only  
- Nhánh GitHub: `from-grok`

---

*Báo cáo so sánh phục vụ định vị khoa học–sản phẩm; số liệu thị trường mang tính bức tranh 2025–2026, không phải audit vendor.*
