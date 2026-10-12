# Báo cáo tổng hợp: Topological Neural Routing (TNR) → Agent ladder program→execute

**Nhánh:** `from-grok` · **Lab:** r/uns + Lineage.md (append-only PASS/FAIL)  
**Ngôn ngữ báo cáo:** Tiếng Việt; thuật ngữ chuyên môn giữ trong ngoặc đơn  

---

## 1. Mục tiêu nghiên cứu

1. Kiểm chứng **một giả thuyết một lần** (không nhảy cóc): inductive bias **NL → program có cấu trúc → execute** có lợi thế OOD so với residual/flat.
2. Xây **agent ladder** từ toy composition → semi-real → production hẹp.
3. Phân biệt **rãnh PASS** (composition có cấu trúc) và **rãnh FAIL** (động lực liên tục / OOD không cấu trúc) — không vá FAIL để “claim AGI”.

---

## 2. Chuỗi nghiên cứu (tóm tắt theo phase)

### 2.1 Lõi lab sớm (composition / routing)

- Nhiều variant module (PureC, KV_causal, spiral, mold…): FAIL dùng để **loại giả thuyết**, không phải điểm dừng tinh thần.
- Kết luận thiết kế: backbone homogeneous + metric traj cần **độ khó và metric preregister** trước khi so kiến trúc.
- Rãnh FAIL (robot continuous, một số OOD dynamics) **đóng băng** có chủ đích.

### 2.2 Agent ladder (4 lớp) — claim khóa

| Lớp | Nội dung | Outcome tiêu biểu |
|-----|----------|-------------------|
| 1 | NL → program → execute (paraphrase / noise) | PASS (M2/M3/P/Q) |
| 2 | Tools / HTTP mock | PASS (N3 exact, R1, R2d) |
| 3 | Multi-turn (revise + state-mid) | PASS (O3/O4) |
| 4 | Control-flow (if / branch) | PASS (CF ~0.70 vs Flat ~0.35) |

### 2.3 Real / semi-real track

| Phase | Giả thuyết | Outcome |
|-------|------------|---------|
| R1 | HTTP mock sạch | PASS |
| R2 | Fail mạng + schema + partial (metric lệch) | UNRESOLVED |
| R2d | Align fault + alias-in-train + retry | PASS (~0.45 vs Flat ~0.26) |
| ProdNL | Free-form synthetic | PASS (~0.53 vs ~0.29) |
| HumanNL curated | Open corpus / lexical cold-start | UNRESOLVED → pass-path synonym-in-train PASS |
| HumanNL-SAIL | Corpus người (SAIL) + annotate | PASS |
| Landmark mock | GOTO/FACE | PASS |
| SAIL map+trace gold | Oracle `%%` traces | PASS (length-aware) |
| SAIL full-graph | Chỉ đi trên cạnh map | PASS |
| R3 Ticket/CRM | Domain production hẹp | PASS (~0.52 vs Flat 0.00) |

### 2.4 Stress bền vững

**3-bền (SAIL)**

| Test | Ý nghĩa | Outcome |
|------|---------|---------|
| A2 facing gold | Orientation từ trace | PASS (~0.94) |
| A1 edge-dropout | Topology hẹp (proxy multi-map) | PASS (~0.97) |
| A4 landmark grounded | Bind NL→item→path trên map | PASS (~1.0) |

**2-bền (tools)**

| Test | Ý nghĩa | Outcome |
|------|---------|---------|
| B5 L=3 | Chuỗi tool dài hơn | PASS (structure OOD) |
| B1 file ops | Domain thứ 2 | PASS |
| B6 fault+retry | Lỗi API aligned | PASS |
| B4 CONFIRM→CLOSE | Constraint control trên tool | PASS |

**Ghi chú protocol:** OOD **kép** (paraphrase + structure) từng làm absolute thấp (UNRESOLVED ngưỡng 0.35); OOD **cấu trúc** (cùng lexicon) mới đo đúng bền composition.

---

## 3. Kiến trúc hiện tại (lab)

### 3.1 Pipeline thống nhất

```text
  NL (có thể multi-turn / paraphrase / free-form)
           │
           ▼
  Encoder (BiLSTM / slot heads)
           │
           ▼
  Program slots  (tool-id | move | GOTO | if-cond | length)
           │
           ▼
  Executor  (deterministic / graph-aware / fault-aware)
           │
           ▼
  World state  (grid / ticket / file / HTTP mock)
```

Đối chứng: **Flat** = NL (+ state) → MLP residual → state/cell (không program tường minh).

### 3.2 Thành phần đã chứng minh

| Thành phần | Vai trò |
|------------|---------|
| Slot / tool heads | Rời rạc hóa ý định |
| Length head | Tránh over-step khi multi-op |
| Executor graph | Tôn trọng cạnh SAIL / blocked |
| Fault mask + retry | Cùng realization lúc sinh y và eval |
| Alias-in-train | Schema OOD không cold-start hoàn toàn |
| Confirm gate | Side-effect có điều kiện |
| Control-flow slots | if blocked/clear → branch |

### 3.3 Không phải (trong lab này)

- End-to-end continuous control robot (rãnh FAIL đã đóng).
- LLM production full open-world không schema.
- Gumbel routing sâu trên mọi layer như “AGI unified” — lab tập trung **program→execute** có đo được.

---

## 4. Đề xuất ứng dụng

### 4.1 Phù hợp ngay (cùng inductive bias)

1. **Agent ngôn ngữ hẹp:** ticket CRM, file ops, lịch, FAQ→API — NL → tool program → execute có schema.
2. **Copilot nội bộ:** lệnh vận hành có confirm / retry / multi-step.
3. **Nav / kho mô phỏng:** lệnh relative + landmark trên graph đã biết.
4. **Đánh giá OOD có cấu trúc:** benchmark composition tool / control-flow trước khi scale model.

### 4.2 Khi nào chọn program→execute vs SOTA liên tục

| Chọn program→execute | Chọn continuous / large net |
|----------------------|-----------------------------|
| State rời rạc, tool/API, exact success | Cảm biến liên tục, dynamics khó formalize |
| Cần audit / replay program | Chỉ cần reward / imitation |
| OOD tổ hợp slot quan trọng | Phân phối gần IID, data cực lớn |

### 4.3 Không over-claim

Lab chứng minh **toy → semi-real → domain hẹp**.  
Chưa chứng minh: log người dùng production raw, multi-tenant live API, robot thật.

---

## 5. Hướng mở rộng

### 5.1 Gần (vẫn một giả thuyết một lần)

1. **A1 thật:** train map-grid, OOD map-jelly / map-l (không chỉ edge-dropout).
2. **Dual OOD có kiểm soát:** paraphrase × structure factorial (tách hiệu ứng).
3. **Domain production thứ 3:** calendar / email mock, exact metric.
4. **Annotate nốt SAIL residual** (~60 câu) hoặc đánh dấu out-of-scope chính thức.
5. **Abstain / confidence:** khi slot entropy cao → không execute.

### 5.2 Trung hạn

1. **Planner mỏng** trên graph khi edge blocked (replan vs fail rõ).
2. **Program DSL** phong phú hơn (loop có giới hạn, try/except tool).
3. **Transfer encoder** giữa ticket ↔ file (cùng protocol, bank khác).
4. **Human-in-the-loop annotate** needs_review với schema JSONL đã có.

### 5.3 Chuyên môn sâu (chọn một trục)

| Trục | Câu hỏi khoa học |
|------|------------------|
| **Compositionality** | Giới hạn length L và depth control-flow trước khi sập |
| **Grounding** | Landmark + partial observe + language |
| **Fault-tolerant agents** | Idempotent tools, saga, compensation |
| **Metric science** | Exact vs MSE; khi nào metric “mua” PASS ảo |
| **Theory-lite** | Program induction bias vs pure seq2seq under systematic OOD |

### 5.4 Không ưu tiên (theo lineage)

- Vá rãnh robot continuous để “đủ AGI”.
- Scale tham số không đổi protocol đo.
- Claim production user logs khi data vẫn curated/oracle.

---

## 6. Bài học phương pháp

1. **FAIL là công cụ thiết kế** nếu preregister và một giả thuyết.
2. **Metric lệch** (y có noise, exec không) → UNRESOLVED giả; sửa process trước kiến trúc.
3. **Alias / synonym phải vào train** nếu OOD lexical là mục tiêu.
4. **Length-aware execute** bắt buộc với multi-step program.
5. **Structure OOD** và **language OOD** cần tách hoặc factorial — gộp sẽ làm absolute thấp dù bias vẫn thắng Flat.

---

## 7. Trạng thái claim hiện tại

```text
✓ Agent ladder lab 4/4
✓ HTTP mock + fault-tolerant (R2d)
✓ Human SAIL + full-graph + 3-bền
✓ R3 CRM + 2-bền tools
△ Dual language+structure OOD production: còn cửa nhưng absolute khó
△ Live production API / raw human logs: chưa
✗ Continuous robot rãnh: đóng băng
```

---

## 8. Tài liệu & artifact chính

- `Lineage.md` — append-only PASS/FAIL  
- `data/human_nl/` — SAIL, schema annotate, trace gold  
- `reports/bao_cao_agent_ladder_4_lop.md`  
- Runs: `r/uns/run_* / metrics.json`  
- GitHub nhánh: `from-grok` (các commit R2d, CF, ProdNL, HumanNL-SAIL, full-graph, 3-ben, 2-ben…)

---

*Tài liệu tổng hợp phục vụ định hướng tiếp theo; không thay thế từng metrics.json từng run.*
