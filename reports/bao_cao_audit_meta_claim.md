# Audit meta-claim — Lab TNR composition families

**Ngày audit:** 2026-10-10  
**Nhánh:** `github.com/minhtri22/TNR` (`from-grok`)  
**Phạm vi:** Mọi phase đã preregister PASS/FAIL/UNRESOLVED (A→L2, G–K transfer, control/prob freezes).

---

## 1. Meta-claim ứng viên (đã dùng trong lab)

> **Mỗi họ composition cần inductive bias khớp cơ chế; không có một kiến trúc end-to-end (PureC / residual / Gumbel router) cover mọi họ.**

Audit dưới đây kiểm tra meta-claim này có **được bằng chứng ủng hộ**, **bị hạn chế**, hay **đánh trống**.

---

## 2. Bảng tổng hợp theo họ

| Họ | Bias đúng lớp (đã thắng hoặc upper bound) | Bias / setup thất bại | Outcome |
|----|-------------------------------------------|------------------------|---------|
| **Discrete-bank** | PureC, `n_mod=n_ops`, routing path | Collision `n_mod`, layout λ quá mạnh | **PASS** |
| **Algebraic** | Matrix compose trên **STATE** | Givens-MLP trên **HIDDEN** | **PASS** (E2, L=4) |
| **Symbolic productivity** | Neural base + **symbolic** `m` (twice/thrice) | Seq2seq / học m neural thuần | **PASS** (F2) |
| **Relational binding** | Symbolic rewrite slot (*opposite*) | Seq2seq OOD = 0 | **PASS** (G) + transfer *around* |
| **Hierarchical** | Stack / recursive *after* | Seq2seq OOD = 0 | **PASS** (H) + nested transfer |
| **Grounded** | Program+Execute trên **world grid** | Residual hồi quy tọa độ OOD ~0.18 | **PASS** (K) + nested/combo transfer |
| **Control** | **Oracle** path + true skills | Gumbel, TF→free, closed-loop, hybrid | **FREEZE** (oracle PASS, learned FAIL) |
| **Probabilistic** | — | Mixture-over-programs MSE/NLL | **FREEZE** (L, L2 FAIL) |

---

## 3. Kiểm định meta-claim

### 3.1 Ủng hộ mạnh

1. **Cùng “tên” composition, khác bias → khác outcome**  
   - Algebraic: matrix@STATE **PASS**, Givens@H **FAIL**.  
   - Symbolic: symbolic `m` **PASS**, seq2seq **OOD 0**.  
   → Không phải “thêm depth”, mà **khớp operator / rule**.

2. **Program + execute đúng domain lặp lại trên nhiều họ PASS**  
   Discrete path, matrix path, symbolic rules, binding, hierarchy, grounded grid — cùng *protocol*, khác *operator*.

3. **Residual / seq2seq thắng ID, gãy systematic OOD** trên language/grid  
   → Shortcut học train distribution; bias cấu trúc mới unlock OOD.

### 3.2 Hạn chế / điều kiện biên

1. **Control & Probabilistic không PASS với learned router/mixture**  
   Meta-claim **không** nói “mọi họ sẽ PASS nếu đúng bias”; nói “sai bias thì FAIL / đúng bias mới có cửa”.  
   Control: đúng physics + **oracle program** PASS → bias “biết program” đủ; **learned OOD program** vẫn gãy.  
   → Bổ sung: *credit assignment / OOD program inference* là họ con khó, chưa giải bằng Gumbel/mixture đơn giản.

2. **Symbolic/grounded PASS phụ thuộc executor rule đúng**  
   Upper bound gần compiler. Claim thực dụng: *inductive bias rule+world*, không phải “neural tự khám phá grammar từ MSE”.

3. **Không có một checkpoint duy nhất re-run mọi task**  
   Unify = **API** (program→execute), không = **một bộ trọng số**. Audit **không** ủng hộ universal weight set.

### 3.3 Không đánh trống

- FAIL control/prob **không** phủ nhận PASS discrete/algebraic/symbolic.  
- Oracle control **không** được đếm như “learned composition PASS”.

---

## 4. Meta-claim sau audit (chốt)

**Giữ, với điều kiện:**

> Composition generalization theo họ đòi hỏi **operator / program representation khớp domain** (routing bank, matrix trên state, symbolic rewrite, stack, grid execute).  
> Flat residual và soft seq2seq đủ ID nhưng **không** systematic OOD trên các họ đã PASS.  
> Khi program **phải học và OOD** dưới dynamics liên tục hoặc mixture (control / probabilistic setups đã thử), residual vẫn thắng — ranh giới **program inference**, không phải “composition vô nghĩa”.

**Không claim:**

- Một PureC/Transformer nhỏ cover control + probabilistic OOD.  
- Neural tự học rule thrice/opposite/after không có bias symbolic.  
- Learned Gumbel skill routing = hierarchical RL trưởng thành.

---

## 5. Phân loại kết quả (khoa học)

| Loại | Ví dụ |
|------|--------|
| **PASS có cơ chế** | E2 matrix, F2 symbolic m, G/H/K execute |
| **FAIL có cơ chế** | E Givens@H, I2–J2 routing, L mixture |
| **Upper bound** | I4 oracle, I5 TF-OOD |
| **Protocol artifact đã loại** | I goal-leak |
| **Freeze có ranh giới** | Control, Probabilistic |

FAIL được dùng như **công cụ thiết kế** (loại bias sai lớp), nhất quán protocol B suốt lab.

---

## 6. Khuyến nghị sau audit

1. **Freeze lab mở rộng** theo meta-claim đã chốt (mục 4).  
2. Báo cáo công bố: nhấn **đa bias theo họ** + bảng PASS/FAIL; không bán “universal TNR”.  
3. Phase mới chỉ khi: bias **mới** cho control program-OOD (planner/search/RL) hoặc probabilistic calibrated đúng task ambiguity — preregister riêng, không sửa claim cũ.

---

## 7. Artifact

- Lineage append-only: `Lineage.md`  
- Freezes: `reports/bao_cao_freeze_3_claim.md`, `reports/bao_cao_freeze_control.md`  
- Code: `code/tnr_phase_*.py` · Runs: `r/uns/run_*`

**Audit status: COMPLETE — meta-claim retained with boundary conditions.**
