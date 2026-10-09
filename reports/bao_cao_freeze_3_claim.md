# Báo cáo freeze lab TNR — 3 claim theo họ composition

**Ngày freeze:** 2026-10-09  
**Nhánh:** `github.com/minhtri22/TNR` (`from-grok`)  
**Trạng thái:** **LAB CLOSED** cho 3 họ đã kiểm chứng. Binding / hierarchical **không** mở trong đợt này.

---

## 1. Ba claim được freeze

### Claim 1 — Discrete-bank transforms (**PASS**)

> Trên task composition **bank op rời rạc** (Lookup-Chain, Function Composition), kiến trúc **PureC** (KV_causal + Gumbel, `n_mod = n_ops`, `λ = 0`) vượt residual depth về OOD, reproducibility đa seed; trajectory recovery đạt ngưỡng khi hết collision module–op.

| Bằng chứng | Kết quả |
|------------|---------|
| Lookup-Chain 5 seeds | PASS (~19% OOD vs HR) |
| n_mod=6 | PASS (traj ≥ 0.45) |
| λ=0 | PASS (layout không bắt buộc) |
| OOD L=4 | PASS |
| Function Composition | PASS (ratio ~0.87) |
| Group-action + MLP modules | UNRESOLVED (ranh giới) |

**Không claim:** mọi continuous group action với module MLP.

---

### Claim 2 — Algebraic composition (**PASS**)

> Trên composition **đại số / group-like** (Givens + scale trên state), **explicit matrix composition trong STATE space** + router chọn module vượt residual; module Givens-MLP trên hidden **FAIL**.

| Bằng chứng | Kết quả |
|------------|---------|
| E structured Givens on H | **FAIL** (ratio ~1.14) |
| E2 matrix compose on STATE | **PASS** (ratio ~0.89, wins 3/3) |
| E2 OOD path L=4 | **PASS** (ratio ~0.84, wins 3/3) |

**Công thức:** compose đúng phép, đúng không gian trạng thái — không xấp xỉ MLP trên hidden.

---

### Claim 3 — Symbolic productivity (**PASS**)

> Trên productivity đếm (`twice` → `thrice`), **neural base unit + symbolic multiplicity** (m đọc từ token) đạt OOD exact = 1.0; seq2seq / thought-vector thuần OOD = 0.

| Bằng chứng | Kết quả |
|------------|---------|
| Mini-SCAN seq2seq thrice hold-out | OOD exact 0 |
| F ProgramExecutor học m neural | UNRESOLVED (tín hiệu yếu) |
| F2 neural base + symbolic m | **PASS** (ID=1, OOD=1 vs S2S OOD=0) |

**Công thức:** tách unit và rule lặp; rule thuộc vocabulary, không học lại từ MSE/CE hành động.

---

## 2. Nguyên tắc rút ra (meta-claim)

```text
Mỗi họ composition  →  một inductive bias khớp cơ chế
Không có một PureC / một seq2seq  cover mọi composition
FAIL có kiểm soát = công cụ thiết kế (loại bias sai lớp)
```

| Họ | Bias sai | Bias đúng |
|----|----------|-----------|
| Discrete | Parallel dense / collision n_mod | Explicit routing, n_mod=n_ops |
| Algebraic | Givens-MLP trên H | Matrix compose trên STATE |
| Symbolic count | Seq2seq / học m | Symbolic repeat + neural base |

---

## 3. Ngoài phạm vi (cố ý không freeze)

- Relational binding (*opposite*, *around*, variable sharing)
- Hierarchical / recursive syntax
- Continuous control / grounded multimodal
- Full SCAN / COGS / production systems

Mở bất kỳ mục trên = **phase mới**, một giả thuyết, preregister riêng — không sửa 3 claim trên.

---

## 4. Artifact

- Lineage: `Lineage.md` (append-only PASS/FAIL)
- Code: `code/tnr_*.py` (A–F2)
- Runs: `r/uns/run_*`
- Báo cáo trước: `reports/bao_cao_dong_goi_lab_phase_A_C.md`

---

## 5. Kết luận đóng lab

Lab **đủ** để khẳng định:  
**(1)** routing discrete-bank, **(2)** matrix algebraic compose, **(3)** symbolic productivity repeat — mỗi cái win khi đúng bias.  

**LAB FREEZE — 3 claims. Không mở binding trong đợt này.**
