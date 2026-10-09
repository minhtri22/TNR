# Báo cáo đóng gói lab TNR — Phase A → C

**Ngày:** 2026-10-09  
**Nhánh:** `github.com/minhtri22/TNR` (`from-grok`)  
**Ngôn ngữ:** Tiếng Việt; thuật ngữ chuyên môn giữ trong ngoặc đơn khi cần.

---

## 1. Mục tiêu nghiên cứu

Kiểm định inductive bias **định tuyến module tường minh (explicit modular routing)** — cụ thể kiến trúc **PureC (KV_causal + Gumbel controller)** — so với baseline depth residual / FFN trên:

1. Task composition rời rạc kiểu **Lookup-Chain** (lab A),
2. Độ phụ thuộc path supervision (lab A2),
3. Length generalization (lab A3),
4. Chuyển claim sang benchmark ngôn ngữ compositional Mini-SCAN (lab C).

---

## 2. Kiến trúc lõi (đã khóa)

```
Input → Linear → H
  → 3 × CausalKVLayer (topo: sparse → ring → complete)
       mỗi layer: SEQ bước
         Additive attention trên bảng KV (mặt nạ nhân quả slot đã ghi)
         Controller → Gumbel-Softmax chọn 1/n_mod
         Module_i(H) gated + residual
         Ghi H vào KV slot
  → Linear → output
```

**Cấu hình PASS cuối cùng (lab A):**

| Thành phần | Giá trị |
|------------|---------|
| n_mod | **6** (= số op / N_KEYS) |
| λ_layout | **0** (A2 chứng minh không bắt buộc) |
| SEQ_LEN | 3–4 tùy length eval |
| Hidden | 48 (Lookup) / 96 (SCAN seq2seq) |

---

## 3. Kết quả theo phase

### Phase A — Lookup-Chain (PASS)

| Thí nghiệm | Kết quả | Ý nghĩa |
|------------|---------|---------|
| PureC λ=0.15 vs residual, 5 seeds | **PASS** | OOD tốt hơn residual ~19%, traj ≥ 0.45 |
| n_mod=6 vs n_mod=4 | **PASS** | Hết collision op%mod → traj 0.47; OOD không xấu |
| A2: λ=0 vs λ=0.15, n_mod=6 | **PASS** | Layout prior **không bắt buộc** |
| A3: train L=2, OOD L=4 | **PASS** | Length gen giữ được (ratio OOD ~0.93 vs HR) |

**Claim A (freeze):**  
Trên Lookup-Chain multi-hop synthetic, PureC (n_mod = n_ops, λ=0) vượt HierResidual về OOD và đạt trajectory recovery ≥ 0.45, reproducibility 5 seeds.

### Phase B — Giảm supervision

**Bỏ** vì A2 PASS: không cần anneal / teacher forcing bắt buộc trước khi mở rộng task.

### Phase C — Mini-SCAN (UNRESOLVED / ranh giới)

| Setup | ID exact | OOD exact | Kết luận |
|-------|----------|-----------|----------|
| Bag encoder | 0 | 0 | Encoder hỏng order |
| LSTM + parallel head | 0 | 0 | Decode song song không đủ |
| Seq2seq (LSTM enc+dec) | **1.0** | 0 | Protocol OK; systematic OOD fail |
| Soft OOD (thrice hold-out) | 1.0 | 0 | Productivity fail cả FFN & PureC |
| Interp OOD (look composition) | 1.0 | 0 | Prim-transfer exact fail; token PureC hơi cao hơn |

**Claim C:**  
PureC thought-vector **không** vượt FFN trên Mini-SCAN compositional splits đã thử. Claim A **không** tự chuyển sang systematic/productivity language composition.

### Các hướng đã loại (FAIL / UNRESOLVED có kiểm soát)

Typed modules nhẹ, curriculum L2→L3 (OOD tốt nhưng traj tụt), Spiral/Mold/Winnow/Full như xương sống, λ≥0.25.

---

## 4. Giới hạn (limits)

1. Task A synthetic, scale nhỏ (~10⁵ params).  
2. Mini-SCAN chỉ mini subset; full SCAN / COGS chưa chạy.  
3. PureC gắn “thought vector” trước decoder — không phải neural module network đầy đủ với layout từ parser.  
4. Metric traj phụ thuộc mapping module↔op (đã sửa n_mod=6).

---

## 5. Kết luận đóng gói

- **Lab thành công** trên class task routing-critical (Lookup-Chain).  
- **Ranh giới rõ:** không claim universal compositional generalization.  
- **Kiến trúc freeze:** PureC + n_mod=n_ops + λ=0 + KV_causal.  
- **Phase D** (tiếp theo): một task **cùng class** A (compositional transform / multi-hop lookup), không SCAN.

---

## 6. Tham chiếu run (Lineage)

- PASS lõi 5-seed, n_mod=6, A2 λ=0, A3 L=4  
- C: seq2seq, soft thrice, interp look  

Chi tiết metrics: `r/uns/run_*`, `Lineage.md`.
