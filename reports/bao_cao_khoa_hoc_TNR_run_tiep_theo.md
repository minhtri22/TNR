# Báo cáo khoa học: Thực hiện kế hoạch run tiếp theo TNR (không thay đổi claim)

**Ngày:** 2026-10-06  
**Phiên bản:** 0.3  
**Run ID:** 20261006_150521  
**Đường dẫn:** `/home/workdir/artifacts/r/uns/run_20261006_150521`  
**Outcome:** **UNRESOLVED**

---

## 1. Mục tiêu của run này

Thực hiện đúng 6 điểm trong kế hoạch run tiếp theo (không sửa claim trung tâm):

1. Tăng capacity (HIDDEN, data) và đơn giản hóa true transform để hướng tới R² baseline > 0.6.  
2. Cân bằng parameter bằng low-rank interface + router nhỏ.  
3. Chuyển sang hard top-K routing + straight-through estimator (STE) + λ nhẹ.  
4. Thêm baseline E = recurrent dense và F = gated random-graph.  
5. Đo route recovery chính xác hơn bằng edge-level precision / recall / F1 trên true path edges.  
6. Preregister ngưỡng PASS rõ ràng trước khi chạy.

Claim gốc vẫn giữ nguyên:

> Adaptive topological routing > fixed depth  
> trên task family có variable computational dependency graph, dưới parameter/compute budget khóa trước.

---

## 2. Preregistration (khóa trước execution)

**Metric chính:** test MSE (thấp hơn tốt hơn).  
**Metric phụ:** route edge F1 (phục hồi cạnh trên latent path).

**Điều kiện PASS (không được thay đổi sau kết quả):**
- MSE_D < min(MSE của A,B,C,E,F) × 0.95  
**VÀ**  
- route_edge_f1_D ≥ 0.30  

**FAIL:** MSE_D > min_baseline × 1.05  
**Còn lại:** UNRESOLVED  

Param mục tiêu khoảng 4k–15k; compute khóa bởi N_ITER = 3, EPOCHS = 30, cùng optimizer/lr.

---

## 3. Thiết kế cải tiến

### 3.1. Task
- INPUT_DIM = 12, OUTPUT_DIM = 4  
- True transform: ma trận linear scale 0.35 + boost 2 chiều riêng theo module + tanh → giữ variance hợp lý (std ≈ 0.45–0.62).  
- 3 case family với path cố định như trước.  
- N_TRAIN = 2000 (phiên bản light để hoàn thành trong thời gian thực tế).

### 3.2. Mô hình
| Ký hiệu | Mô tả | Params |
|---------|-------|--------|
| A | 8-layer MLP | 4 012 |
| B | Residual modular stack | 10 012 |
| C | Static K8 (mean-field + low-rank msg) | 11 492 |
| D | Gated K8 + hard top-K=3 + STE | 14 788 |
| E | Recurrent dense (cùng N_ITER) | 1 612 |
| F | Gated random sparse graph (cùng top-K) | 14 788 |

Router của D/F chỉ nhận (input + mean state) → 32 → N×N, rất nhỏ. Interface message dùng bottleneck 16 chiều.

### 3.3. Routing
- Hard top-K = 3 cạnh đến mỗi module.  
- Straight-through estimator để backprop.  
- λ = 0.0005 trên số cạnh kích hoạt.  
- Metric recovery: trung bình gate qua iteration → so sánh tập cạnh predicted (gate > 0.5) với true path edges → precision/recall/F1.

---

## 4. Kết quả

| Model | MSE ↓ | R² ↑ | Params | Route edge F1 | Avg active edges/iter |
|-------|-------|------|--------|---------------|-----------------------|
| A     | 0.1728 | 0.384 | 4 012  | — | — |
| B     | 0.1744 | 0.378 | 10 012 | — | — |
| C     | 0.1717 | 0.388 | 11 492 | — | — |
| D     | 0.1763 | 0.371 | 14 788 | 0.076 | 8.0 |
| E     | 0.1752 | 0.375 | 1 612  | — | — |
| F     | 0.1739 | 0.380 | 14 788 | 0.051 | 8.0 |

- Best baseline MSE = 0.1717 (C).  
- D MSE = 0.1763 (cao hơn baseline, không đạt 0.95×).  
- Route F1 của D chỉ 0.076 (xa ngưỡng 0.30).  
- R² baseline cao nhất ≈ 0.39 (chưa đạt mục tiêu > 0.6, dù đã cải thiện so với run trước ~0.25).

**Outcome theo preregistration: UNRESOLVED**

---

## 5. Phân tích

1. **R² chưa vượt 0.6:** Capacity vẫn chưa đủ hoặc true composition còn khó với 3 iteration. Cần tăng thêm HIDDEN/N_ITER hoặc làm true transform compositional hơn (ví dụ additive module effects).  
2. **Param tương đối cân bằng** giữa C/D/F (~11k–15k); E bị under-parameterized rõ.  
3. **Hard top-K hoạt động** (active edges ổn định = 8), nhưng router chưa học được latent path (F1 thấp, gần ngẫu nhiên).  
4. **Không có lợi ích đo được** của adaptive K8 so với static K8 hay random-graph gated.  
5. **Ablation sẵn sàng:** vì không có tín hiệu dương, chưa cần chạy thêm ablation mechanism.

Những điểm trên được ghi nhận công khai. Hypothesis gốc không bị sửa.

---

## 6. Artifact

- Code: `code/tnr_experiment_v2.py` (bản đầy đủ) và `tnr_experiment_v2_light.py` (bản chạy thành công)  
- Run: `r/uns/run_20261006_150521/` (metrics.json, history_*, model_*.pt)  
- Lineage.md đã append outcome **UNRESOLVED**  
- Log: `r/uns/v2_light_log.txt`

---

## 7. Kế hoạch run sau (vẫn không đổi claim)

1. Tăng HIDDEN ≥ 64 và/hoặc N_ITER = 5–6 + data 8k để R² baseline > 0.6.  
2. Làm true transform mang tính composition rõ hơn (module cộng dồn feature riêng).  
3. Thử soft-max-K hoặc temperature annealing thay vì pure hard top-K.  
4. Cân bằng param của E (tăng width recurrent).  
5. Thêm metric trajectory recovery (sequence matching) ngoài edge F1.  
6. Giữ nguyên ngưỡng PASS đã preregister.

---

*Toàn bộ nội dung bằng tiếng Việt; thuật ngữ kỹ thuật tiếng Anh chỉ xuất hiện trong ngoặc đơn khi cần thiết.*
