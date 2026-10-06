# Báo cáo khoa học: TNR Run v3 – Kế hoạch run tiếp theo (không đổi claim)

**Ngày:** 2026-10-06  
**Run ID:** 20261006_151416  
**Đường dẫn:** `/home/workdir/artifacts/r/uns/run_20261006_151416`  
**Outcome:** **UNRESOLVED**

---

## 1. Mục tiêu đã thực hiện

Thực hiện đúng 6 điểm kế hoạch:

1. Tăng HIDDEN = 48, N_ITER = 5, data 5k → **R² baseline đạt 0.85–0.86** (vượt mục tiêu > 0.6).  
2. True transform compositional: mỗi module cộng dồn residual trên chiều riêng (unique dims).  
3. Soft gates + temperature annealing (T: 2.0 → 0.5) kết hợp soft top-K.  
4. Cân bằng param E: recurrent wider (2 cell) → ~10k params (trước đó chỉ 1.6k).  
5. Thêm metric trajectory recovery (node recall trên path).  
6. Giữ nguyên ngưỡng PASS đã preregister.

Claim trung tâm **không thay đổi**.

---

## 2. Preregistration (giữ nguyên)

**PASS** chỉ khi:
- MSE_D < min(MSE của A,B,C,E,F) × 0.95  
**VÀ**  
- route_edge_f1_D ≥ 0.30  

**FAIL** nếu MSE_D > min_baseline × 1.05  
**Còn lại** → UNRESOLVED

---

## 3. Kết quả

| Model | MSE ↓ | R² ↑ | Params | Edge F1 | Traj node recall |
|-------|-------|------|--------|---------|------------------|
| A (MLP) | 0.1696 | 0.852 | 15 124 | — | — |
| B (Residual modular) | **0.1603** | **0.860** | 38 644 | — | — |
| C (Static K8) | 0.1653 | 0.855 | 41 976 | — | — |
| **D (Gated K8 soft+temp)** | 0.1679 | 0.853 | 48 232 | **0.064** | 0.449 |
| E (Recurrent dense balanced) | 0.1614 | 0.859 | 10 420 | — | — |
| F (Gated random graph) | 0.1693 | 0.852 | 48 232 | 0.095 | 0.576 |

- Best baseline MSE = 0.1603 (B)  
- D MSE = 0.1679 (> best × 0.95)  
- Edge F1 của D = 0.064 (<< 0.30)  
- Traj node recall D ≈ 0.45 (trung bình)

**Outcome theo preregistration: UNRESOLVED**

---

## 4. Phân tích

1. **Mục tiêu R² > 0.6 đã đạt** (thực tế > 0.85) nhờ true transform compositional residual + capacity tăng.  
2. **Adaptive routing (D) không vượt** residual modular (B) hay recurrent dense (E) về MSE.  
3. **Route recovery vẫn yếu**: edge F1 thấp; traj node recall ~0.45 cho thấy router có tín hiệu nhẹ về node nhưng chưa bắt được thứ tự cạnh.  
4. **Param D/F cao hơn B/C** (~48k vs ~39–42k) do router; E đã cân bằng hơn.  
5. Temperature annealing + soft top-K giữ active edges ổn định (~1.6) nhưng chưa đủ để học latent path dưới λ hiện tại.

Không có tín hiệu dương đủ mạnh → không chạy ablation mechanism thêm trong run này.

---

## 5. Artifact

- Code: `code/tnr_experiment_v3.py`  
- Run: `r/uns/run_20261006_151416/` (metrics.json, history_A…F.json)  
- Log: `r/uns/v3_log.txt`  
- Lineage.md đã append **UNRESOLVED**  
- Báo cáo này: `reports/bao_cao_khoa_hoc_TNR_v3.md`

---

## 6. Kế hoạch run sau (vẫn giữ claim)

1. Giảm param gap (shared router / smaller router / low-rank router mạnh hơn).  
2. Tăng tín hiệu routing: curriculum trên path length hoặc auxiliary route loss nhẹ (chỉ khi preregister).  
3. Thử hard top-K + STE trở lại với temperature schedule khác, hoặc Gumbel-softmax.  
4. Đo trajectory sequence matching (LCS hoặc exact path recovery) chi tiết hơn.  
5. Giữ nguyên ngưỡng PASS.  
6. Nếu vẫn UNRESOLVED sau 1–2 run nữa với capacity đã đủ → xem xét FAIL có kiểm soát hoặc thu hẹp task geometry.

---

*Toàn bộ nội dung bằng tiếng Việt; thuật ngữ kỹ thuật tiếng Anh chỉ nằm trong ngoặc đơn khi cần thiết.*
