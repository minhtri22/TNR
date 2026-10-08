# Báo cáo: Audit PASS trước + Hybrid Spark-inspired + so sánh 1:1 vs 2:1

**Run ID:** 20261008_041827  
**Outcome:** **UNRESOLVED** (Spark tốt nhất nhưng param ratio vượt ngưỡng)

---

## Phần 1 — Audit PASS run trước (20261007_004313)

| Kiểm tra | Kết quả |
|----------|---------|
| Tính mean OOD | Đúng: Inter 0.17596 < 0.97×HR 0.17630 |
| Khoảng cách tới ngưỡng | **0.0003 MSE** — razor-thin |
| Số seed | Chỉ 3 |
| Param Inter/HR | **1.36** (+36%) — không parity |
| Wins Inter vs HR | 2/3 (seed 7 thua) |
| Traj trong PASS criterion? | **Không** (chỉ OOD MSE) |

**Kết luận audit:** PASS trước đó **không đủ vững** về mặt khoa học. Giữ trong lineage như tín hiệu hướng, không tuyên bố mạnh.

---

## Phần 2 — Cảm hứng Spark-X2.5 Hybrid Attention

Spark-X2.5: **1 full-attention + 3 sliding-window attention**  
→ TNR map: **1 Full all-to-all + 3 KV_causal** (pattern `CCFC`)

So sánh thêm:
- R21 = `CFC` (2:1, cấu hình PASS biên trước)
- R11 = `CFCF` (1:1)
- HierResidual (tăng block để gần param hơn)

Preregister chặt hơn: OOD < 0.97×HR **và** wins≥2/3 **và** param_ratio ≤ 1.15

---

## Phần 3 — Kết quả (3 seeds)

| Model | Pattern | OOD MSE ↓ | Traj recovery | Params | Ratio vs HR |
|-------|---------|-----------|---------------|--------|-------------|
| HierResidual | — | 0.1880 ± 0.003 | — | 38 056 | 1.00 |
| **Spark (3C+1F)** | CCFC | **0.1793 ± 0.006** | **0.258** | 58 271 | 1.53 |
| R21 (2:1) | CFC | 0.1804 ± 0.007 | 0.232 | 43 800 | 1.15 |
| R11 (1:1) | CFCF | 0.1813 ± 0.003 | 0.215 | 57 898 | 1.52 |

- Spark thắng HR ở **3/3 seeds**, OOD tốt nhất.
- R11 (1:1) **traj thấp nhất** — đúng dự đoán “Full dày làm cắt trajectory”.
- R21 gần Spark về OOD, traj trung bình, ratio param tốt hơn Spark.
- **Outcome UNRESOLVED**: Spark vượt ngưỡng OOD nhưng **param_ratio 1.53 > 1.15**.

---

## Phần 4 — Ý nghĩa

1. **1:1 không tốt hơn 2:1 / 3:1** trên traj — Full quá dày hại recovery.
2. Pattern kiểu Spark (nhiều sliding/causal, ít full) **hướng đúng** (giống Spark: duy trì mạch lạc + tiết kiệm).
3. Lợi thế OOD vẫn **dính với capacity** — khi siết param parity, claim PASS chưa đứng vững.
4. Audit đã sửa điểm gãy: không còn tuyên bố PASS trên kết quả sát ngưỡng + lệch param.

---

## Artifact
- Code: `code/tnr_hybrid_spark_inspired.py`
- Run: `r/uns/run_20261008_041827/`
- Lineage đã append
