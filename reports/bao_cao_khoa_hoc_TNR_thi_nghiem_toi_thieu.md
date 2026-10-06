# Báo cáo khoa học: Thí nghiệm tối thiểu đầu tiên của nền tảng nghiên cứu Topological Neural Routing (TNR)

**Ngày:** 2026-10-06  
**Phiên bản:** 0.2 (hai run đầu tiên)  
**Trạng thái outcome:** UNRESOLVED (cả hai run)  
**Run ID chính:** 20261006_144921 (và 20261006_144648)  
**Đường dẫn run:** `/home/workdir/artifacts/r/uns/`

---

## 1. Mục đích và phạm vi

Tài liệu này báo cáo kết quả thực hiện thí nghiệm tối thiểu đầu tiên được mô tả trong nền tảng nghiên cứu TNR (mục 11).

Mục tiêu không phải chứng minh ngay lập tức sự vượt trội của adaptive topological routing, mà là:

- Xây dựng môi trường synthetic có dependency geometry biết trước và thay đổi theo case.
- So sánh bốn họ mô hình dưới constraint parameter/compute gần tương đương.
- Ghi nhận metric task performance, routing behavior và outcome theo quy tắc preregistered (PASS / FAIL / UNRESOLVED).
- Lưu trữ đầy đủ artifact để lineage khoa học có thể kiểm chứng lại.

Giả thuyết trung tâm được khóa trước (mục 12):

> Adaptive topological routing > fixed depth  
> trên task family có variable computational dependency graph, dưới parameter/compute budget được khóa trước.

Dấu “>” được operationalize thành: test MSE thấp hơn rõ rệt (ngưỡng 5%) so với baseline tốt nhất, hoặc compute thấp hơn tại cùng accuracy. Không được thay đổi định nghĩa sau khi nhìn kết quả.

---

## 2. Thiết kế thí nghiệm

### 2.1. Task geometry (synthetic)

- Input: \( x \in \mathbb{R}^{16} \)
- Output: \( y \in \mathbb{R}^{4} \) (hồi quy)
- Ba case family với latent dependency path cố định (không cung cấp cho mô hình):
  - Family 0: module 0 → 3 → 6
  - Family 1: module 0 → 5 → 2
  - Family 2: module 1 → 7 → 4 → 2
- Ground-truth composition: mỗi module là một phép biến đổi tuyến tính + tanh cố định (được sinh một lần và giữ nguyên).
- Noise nhỏ được thêm vào target.

Số lượng mẫu: 4000 train / 800 val / 800 test. Seed = 42.

### 2.2. Các mô hình so sánh

| Ký hiệu | Mô tả | Ghi chú |
|---------|-------|---------|
| A | 8-layer MLP (fixed depth) | Baseline cổ điển |
| B | Residual modular (8 module xếp chồng + residual) | Dense/residual modular baseline |
| C | Static K8 modular | Complete adjacency, không gate, 3 iteration |
| D | Gated K8 modular | Dynamic soft gate + route cost \(\lambda \mathcal{C}_{route}\) |

Tất cả dùng hidden width = 24, 3 iteration cho C/D, optimizer Adam, lr = 1e-3, 80 epoch, early-stop theo val loss tốt nhất.

Router pressure cho D: \(\mathcal{L} = \mathcal{L}_{task} + 0.01 \times \sum g_{ij}\) (soft).

### 2.3. Metric families (preregistered)

- **Task performance:** MSE (chính), R²
- **Compute proxy:** số parameter, số active edge trung bình (cho D)
- **Routing:** route fidelity (tỷ lệ gate mass nằm trên true edges so với tổng)
- Không kết luận mechanism chỉ từ accuracy.

### 2.4. Kiểm soát confound

- Seed cố định
- Cùng số epoch / lr / optimizer
- Width và iteration cố định trước
- (Lưu ý: số parameter thực tế chưa hoàn toàn bằng nhau — D cao hơn do router; đây là hạn chế của run đầu)

---

## 3. Kết quả

### 3.1. Run 20261006_144648 (HIDDEN=24, N_ITER=3, λ=0.01)

| Model | Params | MSE ↓ | R² ↑ | Route fidelity | Avg active edges / iter |
|-------|--------|-------|------|----------------|-------------------------|
| A     | 4 108  | 0.4045 | 0.243 | — | — |
| B     | 10 108 | 0.4032 | 0.246 | — | — |
| C     | 11 380 | 0.3996 | 0.253 | — | — |
| D     | 24 548 | 0.4013 | 0.249 | 0.043 | ≈ 0.002 |

Outcome: **UNRESOLVED** (D không vượt baseline tốt nhất một cách rõ rệt; router gần như tắt).

### 3.2. Run 20261006_144921 (HIDDEN=32, N_ITER=4, λ=0.001, data lớn hơn)

| Model | Params | MSE ↓ | R² ↑ | Route fidelity | Avg active edges / iter |
|-------|--------|-------|------|----------------|-------------------------|
| A     | 7 012  | 0.3954 | 0.256 | — | — |
| B     | 17 572 | 0.3994 | 0.248 | — | — |
| C     | 19 524 | 0.3985 | 0.250 | — | — |
| D     | 35 764 | 0.3951 | 0.256 | 0.046 | ≈ 0.92 |

Outcome: **UNRESOLVED** (D MSE gần bằng A; fidelity vẫn thấp; param của D cao hơn đáng kể).

### 3.3. Nhận xét chung

Tất cả mô hình đạt R² chỉ khoảng 0.25, cho thấy task composition hiện tại vẫn quá khó với capacity đã thử. Router của D không học được latent path (fidelity ≈ 0.04–0.05, gần mức ngẫu nhiên). Không có tín hiệu dương rõ ràng ủng hộ giả thuyết adaptive topological routing trong hai run này.

---

## 4. Phân tích và hạn chế

1. **Task signal yếu / capacity thấp:** R² chỉ ~0.25 cho thấy composition ground-truth khó học với width 24 và 3 iteration. Cần tăng capacity hoặc đơn giản hóa true transform.
2. **Parameter imbalance:** D có nhiều param hơn do router. Cần low-rank router hoặc chia sẻ parameter tốt hơn ở run sau.
3. **Route cost quá mạnh:** \(\lambda=0.01\) kết hợp với sum gate đã gần như tắt hết cạnh. Cần schedule \(\lambda\) hoặc constraint soft-max-K.
4. **Chưa có ablation:** Vì không có tín hiệu dương, ablation (static vs gated, random graph, fixed route, …) chưa được chạy.
5. **Chưa đo FLOPs thật / activation count:** chỉ dùng proxy parameter + active edges.

Những hạn chế này được ghi nhận công khai và sẽ được xử lý trong lineage tiếp theo mà không “sửa hypothesis hồi tố”.

---

## 5. Kết luận khoa học của run này

- Run đầu tiên **không ủng hộ** cũng **không bác bỏ** giả thuyết trung tâm trong phạm vi đã khóa.
- Outcome chính thức: **UNRESOLVED**.
- Giá trị của run: thiết lập pipeline reproducible, metric, lineage và artifact storage; đồng thời chỉ ra các confound cần khóa chặt hơn (capacity, \(\lambda\), param parity).

Theo nguyên tắc của nền tảng TNR (mục 13): UNRESOLVED không được biến thành PASS. Hypothesis gốc được giữ nguyên cho các run tiếp theo.

---

## 6. Artifact và reproducibility

- Code: `/home/workdir/artifacts/code/tnr_minimal_experiment.py`
- Metrics + config: `r/uns/run_20261006_144648/metrics.json`
- Histories: `history_A/B/C/D.json`
- Model weights: `model_A/B/C/D.pt`
- Lineage append-only: `/home/workdir/artifacts/Lineage.md`

Mọi run tiếp theo phải append vào Lineage.md và lưu dưới `r/uns/run_YYYYMMDD_HHMMSS`.

---

## 7. Kế hoạch run tiếp theo (không thay đổi claim)

1. Tăng HIDDEN hoặc N_ITER để đưa R² baseline lên > 0.6.
2. Cân bằng parameter (low-rank interface + smaller router).
3. Điều chỉnh \(\lambda\) hoặc dùng hard top-K routing.
4. Thêm baseline recurrent dense và random-graph gated.
5. Đo route recovery chính xác hơn (Hungarian matching trên path).
6. Preregister ngưỡng PASS rõ ràng hơn trước khi chạy.

---

*Báo cáo được viết hoàn toàn bằng tiếng Việt; các thuật ngữ kỹ thuật tiếng Anh được giữ trong ngoặc đơn khi cần thiết để chính xác khoa học.*
