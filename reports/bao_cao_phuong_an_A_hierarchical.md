# Báo cáo: Phương án A – Hierarchical Topological Stacking

**Ngày:** 2026-10-06  
**Run ID:** 20261006_154757  
**Outcome:** **UNRESOLVED**

---

## 1. Mục tiêu đã thực hiện

- Message-passing **song song** trong mỗi lớp.
- Hai biến thể:
  - **SameTopo**: cùng topology (ring) lặp lại qua 3 lớp.
  - **DiffTopo**: topology khác nhau mỗi lớp (sparse → ring → complete).
- So sánh với baseline: FlatMLP, Hierarchical Residual, SameTopo/DiffTopo static (không gate).
- Prior: **nhẹ** (temperature annealing + soft top-K). Không teacher forcing, không auxiliary path loss.

---

## 2. Thiết kế

- 3 lớp hierarchical, mỗi lớp 4 module (tổng 12 đơn vị).
- Trong lớp: parallel message passing + soft top-K.
- Giữa lớp: residual mix (mean → linear → cộng vào mọi module).
- Task: compositional residual (giống v3, R² cao dễ đạt).

---

## 3. Kết quả

| Model | MSE ↓ | R² ↑ | Params | Avg active |
|-------|-------|------|--------|------------|
| FlatMLP | 0.1617 | 0.855 | 9 044 | — |
| **HierResidual** | **0.1585** | **0.858** | 40 204 | — |
| SameTopo_static | 0.1624 | 0.854 | 47 972 | — |
| SameTopo_gated | 0.1675 | 0.850 | 57 332 | 8.0 |
| DiffTopo_static | 0.1603 | 0.856 | 47 972 | — |
| DiffTopo_gated | 0.1645 | 0.852 | 57 332 | 9.9 |

- Best baseline = HierResidual (0.1585)
- Best gated = DiffTopo_gated (0.1645) → **không tốt hơn** baseline
- DiffTopo_static hơi tốt hơn SameTopo_static
- Gated versions đều kém hơn static counterparts một chút

**Outcome: UNRESOLVED** (theo ngưỡng preregistered)

---

## 4. So sánh SameTopo vs DiffTopo

| Khía cạnh | SameTopo (ring lặp) | DiffTopo (sparse→ring→complete) |
|-----------|---------------------|---------------------------------|
| MSE static | 0.1624 | **0.1603** (tốt hơn nhẹ) |
| MSE gated | 0.1675 | **0.1645** (tốt hơn nhẹ) |
| Gate entropy | 0.67 | 0.92 (phân tán hơn) |
| Active edges | 8.0 | 9.9 |

**Nhận xét:**  
DiffTopo (thay đổi topology theo lớp) cho kết quả task hơi tốt hơn SameTopo, nhưng khoảng cách rất nhỏ và cả hai đều không vượt Hierarchical Residual. Routing vẫn không mang lại lợi ích rõ.

---

## 5. Phân tích nguyên nhân

1. Hierarchical Residual (không topology, không gate) đã rất mạnh → depth + residual đủ để giải task compositional này.
2. Thêm topology + gating làm tăng param và tối ưu khó hơn, trong khi không có tín hiệu buộc router phải chọn trajectory đúng.
3. Prior quá nhẹ (chỉ temperature + soft top-K) → router không bị ép học latent dependency.
4. Inter-layer chỉ residual mean → thông tin lan truyền “mềm”, không tạo áp lực chọn path rõ ràng.

Kết luận: Hierarchical stacking **có tiềm năng tổ chức depth**, nhưng với prior hiện tại, **routing adaptive vẫn chưa chứng minh được giá trị**.

---

## 6. Quyết định về prior cho run sau

Dựa trên thực nghiệm:
- F1 / recovery gần như không đo được hữu ích (proxy active/entropy cho thấy router hoạt động nhưng không tập trung đúng path).
- Cần **tăng prior có kiểm soát**:
  - Thêm auxiliary loss nhẹ khuyến khích gate mass trên cạnh “hợp lý” theo layer (không full teacher forcing).
  - Hoặc curriculum: giai đoạn đầu cho router nhìn soft target path, sau đó tắt dần.
  - Giữ hierarchical structure (DiffTopo ưu tiên hơn SameTopo).

---

## 7. Artifact

- Code: `code/tnr_hierarchical_A.py`
- Run: `r/uns/run_20261006_154757/`
- Log: `r/uns/hier_A_log.txt`
- Lineage đã append **UNRESOLVED**

---

*Nội dung bằng tiếng Việt; thuật ngữ kỹ thuật tiếng Anh chỉ trong ngoặc đơn khi cần.*
