# Topological Neural Routing (TNR) — gói nghiên cứu

Xem `VERSIONS.md` (bản đồ v1–v6), `REPORT.md` (báo cáo), `runs/Lineage.md` (sổ dòng dõi chỉ-ghi-thêm, có chuỗi băm), `colab/TNR_v6_colab.ipynb` (chạy v6 trên Colab).

# Gói nghiên cứu TNR (v1)
- `REPORT.md` — báo cáo khoa học (tiếng Việt)
- `prereg/PREREG_v1.md` — tiền đăng ký đã khóa
- `runs/Lineage.md` — sổ dòng dõi chỉ-ghi-thêm (chuỗi băm), có nhãn ĐẠT/KHÔNG ĐẠT/CHƯA KẾT LUẬN
- `runs/run_001_main/` — 96 tệp kết quả (8 seed × 12 cấu hình), `meta.json`
- `results/results.json` — số liệu tổng hợp; `figures/` — hình
- `src/` — mã nguồn (JAX); `pilot*.py`, `diag.py`, `patch*.py` là mã thăm dò/vá thiết kế, giữ lại để truy vết
- `prereg/PREREG_v2.md`, `prereg/PREREG_v3.md` — tiền đăng ký v2, v3; `runs/run_002_main/`, `runs/run_003_main/` — kết quả v2, v3; `results/results_v2.json`, `results_v3.json`; `src/analyze_v2.py`, `analyze_v3.py`, `run_v2.py`, `run_v3.py`, `models_v3.py`
- `prereg/PREREG_v4.md`, `runs/run_004_main/`, `results/results_v4.json`, `figures/v4_ket_qua.png`, `src/run_v4.py`, `src/analyze_v4.py`, `src/models_v4.py`
- `prereg/PREREG_v5.md`, `runs/run_005_main/`, `results/results_v5.json`, `figures/v5_ket_qua.png`, `src/run_v5.py`, `src/analyze_v5.py`
