# Bản đồ các phiên bản (v1–v5)

| Phiên bản | Tiền đăng ký | Thư mục kết quả | Script chạy / phân tích | Kết quả tổng hợp | Hình | Phần báo cáo | Mục Lineage |
|---|---|---|---|---|---|---|---|
| v1 | `prereg/PREREG_v1.md` | `runs/run_001_main/` | `src/run_all.py`, `src/analyze.py` | `results/results.json` | `figures/` (`mse_theo_mo_hinh.png`, …) | REPORT §1–7 | L001–L009 |
| v2 | `prereg/PREREG_v2.md` | `runs/run_002_main/` | `src/run_v2.py`, `src/analyze_v2.py` | `results/results_v2.json` | `figures/v2_*` | REPORT Phần II | L010–L022 |
| v3 | `prereg/PREREG_v3.md` | `runs/run_003_main/` | `src/run_v3.py`, `src/analyze_v3.py` | `results/results_v3.json` | `figures/v3_ket_qua.png` | REPORT Phần III | L023–L036 |
| v4 | `prereg/PREREG_v4.md` | `runs/run_004_main/` | `src/run_v4.py`, `src/analyze_v4.py` | `results/results_v4.json` | `figures/v4_ket_qua.png` | REPORT Phần IV | L037–L049 |
| v5 | `prereg/PREREG_v5.md` | `runs/run_005_main/` | `src/run_v5.py`, `src/analyze_v5.py` | `results/results_v5.json` | `figures/v5_ket_qua.png` | REPORT Phần V | L050–L061 |

Dùng chung: `src/tasks.py` (nhiệm vụ), `src/models.py`, `src/models_v3.py`, `src/models_v4.py` (mô hình), `src/train.py` (huấn luyện/đánh giá), `src/lineage.py` (sổ chỉ-ghi-thêm, kiểm tra chuỗi băm: `python src/lineage.py verify`). Mã thăm dò/hiệu chỉnh: `src/pilot*.py`, `calib_oracle.py`, `diag.py`, `patch*.py`.

## Về checkpoint
Các lượt v1–v5 **không lưu trọng số mô hình** (chỉ lưu số liệu đánh giá, đường cong huấn luyện và, với D, các chỉ số cơ chế, trong tệp JSON của `runs/`); vì vậy repo không có checkpoint cho v1–v5. Từ v6, `src/run_v6.py` lưu tham số cuối của mỗi lần chạy vào `ckpt/*.npz` cùng thư mục kết quả.

## Cách tái lập
`pip install -r requirements.txt`; `python src/run_v6.py --worlds 90 --out runs/run_006_main`; `python src/analyze_v6.py`. Mã v1–v5 dùng cùng `src/train.py`, `src/models*.py`.
