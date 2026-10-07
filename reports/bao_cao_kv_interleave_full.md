# Báo cáo: Xen kẽ KV_causal + 1 lớp Full all-to-all

**Run ID:** 20261007_004313  
**Outcome:** **PASS** (biên, theo ngưỡng 0.97×)

## Thiết kế
- **Interleave:** Layer0 KV_causal (sparse) → Layer1 Full all-to-all → Layer2 KV_causal (complete)
- So sánh với HierResidual và KV_causal thuần (3 lớp causal)
- 3 seeds: 42, 7, 123

## Kết quả OOD (mean ± std)

| Model | OOD MSE ↓ | Traj recovery |
|-------|-----------|---------------|
| HierResidual | 0.1818 ± 0.004 | — |
| KV_causal Pure | 0.1791 ± 0.004 | 0.250 ± 0.015 |
| **Interleave (Causal–Full–Causal)** | **0.1760 ± 0.002** | **0.266 ± 0.015** |

- Inter thắng residual theo ngưỡng 0.97× (0.1760 < 0.1763) → **PASS biên**.
- Inter hơi tốt hơn Pure trên OOD; traj tương đương / hơi cao hơn.
- Wins: Inter tốt hơn HierRes ở 2–3/3 seeds.

## Phân tích
1. Lớp Full all-to-all ở giữa có thể giúp **trộn thông tin toàn cục** một lần, bù cho hạn chế của causal tuần tự (chỉ nhìn lịch sử local).
2. Hai lớp KV_causal ngoài vẫn giữ trajectory có thứ tự.
3. Kết quả PASS rất sát ngưỡng — cần thận trọng, không tuyên bố mạnh.
4. Đây là tín hiệu đầu tiên đạt PASS theo preregistration kể từ đầu lineage trên task này.

## Kết luận tạm
Xen kẽ 1 lớp full all-to-all vào giữa các lớp KV_causal **có vẻ hữu ích** (trộn global + routing local). Nên:
- Xác nhận thêm 1–2 seed, hoặc
- Giữ cấu hình này làm candidate chính trước khi thu hẹp claim / đổi task.

## Artifact
- Code: `code/tnr_kv_interleave_full.py`
- Run: `r/uns/run_20261007_004313/`
- Lineage đã append.
