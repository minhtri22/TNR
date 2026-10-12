# TNR / Agent ladder — mã nguồn thí nghiệm

## Cấu trúc

- `tnr_phase_*.py` — từng phase agent ladder (một giả thuyết / file)
- `tnr_purec_*.py`, `tnr_kv_*.py`, `tnr_seq_*.py` — lõi lab sớm
- `../data/human_nl/` — SAIL corpus, annotate, trace gold
- `../r/uns/` — metrics.json + logs theo run
- `../reports/` — báo cáo tiếng Việt
- `../Lineage.md` — PASS/FAIL append-only

## Phase agent (thứ tự logic)

| File | Phase |
|------|--------|
| tnr_phase_M*.py | NL→program |
| tnr_phase_N*.py | Tools bank |
| tnr_phase_O*.py | Multi-turn |
| tnr_phase_CF_control_flow.py | Control-flow |
| tnr_phase_R1/R2*.py | HTTP mock / fault |
| tnr_phase_R2d_fault_alias.py | Fault aligned + alias (PASS) |
| tnr_phase_ProdNL_freeform.py | Free-form NL |
| tnr_phase_HumanNL_open_corpus.py | Human curated |
| tnr_phase_R3_ticket_crm.py | Domain CRM hẹp |
| tnr_phase_SAIL_*.py | SAIL human + full-graph + 3-bền |
| tnr_phase_2ben_tools.py | 2-bền tools (B5/B1/B6/B4) |
| tnr_phase_3ben_sail.py | 3-bền SAIL (A2/A1/A4) |

## Chạy

```bash
cd /path/to/artifacts
python code/tnr_phase_R3_ticket_crm.py
```

Yêu cầu: Python 3.10+, `torch`, `numpy`. Một số phase SAIL cần `data/human_nl/sail/`.

## Lưu ý

Nhiều run muộn từng chạy qua REPL; các file `*_R2d*`, `*_R3*`, `*_SAIL*`, `*_2ben*` được **đóng gói lại** từ protocol đã PASS để repo đủ mã nguồn tái lập.
