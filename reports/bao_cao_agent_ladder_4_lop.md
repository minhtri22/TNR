# Claim: Agent ladder đủ 4 lớp

**Ngày:** 2026-10-11  
**Nhánh:** from-grok

## Bốn lớp

| # | Lớp | Phase chứng minh | Outcome |
|---|-----|------------------|---------|
| 1 | **NL → program → execute** (single-turn, paraphrase/noise) | M2, M3, P, Q | PASS |
| 2 | **Tools / HTTP mock** | N3 exact, R1, R2d | PASS |
| 3 | **Multi-turn** (revise + state-mid) | O3, O4, O4b | PASS |
| 4 | **Control-flow** (if / branch) | **CF** | **PASS** (0.70 vs Flat 0.35) |

## Protocol chung

```text
NL (± multi-turn / condition) → structured program slots → Execute on world/tools
```

Flat residual ID cao, OOD systematic thấp hơn trên cả 4 lớp (khi metric đúng: exact / cell-acc).

## Ranh giới còn lại

- Multi-turn O/O2 encode-only UNRESOLVED (đã thay bằng O3/O4)
- R2 raw metric UNRESOLVED (đã R2d)
- Continuous robot control FREEZE (lab composition)
- Production HTTP + human NL tự do: chưa

## Kết luận

Agent ladder lab: **4/4 lớp PASS** với inductive bias program→execute.
