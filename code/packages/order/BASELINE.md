# Order pack — FROZEN production baseline (lab)

**Status:** FROZEN  
**Date:** 2026-10-14

## What is in the freeze

| Layer | Artifact | Gate |
|-------|----------|------|
| Train / structure OOD (neural) | `eval/suite_order_v0b.py` | **PASS** Prog=1.0 Flat=0.59 |
| Kernel policy CONFIRM | `schema/tools.json` + kernel validate | **PASS** |
| File / CSV / HTTP connectors | `conn_v1`–`v3` | **PASS** |
| Shopee/Sapo-shaped HTTP | `conn_v4` + adapters | **PASS** |
| Oracle tools → HTTP | `suite_order_oracle_http_v1.py` | **PASS** exact=1.0 |
| **NL path (controlled phrases)** | **`suite_order_nl_lexicon_v1.py`** | **PASS** match=1.0 HTTP=1.0 |

## Stack to ship against

```text
NL (listed / synonym phrases)
  → lexicon slot-fill     (nl_lexicon_v1)
  → validate + CONFIRM
  → HTTP marketplace mock  (conn_v4 Shopee/Sapo shape)
  → audit
```

Regression commands:

```bash
python code/packages/order/eval/suite_order_v0b.py
python code/packages/order/eval/suite_order_conn_v4.py
python code/packages/order/eval/suite_order_nl_lexicon_v1.py
python code/packages/order/eval/suite_order_oracle_http_v1.py
```

## Documented open breaks (not baseline)

| Suite | Outcome | Meaning |
|-------|---------|---------|
| v0c / lexicon_v1_cold | UNRESOLVED / **FAIL** | true lexical cold-start |
| v0d hybrid word | FAIL | partial synonym insufficient |
| v0e hybrid char | UNRESOLVED | char encoder below bar |
| e2e_v1 / v1b neural→HTTP | **FAIL** | NL neural gap (oracle/lexicon separate it) |

## Rules after freeze

1. Do not loosen PASS thresholds to absorb cold-start or neural e2e.
2. Cold-start NL and OS-LLM fine-tune are **future packs**, not required for this baseline.
3. New connectors must pass conn_v4-style gate + v0b metrics still PASS.
4. Ticket / other domains = new `packages/<name>/`, same kernel.
