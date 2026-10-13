# Order pack — FROZEN baseline mock

**Status:** FROZEN  
**Baseline suite:** `eval/suite_order_v0b.py`  
**Commit reference:** order_v0b PASS (Prog=1.000, Flat=0.588, wins=3/3)

## What is frozen

| Item | Value |
|------|--------|
| Pack id | `order` |
| Schema | `schema/tools.json` (LOOKUP / UPDATE_ADDRESS / CONFIRM / CANCEL / SET_STATUS / CREATE / NOP) |
| Policy | `CANCEL_ORDER` requires `CONFIRM` before execute |
| Connector | `connectors/mock_store.py` (in-memory) |
| Eval regime | L=2 structure OOD + **synonym-in-train** |
| Preregister | PASS if mean Prog > Flat+0.05 AND Prog≥0.35 AND wins≥2/3 |

## Suite ladder (do not treat as baseline)

| Suite | Hypothesis | Outcome |
|-------|------------|---------|
| v0 | word, limited synonym | UNRESOLVED |
| **v0b** | **full synonym-in-train** | **PASS — BASELINE** |
| v0c | true lexical cold-start | UNRESOLVED |
| v0d | hybrid partial synonym | FAIL |
| v0e | hybrid + char encoder | UNRESOLVED |

## Kernel (shared, not order-only)

```
code/kernel/  program, validate, executor, audit, scorecard
```

## Run baseline

```bash
python code/packages/order/eval/suite_order_v0b.py
```

## Rules after freeze

1. Do not change v0b schema/metric to “force PASS” on harder lexical regimes.
2. Harder lexical (v0c–v0e) stays documented as FAIL/UNRESOLVED — open problems.
3. Next product step = thin real connector **or** new pack; regression must still pass v0b.
4. Ticket / other domains = new packs under `code/packages/`, same kernel.

## Connector (thin, post-freeze)

| Item | Value |
|------|--------|
| File connector | `connectors/file_store.py` |
| Gate suite | `eval/suite_order_conn_v1.py` |
| Status | **PASS** (policy+CRUD on disk + v0b metrics still PASS) |

```bash
python code/packages/order/eval/suite_order_conn_v1.py
# optional full v0b retrain gate:
ORDER_CONN_RERUN_V0B=1 python code/packages/order/eval/suite_order_conn_v1.py
```

| CSV / Sheet connector | `connectors/csv_store.py` |
| Gate suite | `eval/suite_order_conn_v2.py` |
| Status | **PASS** (CSV policy+CRUD + v0b PASS) |

```bash
python code/packages/order/eval/suite_order_conn_v2.py
```
