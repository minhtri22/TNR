# Domain packs + kernel

```
code/kernel/                 # shared program → validate → execute → audit → scorecard
code/packages/order/         # first domain pack (FROZEN baseline = v0b)
  BASELINE.md
  schema/tools.json
  connectors/mock_store.py
  eval/
    suite_order_v0.py        # UNRESOLVED
    suite_order_v0b.py       # PASS — BASELINE
    suite_order_v0c.py       # UNRESOLVED cold-start
    suite_order_v0d.py       # FAIL hybrid
    suite_order_v0e.py       # UNRESOLVED hybrid+char
```

## Baseline (frozen)

```bash
python code/packages/order/eval/suite_order_v0b.py
```

See `order/BASELINE.md`. Ticket and other domains = additional packs, same kernel.
