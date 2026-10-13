# Domain packs + kernel

```
code/kernel/          # program, validate, executor, audit, scorecard
code/packages/order/  # first domain pack
  schema/tools.json
  connectors/mock_store.py
  eval/suite_order_v0.py
```

Ticket (and others) = later packs reusing the same kernel.

```bash
python code/packages/order/eval/suite_order_v0.py
```
