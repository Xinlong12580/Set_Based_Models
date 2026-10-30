# Archived casual tests

These files preserve earlier study configurations and output formats. They are
not dependencies of the current ERL runner and should not be used as the default
workflow. Nothing in this directory is executed by importing this package.

- Older ERL runners: `run_ERL.py`, `run_ERL_reference_anomalous.py`, and all
  `run_ERL_Gini*.py` variants here. Their replacement is
  `../train_ERL.py`, including its `--event-only` mode.
- `train_ERL_pruned.py`: former trainer that selected final rules on training
  data. The maintained `../train_ERL.py` saves unpruned candidates;
  `../train_ERL_validation.py` coordinates training and validation selection.
- Older EDT runners: object-only, event/object, and quantitative variants. The
  maintained event-only entry point is `../run_EDT_event_only.py`; EDT remains
  an optional extension.
- `_nested_rule_pruning.py`: compatibility implementation of an earlier
  pruning policy. Production pruning is in the core model package.
- `ERL_GINI_NESTED.md`, `ERL_GINI_PENALIZED.md`, and `EVENT_ONLY.md`: historical
  usage notes; their settings may differ from the current runner.
- `EDT.log`, `ERL.log`, `data/`, and `__pycache__/`: old outputs, example data,
  and generated caches retained without deletion.
- `make_dataset.py`: old external-repository-dependent data preparation script.
  It contains top-level I/O; do not import or execute it to inspect the archive.

Archived CLI runners with argument parsers can still be invoked from the
repository root, for example:

```bash
python casual_tests/stale/run_ERL_Gini_nested_event_only.py --help
```

Repository-root discovery and internal imports were adjusted for relocation;
their search settings and algorithms were not changed. Legacy regression tests
import these runners via `casual_tests.stale` to keep the older behavior covered.
