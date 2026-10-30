# Optional EDT extension

`set_based_edt` contains the existential decision tree: fitting, nodes, routing,
prediction, and path evaluation. It depends on the shared core; ERL and its
conversion/reporting/plotting workflow do not depend on this extension.

## Install

From the repository root, install the core first and EDT only if needed:

```bash
python -m pip install -e .
python -m pip install -e extensions/edt
```

For source-only use without installation:

```bash
export PYTHONPATH="$PWD/src:$PWD/extensions/edt/src${PYTHONPATH:+:$PYTHONPATH}"
```

The extension itself needs only the core and its NumPy dependency. The example
PT-file runners additionally need PyTorch and their existing study dependencies.

## Imports and compatibility

Prefer explicit extension imports in new code:

```python
from set_based_models import ObservableSpec, gini_gain
from set_based_edt import ExistentialDecisionTree

tree = ExistentialDecisionTree(
    (ObservableSpec("pt", "continuous"),), gain_function=gini_gain,
)
```

With EDT installed, historical explicit imports from `set_based_models`,
`set_based_models.decision_tree`, and `set_based_dt` remain compatibility aliases.
They refer to the same classes, not separate implementations. Without EDT,
requesting those tree APIs raises an import error; importing ERL does not.
`from set_based_models import *` intentionally exports only core/ERL symbols,
so wildcard imports never require the extension.

The algorithms, defaults, masks, scoring arithmetic, and report format are
unchanged. Shared partition statistics and split gains now live in
`set_based_models.core.statistics` and `set_based_models.core.objectives`.
EDT-specific report assembly lives in `casual_tests/_edt_reporting.py` and is
loaded only when an EDT runner is called. Reading/plotting an existing EDT JSON
report does not require EDT.

## Run and test

The existing EDT runner commands are unchanged after installing the extension:

```bash
python casual_tests/run_EDT_event_only.py \
  --signal path/to/anomalous.pt --background path/to/reference.pt \
  --report reports/EDT_event_only.txt

PYTHONPATH=src:extensions/edt/src python -m pytest -q
```

Without EDT on the import path, EDT-dependent tests skip while core/ERL tests
remain runnable. `tests/test_optional_edt.py` also checks an isolated checkout
without the extension, including all four flexible-ERL input modes.

To uninstall EDT while retaining ERL:

```bash
python -m pip uninstall set-based-models-edt
```

Also remove `extensions/edt/src` from any manually configured `PYTHONPATH`.
The ERL-only source checkout needs `src/`, not `extensions/edt/`.
