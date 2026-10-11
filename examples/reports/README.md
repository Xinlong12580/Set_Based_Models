# Small quantitative report examples

These are **hand-authored synthetic reporting fixtures**, not measured physics
results, model-performance benchmarks, or the output of a production fit.
They follow the current JSON structure; their counts, gains, penalties, ranks,
and pruning examples are checked by `tests/test_report_documents.py`.
No dataset files or credentials are included. The `synthetic/*.pt` paths are
illustrative identifiers, not downloadable inputs.

- [ERL example](erl_synthetic.json): same-object conditions, priority completion,
  score ranking, threshold simplification and logical pruning.
- [EDT example](edt_synthetic.json): event-count cuts, positive/negative branches,
  leaf ranking and the default predicted-positive selection.
- [Format reference](../../docs/REPORT_FORMAT.md) and
  [JSON Schema](../../schemas/quantitative_report.schema.json).

Both examples use 100 mixture events and 100 reference events. In this toy
label convention only, label 1 denotes an ordinary component and label 2 an
injected component; the mixture contains 80 of the former and 20 of the latter.
These diagnostic labels do not define the model's training class.

## ERL: expected interpretation

The optimization score is Gini gain minus reference acceptance, with penalty 1.
Priority is `type_id`, then `pt:>`, and full completion is required for final
retention. Similarity and rate filtering are disabled; logical improvement is 0.

| Rule ID / global rank | Condition on the same object | Mixture/reference passes | Score | Final decision |
| --- | --- | --- | --- | --- |
| 0 / 1 | Photon, PT > 25 GeV | 20 / 0 | 0.055556 | Retained |
| 1 / 2 | Photon, PT > 25 AND PT > 35 GeV | 15 / 0 | 0.040541 | Stricter rule fails improvement against rule 0 |
| 2 / 3 | Photon | 30 / 4 | 0.019887 | Priority incomplete: missing PT cut |

Expected explanations:

- Rule 0 has 20% mixture acceptance and no observed reference passes. Its
  reference rate estimate is zero, but 100 reference events cannot establish
  a zero physical rate. No hard rate ceiling was enabled.
- Rule 1's two PT bounds simplify to PT > 35. Original depth is 3, effective
  depth is 2, and its ordered history remains intact. It accepts fewer mixture
  events without reducing observed reference acceptance. Its score change is
  approximately -0.015015, failing the required strictly positive improvement.
  A threshold of -0.02 would let it pass this logical check, holding everything
  else fixed; other enabled filters could still remove it.
- Rule 2 has the **largest raw Gini** (0.059887), but its 4% reference acceptance
  incurs a 0.04 penalty. It therefore ranks below rule 0 by the actual score.
  Completion rejects it before it can act as a logical-pruning witness.
- `retained_after_similarity_pruning = false` does not mean similarity removed
  either rejected rule: this runner uses that field as a final-retention alias.
  The `pruning.status` values state the real reasons.
- Rule 2's training-class purity is 30/(30+4), approximately 88.2%. Its injected
  fraction *within accepted mixture events* is 20/30, approximately 66.7%.
  These are different quantities. The latter is computable here only because
  the fixture explicitly supplies the toy label meanings.

## EDT: expected interpretation

The tree's positive root test is `n_photons_pt20 > 0.5`. For an integer count,
this means at least one photon above 20 GeV. The negative branch means no such
photons, **not** the same positive test shown in the stored cut record.

| Leaf ID / rank | Selected condition | Mixture/reference passes | Leaf mixture fraction | Default prediction |
| --- | --- | --- | --- | --- |
| 1 / 1 | Count >= 1 | 30 / 4 | 0.882353 | Positive |
| 2 / 2 | Count <= 0 | 70 / 96 | 0.421687 | Negative |

Expected explanations:

- Leaves partition both inputs: their counts sum to 100 per class.
- At threshold 0.5, only leaf 1 is predicted positive.
  `summary.default_prediction` therefore accepts 30 mixture and 4 reference
  events. An OR of both leaves accepts all events, not just the enriched leaf.
- The shared root split gain is 0.059887. It is a node-level separation gain;
  it is not a leaf score or an ERL penalized score.
- The count feature has meaningful zeros and no validity guard. This example
  does not demonstrate padding protection for angles or other undefined values.
- Comparing these fixtures does not establish EDT/ERL relative performance:
  they illustrate different input descriptions and reporting semantics.

## Example analysis prompts

- Which candidates survive, and which recorded filter removed each alternative?
- Why does the highest raw-Gini candidate not have the highest score?
- Does a zero reference pass count justify a zero trigger rate?
- How does an EDT negative branch change the interpretation of its stored cut?
- What information is missing before calling these selections validated triggers?

A correct answer should identify the toy nature of the reports, distinguish
mixture acceptance from process efficiency, and avoid claiming independent
validation or an OR acceptance from ERL marginal counts alone.

## Checks and plots

From the repository root, with the usual Python environment:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_report_documents.py -v

python scripts/plot_acceptance_report.py examples/reports/erl_synthetic.json \
  --output reports/ERL_synthetic_example.png

python scripts/plot_acceptance_report.py examples/reports/edt_synthetic.json \
  --direction all --output reports/EDT_synthetic_example.png
```

Arithmetic and link checks use the standard library. Schema checks additionally
use the optional `jsonschema` package and are reported as skipped if it is absent;
it is not added as a model/runtime dependency. Plotting uses the existing
Matplotlib dependencies. No conversion or fitting is needed for these examples.
