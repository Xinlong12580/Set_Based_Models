# Print an ERL model

`scripts/print_erl_model.py` prints the important information in a core model
JSON without input tensors, fitting, inference or changes to the model. It uses
NumPy and the core package, with no PyTorch, Matplotlib or EDT dependency.

```bash
python scripts/print_erl_model.py \
  training_outputs/ggHgammagamma_scan.validation.model.json --top 10
```

The default prints the summary and ten highest-scoring retained rules. Before
final pruning, it prints the highest-scoring candidates instead. An empty final
selection stays empty. Each entry includes the rule ID, score, pruning status,
original depth, simplified cut count, readable predicates, accepted event
counts and weighted acceptance for the mixture and reference.

## Controls

| Option | Meaning |
|---|---|
| `--verbosity 0` or `-v 0` | Model summary only |
| `--verbosity 1` | Summary and top rules; default |
| `--verbosity 2` | Also show prefix statistics, accepted weights, pruning decisions and provenance |
| `--verbosity 3` | Also show full configuration, feature schemas and complete parent-ID lists |
| `--top N` | Show the first N rules; `0` shows the whole chosen pool |
| `--all-candidates` | Include candidates removed during final pruning |
| `--rule-ids ID ...` | Inspect specific candidate IDs, whether retained or discarded |

`--all-candidates` and `--rule-ids` are alternatives. Selection is always ordered
by stored score descending, then rule ID. Displayed rank is within that pool;
`global_rank` is the position among all candidates in the model. Neither is an
estimate of physics importance or a ranking by true signal efficiency.

```bash
# Inspect a particular rule and its cut-by-cut history.
python scripts/print_erl_model.py MODEL.json --rule-ids 548 --verbosity 2

# Inspect the top candidates, including discarded ones.
python scripts/print_erl_model.py MODEL.json --all-candidates --top 20

# Save the text using ordinary shell redirection.
python scripts/print_erl_model.py MODEL.json --top 20 -v 2 > model_summary.txt
```

Counts are unweighted event counts; acceptance fractions use the stored sample
weights. The input called `signal` by the core is labeled **mixture** in this
printer. No process labels, physical rates or true signal purity are inferred.
The stored statistics-source marker is printed as saved: `recalculated` alone
does not establish validation independence. Scores are read without executing
or assuming a scoring function. An unrecalculated merged pool prints a reminder
that scores from different sources may not be comparable.

Predicates use the core's logical normalization and rendering. Automatic True
validity guards, same-slot conjunctions, category labels and minimum object
counts remain visible. Prefix histories retain their stored cut order and
statistics. Dense Jaccard matrices are summarized by shape and provenance;
their numbers are never printed or reconstructed as NumPy arrays. Loading a
JSON still parses the file, including its stored matrices.

## Python use

```python
from scripts.print_erl_model import print_model

print_model(model, top=5, verbosity=2)
print_model("models/erl.validation.model.json", top=10)
```

The function also accepts a core model dictionary. Pass `stream=handle` to write
to an open text file. Existing model instances and dictionaries are unchanged.
Runner reports and MDS layouts are separate formats; use the core model JSON.
