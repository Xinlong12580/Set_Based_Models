# Particle-pair tensors

Pair generation is optional and independent of ERL. It adds a separate tensor
to the same converted `.pt` file without changing `obj`, `event_features`, or
the saved native event information. The core ERL model needs no pair-specific
changes; the flexible runner can select this tensor with `--pair-tensor`.

## Conversion

From the repository root, in the usual conversion environment:

```bash
python -m raw_event_conversion \
  --input /path/to/raw/sample.parquet \
  --output data/sample_with_pairs.pt \
  --label 2 --max-events 10000 --with-pair-tensor
```

Local files and remote file URLs work as for the existing converter. Existing
outputs are overwritten with a warning; use `--no-overwrite` to refuse this.
Enable the option for both signal and background before mixing them. The
existing mixture script preserves the pair tensors, masks, and metadata.

Python entry points:

```python
from raw_event_conversion import build_pair_tensor, convert_raw_parquet_to_pt

pairs, presence_mask = build_pair_tensor(raw_records)  # Awkward raw records
convert_raw_parquet_to_pt(
    "sample.parquet", "sample.pt", label=2, include_pair_tensor=True,
)
```

## Saved data and ordering

| Key | Contents |
| --- | --- |
| `pair_obj` | Float32 tensor `[events, 231, 68]` |
| `pair_slot_mask` | Boolean tensor `[events, 231]`; true when both members exist |
| `pair_obj_columns` | Ordered feature names |
| `pair_obj_feature_schema` | Feature kinds, units, categories, and validity links |
| `pair_object_indices` | List of `[i, j]` source indices into `obj`, one per pair slot |

- Use the same retained objects as `obj`: 10 PUPPI jets, 4 tight muons,
  4 electrons, and 4 tight photons. MET is not a pair member.
- Each collection is ranked by decreasing finite positive PT, with stable
  producer-order ties. Objects beyond these caps do not appear in pairs;
  event summaries still use the full collections.
- Enumerate every source-slot combination `i < j`: 231 possible slots,
  with no self-pairs or reversed duplicates.
- Member ordering is **type first**, then PT within a type. Type IDs are
  `0 = jet`, `1 = muon`, `2 = electron`, `3 = photon`. For example, a jet-photon
  pair always has the jet as member 1; for two photons, member 1 has higher PT.
- If an event retains `k` objects, its mask has `k*(k-1)/2` true entries.
  Fewer than two objects gives no valid pairs. All padded pair rows are zero.
- Pair calculations use the raw values before float32 rounding. Missing
  coordinates retain false validity rather than becoming physical zero angles.

Pair-enabled output has conversion schema version 8; ordinary conversion
remains version 7. Pair definitions and mass hypotheses are recorded under
`conversion_metadata["pair_tensor"]`.

**Memory:** the pair tensor alone uses about 599 MiB for 10,000 events. Loading
both classes and ERL's search state needs additional memory. This quadratic
increase in slots is why pair generation is opt-in.

## Features

Each member copies all 20 original object columns, suffixed `_1` or `_2`:

- `pt`, `eta`, `phi`, `type_id`.
- `abs_d0`, `isolation_rho_corr`, `mass`, `n_charged`, `n_neutrals`,
  `charged_fraction`, `ehad_over_eem`, `charge`, with their original validity
  flags and type applicability.
- Two additional coordinate guards per member: `eta_valid_1`, `phi_valid_1`,
  `eta_valid_2`, and `phi_valid_2`.

The remaining 24 columns are 12 pair properties and their `<name>_valid` flags:

| Pair property | Definition |
| --- | --- |
| `min_pt` | Smaller member PT, GeV |
| `sum_pt` | Scalar `pt_1 + pt_2`, GeV |
| `pair_pt` | Magnitude of the vector transverse-momentum sum, GeV |
| `pt_balance` | Smaller PT / larger PT |
| `pair_mass` | Invariant mass of the two four-vectors, GeV |
| `abs_delta_eta` | Absolute pseudorapidity separation |
| `abs_delta_phi` | Absolute wrapped azimuthal separation in `[0, pi]`, radians |
| `delta_r` | `sqrt(abs_delta_eta**2 + abs_delta_phi**2)` |
| `max_isolation_rho_corr` | Larger corrected isolation; requires two valid measurements |
| `min_abs_d0` | Smaller absolute D0, mm; requires two valid measurements |
| `opposite_sign` | Opposite charges; valid only for two electrons/muons with known charges `+/-1` |
| `opposite_eta_hemispheres` | Whether the two eta values have strictly opposite signs |

Mass uses valid stored jet `Mass`, massless photons, and fixed muon/electron
masses of 0.1056583755/0.00051099895 GeV from the
[PDG muon](https://pdg.lbl.gov/2025/listings/rpp2025-list-muon.pdf) and
[electron](https://pdg.lbl.gov/2025/listings/rpp2025-list-electron.pdf) listings.
Missing jet mass makes `pair_mass` invalid; it is not silently set to zero mass.
Undefined or nonfinite derived results are zero-filled with a false guard.
Measured zeros can be valid. Finite member values exceeding float32 storage
raise an error rather than silently overflowing.

To add pair properties later, extend `PAIR_FEATURES` and their calculations in
`pairs.py`. Column names, guard entries, and tensor width follow that mapping.

## Core ERL interface

Pair slots are ordinary ERL object slots: all cuts in a rule apply to the **same
pair**, and an event passes when enough pairs satisfy those cuts. Pass the
saved mask and schema so padded slots and invalid measurements cannot pass.
The flexible runner supports both representations; see the command below.

```python
import torch
from set_based_models import ExistentialRuleList, ObservableSpec, TPRFPRRuleScore

signal = torch.load("anomalous.pt", map_location="cpu", weights_only=True)
background = torch.load("reference.pt", map_location="cpu", weights_only=True)
schema = signal["pair_obj_feature_schema"]
if schema != background["pair_obj_feature_schema"]:
    raise ValueError("Pair schemas must match.")
specs = tuple(
    ObservableSpec(item["name"], item["kind"],
                   categories=item.get("categories"),
                   validity_column=item.get("validity_column"))
    for item in schema
)
model = ExistentialRuleList(
    specs,
    scoring_function=TPRFPRRuleScore(),  # Example only; choose your usual score.
    object_cut_priority=["type_id_1", "type_id_2"],
    max_rule_depth=4,
    min_signal_efficiency=0.01,
)
model.fit(
    signal["pair_obj"].numpy(), background["pair_obj"].numpy(),
    signal_slot_mask=signal["pair_slot_mask"].numpy(),
    background_slot_mask=background["pair_slot_mask"].numpy(),
)
```

Type priority controls expansion order, not, by default, a requirement that every
returned rule contain both types: incomplete prefixes remain eligible.
`--require-complete-priority` can make the whole list compulsory for final
retention without removing incomplete prefixes from search. After two photon-type
cuts, `min_pt > 30` describes a pair of distinct
photons both above 30 GeV. Separate `pt_1`/`pt_2` cuts allow asymmetric thresholds.
Two accepted pair slots may share a particle, so requiring two pairs is **not**
equivalent to requiring four distinct particles.

Priorities can repeat a feature and specify numerical directions. For example,
`--object-cut-priority type_id_1 type_id_2 'min_pt:>' 'min_pt:<='` requests both
member types followed by lower and upper bounds on the smaller member momentum.
Python accepts strings or `(feature, direction)` pairs. Thresholds are optimized,
each bound consumes a separate step, and both must survive normal search and
pruning. Add `--require-complete-priority` to retain only completed sequences;
it checks original cut histories, not simplified displayed conditions.

## Flexible ERL runner

After converting both classes with `--with-pair-tensor` and mixing them:

```bash
python casual_tests/run_ERL_Gini_nested.py \
  --signal data/example_mixture/anomalous.pt \
  --background data/example_mixture/reference.pt \
  --pair-tensor --object-only \
  --progress \
  --object-cut-priority type_id_1 type_id_2 \
  --reference-penalty 10 \
  --stricter-score-improvement 0.001 \
  --beam-similarity-filter --final-similarity-filter \
  --signal-similarity-threshold 0.9 --background-similarity-threshold 0.9 \
  --max-trigger-rate-khz 40 --input-rate-mhz 40 \
  --report reports/ERL_example_pairs.txt --overwrite-report
```

- `--pair-tensor` (alias `--use-pair-tensor`) replaces single-particle `obj`
  with `pair_obj`; it does not concatenate them.
- Remove `--object-only` to combine pair cuts with event features.
- Without the pair switch, single-particle behavior is unchanged. Combining
  `--pair-tensor` with `--event-only` is an error.
- `--basic-object-features` selects PT, eta, phi and type ID for both members,
  plus their required coordinate guards (12 columns). Derived pair quantities
  and other member properties are excluded in that mode.
- Type priority is optional and explicitly specified, not automatically added.
  The current runner depth of four allows two type cuts followed by two
  property cuts, such as separate asymmetric PT thresholds. Longer sequences
  require increasing the constructor's search depth or using the core API.
- Gini scoring, logical/similarity pruning and the rate filter are unchanged;
  their acceptances count events, not pairs. Validity flags remain guard-only
  unless `--allow-validity-cuts` is supplied.
- Logs identify pair slots; reports record `object_tensor_key`, the selected
  pair schema, and mask policy. Automatically named reports include `_pairs`
  to avoid collision with single-particle runs and remain plot-compatible.
- `--progress` displays live search counters and elapsed time on stderr. It
  works for pairs, single particles and event-only inputs, with no new dependency.
  The progress bar tracks beam parents within the current depth, not a whole-fit
  percentage. Final pruning is outside this search display.
- The example penalty and pruning values are settings you can tune, not
  guaranteed optimal values. Pair search uses substantially more memory than
  single-particle search; the saved float32 inputs are widened for validation.
- ERL automatically packs candidate masks into bits, reuses them for duplicate
  detection and similarity, and unpacks only the current parent for expansion.
  Completed layers release slot masks; discarded expansions do not persist.
  This saves candidate-state memory without changing rule discovery or the
  saved `.pt` format. No extra option or reconversion is needed; the full run
  still needs memory for inputs, validity arrays and temporary calculations.

Run the isolated synthetic checks without production conversion or training:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest raw_event_conversion.tests.test_pairs -v
```
