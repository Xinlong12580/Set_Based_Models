# Direct raw Parquet to event and object tensors

This small module is independent of `data_conversion` and `aidascoutrepo`.
It reads the flat raw fields such as `L1T_Electron_PT`, computes observables,
and saves a `.pt` file. There is no PF regionization, event shuffling, or
empty-object/PF event filter. Event features use the full collections without
object caps or precision downcasts. The separate `obj` tensor uses the old
float32, PT-ranked, top-k layout extended with object properties and validity
flags, excluding the MET slot.

The saved schemas link measurements to their Boolean validity columns. The
current flexible ERL runner reads these links and automatically guards each
measurement cut; linked flags are guard-only by default. If constructing core
inputs manually, preserve these links in `ObservableSpec.validity_column`.
Zero filling alone does not provide this protection.

Optionally add `--with-pair-tensor` to also save a separate particle-pair tensor,
its presence mask, and ERL-compatible feature schema. This does not replace the
original tensors. See [Particle-pair tensors](PAIRS.md) for features, ordering,
memory requirements, and ERL examples. The flexible ERL runner reads `obj` by
default; add `--pair-tensor` to select the saved pairs instead.

## Run

From the repository root in the `hep` environment:

```bash
python -m raw_event_conversion \
  --input /path/to/raw/minbias_part0.parquet /path/to/raw/minbias_part1.parquet \
  --output data/raw_event_features/minbias_10000.pt \
  --label 1 --max-events 10000
```

Supply **raw files**, not `data/regionized/...` files. Inputs are processed in
the supplied order; a local shell glob such as `/path/to/raw/*.parquet` also
works. Remote file URLs (`root://.../file.parquet`) use Awkward/fsspec and require
the optional `fsspec-xrootd` plugin, XRootD bindings, and appropriate access.
Remote sample-directory discovery is deliberately outside this core module.
Select trusted production files: no dataset-version filtering is applied.

`--max-events` is a total row cap, not a network-transfer limit. Selected columns
of each opened file are read before slicing; later files are not opened once
the cap is reached. If fewer rows exist, all available rows are saved.

## Output and compatibility

All default jet observables now use **`L1T_JetPuppiAK4`**, and all default
MET observables use **`L1T_PUPPIMET`**. Electron, tight-muon, and tight-photon
collections are unchanged. This is a source switch, not another application
of candidate PUPPI weights: the supplied PUPPI jet PT and PUPPI MET values
are used as stored. There is no fallback to `L1T_JetAK4` or `L1T_MET` if a
required PUPPI field is missing.

The change applies to every dependent default feature: jet multiplicities,
PT sums/ranks/ratios, ranked eta, same- and cross-type pair angles, total
object PT, MET, total-PT/MET, and all object-MET angles/validity flags.
The PUPPI switch preserved names, ordering, and tensor width, but changed
values and source definitions. Charged raw-PF PT features appended eight more
columns without reordering the existing 139. Impact-parameter features now
append another 38, preserving those 147. Ranked lepton impact parameters append
eight more, preserving those 185. `conversion_metadata.schema_version` is 7
because conversion saves real objects with additional properties and validity flags;
pair-enabled output uses version 8.
`object_pt_columns` and `met_columns` record the default input mappings.
Per-feature `required_columns` also identify the sources. Reconvert both
classes and remake mixtures; existing `.pt` files are not updated or relabeled.
The mixer and event-only runners reject mismatching feature definitions.
Older conversion pipelines and their object layouts are not changed.

- `event_features [N,193]`, float64 by default: PT > 0 multiplicities followed by
  electron thresholds 10/20/30/50, muon 5/10/20/40, photon 10/20/30/50, and jet
  20/30/50/100 GeV. The first four names are `n_electrons`, `n_muons`,
  `n_photons`, `n_jets`. All comparisons are strict `>` and exclude missing or
  nonfinite PT values. No extra eta, isolation, or quality cuts are added to the
  producer's existing electron, tight-muon, tight-photon, and PUPPI AK4-jet collections.
  The original 20 multiplicity columns retain their ordering and ordinal schema.
  Four continuous columns are appended: `sum_pt_electrons`, `sum_pt_muons`,
  `sum_pt_photons`, and `sum_pt_jets`, in GeV. Each is the scalar sum of **every
  finite, positive PT in the corresponding raw object collection**, including
  objects below the multiplicity thresholds, with no slot cap. Empty collections
  give zero; missing, nonfinite, and nonpositive PT entries contribute zero.
  These sums never use `L1T_ScalarHT_HT` and are not vector sums or MET.
  Count-threshold overrides do not change the PT sums. The tensor is float64
  to preserve fractional sums; multiplicity columns remain integer-valued.
  After the original 24 columns, five more are appended:
  - `sum_pt_all_objects` (GeV): sum of the electron, muon, photon and jet PT sums,
    computed from their raw collections, never from ScalarHT. No overlap removal
    between collections is performed, so this is a collection-level activity sum.
  - `met` (GeV): the supplied `L1T_PUPPIMET_MET` magnitude, not recomputed from objects.
  - `met_valid` (Boolean): true for a present, finite, nonnegative MET value.
    Zero MET is valid; empty/missing entries, negative values and nonfinite values
    use a zero placeholder and false validity. More than one value per event is an error.
  - `sum_pt_over_met` (dimensionless): `sum_pt_all_objects / met`, valid only for
    positive valid MET and a finite result. Undefined or overflowing ratios use
    a zero placeholder, not infinity or an arbitrary large sentinel.
  - `sum_pt_over_met_valid` (Boolean): true exactly when that ratio is defined.
    Zero activity with positive MET has a valid zero ratio; zero MET does not.

  All five are available as event inputs. Schema-aware measurement cuts require
  the linked validity flag automatically, so invalid zero placeholders cannot
  satisfy a MET/ratio cut. Manual or legacy loaders that discard validity links
  must supply this requirement explicitly.
  Add `--without-met-features` (Python: `include_met_features=False`) to omit
  MET, its ratio, object-MET angles, and all their flags; other features remain (179 columns).

  Next, eight continuous features in GeV are appended, preserving the first
  29 columns in the default configuration:

  - `leading_pt_electrons`, `subleading_pt_electrons`
  - `leading_pt_muons`, `subleading_pt_muons`
  - `leading_pt_photons`, `subleading_pt_photons`
  - `leading_pt_jets`, `subleading_pt_jets`

  These are the largest and second-largest **finite, positive PT values** in
  each full raw collection, not the first two entries in producer order. No
  object cap or count-threshold cut is applied; equal-PT objects are distinct
  entries and can occupy both ranks. Fewer than one/two valid objects gives
  zero in the corresponding feature. For upper-bound cuts, require
  `n_<type> >= 1`/`>= 2` to avoid selecting missing-object zero placeholders.
  Positive lower bounds naturally enforce presence. For example,
  `leading_pt_photons > 30 AND subleading_pt_photons > 20` expresses an
  asymmetric double-photon condition directly.

  Next, two dimensionless ratios and two Boolean validity flags for each
  object type are appended, preserving the original 37 columns. For
  `<type>` = `electrons`, `muons`, `photons`, `jets`, in that order:

  - `leading_pt_fraction_<type>` = `leading_pt_<type> / sum_pt_<type>`: measures
    concentration **within that type**, not its share of all-object activity.
  - `leading_pt_fraction_<type>_valid`: true for a positive finite same-type
    sum and finite ratio; an empty collection is invalid. A single object has
    a valid fraction of 1.
  - `pt_balance_<type>` = `subleading_pt_<type> / leading_pt_<type>`: the bounded
    version of the pair ratio, with 1 for equal PT and smaller values for imbalance.
  - `pt_balance_<type>_valid`: requires at least two finite, positive PT values
    and a finite ratio; a single object has an invalid pair balance.

  Ratio columns are continuous, use only their own raw PT collection, and are
  unaffected by count thresholds. Defined ratios lie in [0, 1]. Undefined ratios
  use zero with a false flag, never NaN/infinity. The saved schema links each
  ratio to its flag for automatic guarding by schema-aware model inputs.
  No reciprocal leading/subleading ratio or cross-type composition ratio is added.

  Next, absolute pseudorapidity and Boolean validity flags are appended for
  the PT-leading and PT-subleading objects, preserving the original 53 columns.
  For `<type>` = `electrons`, `muons`, `photons`, `jets`, in that order:

  - `leading_abs_eta_<type>` and `leading_abs_eta_<type>_valid`
  - `subleading_abs_eta_<type>` and `subleading_abs_eta_<type>_valid`

  Each eta is taken from the **same object** that supplies the corresponding
  ranked PT, not from the largest absolute eta in the collection. Objects are
  ranked by finite positive PT alone; equal-PT ties retain producer order.
  Count thresholds do not affect this ranking. The raw eta fields are
  `L1T_Electron_Eta`, `L1T_MuonTight_Eta`, `L1T_PhotonTight_Eta`, and
  `L1T_JetPuppiAK4_Eta`; PT and eta arrays must have matching lengths per event.
  Missing objects or missing/nonfinite eta values use zero with a false flag.
  A real eta of zero has a true flag. Invalid eta does **not** promote another
  object into its rank or change multiplicities/PT sums. Schema-aware cuts
  automatically require the matching validity flag.
  No eta acceptance cut or signed eta/phi feature is added.

  Next, three pair separations and their Boolean validity flags are appended,
  preserving the original 69 columns. For `<type>` = `electrons`, `muons`,
  `photons`, `jets`, in that order:

  - `abs_delta_eta_<type>` and `abs_delta_eta_<type>_valid`:
    `abs(eta_leading - eta_subleading)`, using **signed** raw eta values.
  - `abs_delta_phi_<type>` and `abs_delta_phi_<type>_valid`:
    `abs(atan2(sin(phi_leading - phi_subleading), cos(phi_leading - phi_subleading)))`,
    in radians and bounded by [0, pi]. The shortest periodic separation is used:
    angles just below +pi and just above -pi are close, not almost 2pi apart.
  - `delta_r_<type>` and `delta_r_<type>_valid`:
    `hypot(abs_delta_eta_<type>, abs_delta_phi_<type>)`.

  The pair always comprises the two highest finite, positive PT objects of the
  **same type**, with producer-order tie breaking and no multiplicity-threshold
  cut. Invalid eta/phi does not promote a third object. PT, eta, and phi arrays
  must have matching object counts per event. Raw phi columns use the same
  collection prefixes as eta, ending in `_Phi`; values are assumed to be radians.
  Delta eta requires two finite eta values, delta phi requires two finite phi
  values, and delta R requires both. A missing pair or undefined/nonfinite result
  is stored as zero with a false validity flag; a physical zero separation has
  a true flag. Schema-aware angular cuts automatically require the corresponding
  validity flag.

  Next, the same three angular separations and their validity flags are
  appended for the **leading objects of different types**, preserving the first
  93 columns. The six unordered pairs, in output order, are:

  - `electrons_muons`
  - `electrons_photons`
  - `electrons_jets`
  - `muons_photons`
  - `muons_jets`
  - `photons_jets`

  For each `<pair>`, the names are `abs_delta_eta_leading_<pair>`,
  `abs_delta_phi_leading_<pair>`, and `delta_r_leading_<pair>`, each followed by
  its `<name>_valid` flag. This adds 18 continuous values and 18 Boolean flags.
  Definitions, units, phi wrapping, missing-value handling, and independent
  eta/phi validity follow the same-type pair features above. Each side is
  selected independently by finite positive PT, with producer-order tie breaking;
  only **one object per type** is needed. These are not minimum-distance pairings,
  and objects shared across collections are not removed. No extra PT/eta cut
  is applied and invalid angular coordinates never promote another object.
  No new raw columns are required beyond the existing PT/Eta/Phi collections.
  The exported `cross_type_angular_features(first_type, second_type)` factory
  accepts any two distinct supported types; reversing its arguments gives the
  same canonical names and definitions rather than a second copy of the pair.

  Next, five absolute object-MET azimuthal separations and their Boolean
  validity flags are appended, preserving the first 129 columns:

  - `delta_phi_leading_jet_met`
  - `min_delta_phi_jets_met`
  - `delta_phi_leading_photon_met`
  - `delta_phi_leading_electron_met`
  - `delta_phi_leading_muon_met`

  Each value is immediately followed by `<name>_valid`. Despite the shorter
  `delta_phi` names, these are **absolute wrapped** angles in [0, pi], in radians,
  using the same periodic formula as pair delta phi. They use native
  `L1T_PUPPIMET_MET` and `L1T_PUPPIMET_Phi`, not MET recomputed from the object collections.
  MET magnitude and phi must each contain one value or be empty per event.
  Direction requires finite **positive** MET and finite MET phi; zero MET still
  has `met_valid=True` but all its angular features are invalid.

  Leading objects are selected by finite positive PT alone, with producer-order
  tie breaking. Missing/nonfinite leading phi does not promote another object.
  `min_delta_phi_jets_met` instead takes the minimum over the **full raw jet
  collection**, with finite PT > 0 and finite phi, without a jet cap or a further
  threshold. Consequently even a soft jet can determine this minimum. Invalid
  jet phi entries are excluded from that minimum. No eligible jet, undefined MET
  direction, or an invalid leading object gives zero plus a false validity flag;
  real zero alignment is valid. Each angle's schema links its automatic guard.

  These features are enabled even with `--without-native-event-info`, but are
  omitted by `--without-met-features` (`include_met_features=False` in Python).
  The exported `met_angular_features()` factory returns their ten definitions.

  Next, eight charged raw-PF features are appended, preserving the first
  139 columns (or 125 when MET features are disabled). Each count is followed
  by its scalar PT sum:

  | PT selection (GeV) | Multiplicity | Scalar PT sum (GeV) |
  | --- | --- | --- |
  | PT > 2 | `n_charged_pf_pt_gt2` | `sum_pt_charged_pf_pt_gt2` |
  | 2 < PT < 5 | `n_charged_pf_pt_gt2_lt5` | `sum_pt_charged_pf_pt_gt2_lt5` |
  | 2 < PT < 10 | `n_charged_pf_pt_gt2_lt10` | `sum_pt_charged_pf_pt_gt2_lt10` |
  | PT > 10 | `n_charged_pf_pt_gt10` | `sum_pt_charged_pf_pt_gt10` |

  These use **unweighted** `L1T_PFPart_PT` and finite, nonzero
  `L1T_PFPart_Charge`, not `L1T_PUPPIPart_PT`, PID, truth pileup labels or
  PUPPI weights. Both positive and negative charges count. All PT boundaries
  are strict: exactly 2 GeV is excluded everywhere, and exactly 10 GeV is
  included only in PT > 2. Windows intentionally overlap; they are not a
  disjoint partition. Multiplicities have ordinal/count schema and sums have
  continuous/GeV schema. Missing/nonfinite PT or charge, neutral candidates,
  and zero padding are excluded. Empty collections give zero. PT and charge
  must have matching candidate counts per event, or conversion fails.

  **There is no vertex association, eta acceptance, or extra quality cut.**
  These describe all stored charged PF candidates and can be pileup-sensitive;
  they must not be interpreted as primary-vertex track counts. No additional
  candidate cap is applied, but any producer truncation (the documented L1 PF
  arrays have 1000 fixed slots) cannot be undone. Conversion promotes stored
  PT to float64 for sums but cannot recover producer precision or missing
  candidates. Only the PT/Charge PF columns are loaded for these features.
  Missing either is an error, not an assumed zero count. Object threshold
  overrides and the MET/native-info flags do not disable or redefine them.

  To add another window, use the exported `charged_pf_bin_features(pt_min,
  pt_max=None)` factory in `extra_features`; for example,
  `extra_features=charged_pf_bin_features(2, 20)`. It returns a count and a
  scalar sum, with the strict window and source columns recorded in schema.

  Next, 38 charged-PF impact-parameter features are appended, preserving
  the first 147 columns (133 without MET). All require finite unweighted
  PT > 2 GeV and finite nonzero charge. There is no vertex-association cut.

  - Six **absolute D0** bins in mm: `[0,0.1)`, `[0.1,0.5)`, `[0.5,1)`,
    `[1,5)`, `[5,10)`, `[10,infinity)`.
  - Six **absolute significance** bins: `[0,2)`, `[2,3)`, `[3,5)`,
    `[5,10)`, `[10,20)`, `[20,infinity)`.
  - Three cumulative absolute-D0 selections: `> 0.5`, `> 1`, `> 5` mm.
  - Three cumulative absolute-significance selections: `> 3`, `> 5`, `> 10`.

  Each selection supplies an ordinal multiplicity followed by a continuous
  scalar PT sum in GeV. The 36 names follow these patterns:

  ```text
  n_charged_pf_pt_gt2_abs_dxy_mm_ge0_lt0p1
  sum_pt_charged_pf_pt_gt2_abs_dxy_mm_ge0_lt0p1
  n_charged_pf_pt_gt2_abs_dxysig_ge3_lt5
  sum_pt_charged_pf_pt_gt2_abs_dxysig_ge3_lt5
  n_charged_pf_pt_gt2_abs_dxy_mm_gt0p5
  n_charged_pf_pt_gt2_abs_dxysig_gt5
  ```

  `ge` means inclusive lower bound, `lt` exclusive upper bound, and `gt`
  a strict cumulative threshold. Open-ended bins omit `lt`. Decimal points
  use `p` in names. Bin definitions, units, comparisons, source columns and
  invalid-value policy are saved in `event_feature_schema`. None, rather
  than non-JSON infinity, represents an open upper bound. The two sets of
  six bins separately partition valid eligible candidates; cumulative
  selections overlap them and each other. No PT-by-displacement cross-grid
  is constructed. Object threshold overrides do not change the PT > 2 cut.

  D0 comes directly from **`L1T_PFPart_D0` in mm**; no factor of 10 is applied.
  It is the producer's beam-spot/origin-referenced transverse impact
  parameter, not dxy recalculated with respect to a reconstructed PV.
  Significance is `abs(D0) / ErrorD0`, where **`L1T_PFPart_ErrorD0` is also
  in mm**, and uses no epsilon or clipping. It requires finite D0, finite
  strictly positive ErrorD0, and a finite ratio. Invalid candidates are
  excluded from the corresponding bins/cumulative cuts, never zero-filled
  into the first bin. D0=0 is valid if finite; zero significance is valid
  only with a positive valid uncertainty. The raw arrays must align with
  PF PT/Charge; missing required fields or length mismatches are errors.

  Two ordinal diagnostic counts finish the charged-PF block:

  - `n_charged_pf_pt_gt2_abs_dxy_mm_invalid`
  - `n_charged_pf_pt_gt2_abs_dxysig_invalid`

  Each counts eligible charged candidates whose corresponding measurement
  is invalid. Subtract it from `n_charged_pf_pt_gt2` to obtain the number
  with valid measurements. A conversion warning reports invalid counts.
  These columns describe measurement availability, not exotic physics. They are
  diagnostic counts, not Boolean validity guards for other measurements.

  **Important production limitation:** the checked COLLIDE2V v1.0 card
  does not execute impact-parameter smearing. In a 32-event check of one
  local minbias shard, all 1365 eligible charged PF candidates had zero
  ErrorD0. Such candidates have undefined significance, so their
  significance bins are zero and their invalid count is nonzero. This
  does not establish that there are no displaced particles. Do not claim
  realistic L1 displaced-track performance from these fields alone or
  substitute an arbitrary resolution for missing uncertainties.
  See [IMPACT_PARAMETERS.md](IMPACT_PARAMETERS.md) for the pinned source
  chain, units verification, reference point and sample-check scope.
  The exported `charged_pf_impact_features()` factory returns all 38 definitions.

  Finally, four absolute lepton impact parameters, each followed by its
  Boolean `<name>_valid` flag, are appended, preserving the first 185 columns:

  - `leading_abs_dxy_mm_electrons`
  - `subleading_abs_dxy_mm_electrons`
  - `leading_abs_dxy_mm_muons`
  - `subleading_abs_dxy_mm_muons`

  Values are `abs(D0)` in **mm**, from `L1T_Electron_D0` and
  `L1T_MuonTight_D0`, without rescaling. The name `dxy` denotes the stored
  producer D0: beam-spot/origin referenced, not recalculated relative to a
  reconstructed PV. Each belongs to the same object as its ranked PT.
  Ranking uses finite positive PT, with stable producer-order ties and no
  multiplicity-threshold cut. Missing/nonfinite D0 does **not** promote a
  lower-PT lepton. A missing rank or undefined D0 gives zero plus a false
  flag; a finite physical D0 of zero has a true flag. PT and D0 arrays must
  align; missing required columns or length mismatches are errors.

  Schema-aware cuts automatically require the corresponding flag, including
  upper-bound cuts that could otherwise accept invalid zero placeholders.
  No lepton significance or ErrorD0 input is added.
  The producer-resolution caveat above also applies to these D0 observables;
  their presence alone does not establish realistic displaced-trigger performance.
  The exported `ranked_abs_dxy_features("electrons")` or `("muons")` factory
  returns that type's four definitions. Reconvert both classes and remake
  mixtures to include the new columns with consistent schemas.
- `event_feature_columns` and `event_feature_schema`: names, types, units,
  required raw columns, and definitions.
- `native_event_info [N,10]`, float64: MET and phi, PUPPI MET and phi, native HT,
  and five producer-order rho entries. Names, units, and Boolean validity masks
  are saved alongside them. Undefined values use zero plus false validity;
  zero MET is valid but its direction is not. These quantities are preserved
  separately, **not automatically merged into the model inputs**; only the
  explicit `met` feature uses PUPPI MET, and PUPPI phi is used to calculate
  the object-MET angles described above. The separate native block retains
  its original order and names: `met`/`met_phi` there still mean `L1T_MET`,
  while `puppi_met`/`puppi_met_phi` mean `L1T_PUPPIMET`. Their saved raw-column
  schemas are explicit; the unweighted fields are diagnostics, not model inputs.
  Use `--without-native-event-info` for raw files lacking these fields. This
  flag does not disable event-level MET features; if PUPPI MET magnitude or phi is also
  absent, supply **both** `--without-native-event-info --without-met-features`.
- `label [N]`: the explicit process label supplied by the caller. It is not
  inferred from a filename and is not the runner's training-class assignment.
- `raw_source_file [N]` and `raw_source_row [N]`, int64: original file/row indices.
  The adjacent `<output-stem>.sources.json` stores the ordered input filenames.
- `obj [N,22,20]`, float32: the first four columns retain the old
  `[pt, eta, phi, type_id]` format, followed by eight properties and eight
  matching validity flags, with **no MET slot**. Slot groups and type IDs are:

  | Slots (zero-based) | Collection | Type ID |
  | --- | --- | --- |
  | 0–9 | `L1T_JetPuppiAK4` | 0 |
  | 10–13 | `L1T_MuonTight` | 1 |
  | 14–17 | `L1T_Electron` | 2 |
  | 18–21 | `L1T_PhotonTight` | 3 |

  Within each collection, select finite positive PT, stably sort descending,
  take the highest-PT objects up to its cap, and zero-pad unused slots.
  Coordinates follow the same selected objects; PT ties retain producer order.
  All fields of padded slots are zero; use `obj[..., 0] > 0` as the slot mask.
  Missing/nonfinite eta or phi
  is zero-filled, matching the old format, not a physical measurement; there
  is no automatic validity guard for these object angles.
  No additional eta, isolation, overlap-removal or threshold-grid cuts are applied.
  **Event counts/sums are calculated before top-k truncation**, so a count can
  exceed the corresponding number of stored object slots. `object_tensor`
  metadata records sources, caps, type IDs, sorting and padding. Event-level
  MET features and optional native MET information are unchanged.

  The eight new columns are appended in this order; the corresponding
  `*_valid` columns follow in the same order:

  | Column | Applicable types | Definition |
  | --- | --- | --- |
  | `abs_d0` | Muons, electrons | `abs(D0)` in mm; producer beam-spot/origin reference |
  | `isolation_rho_corr` | Muons, electrons, photons | Stored `IsolationVarRhoCorr`, without an extra cut or rescaling |
  | `mass` | Jets | Stored `Mass` in GeV |
  | `n_charged` | Jets | Stored `NCharged` |
  | `n_neutrals` | Jets | Stored `NNeutrals` |
  | `charged_fraction` | Jets | `NCharged / (NCharged + NNeutrals)`; a count fraction, not an energy fraction |
  | `ehad_over_eem` | Electrons, photons | Stored `EhadOverEem` |
  | `charge` | Jets, muons, electrons | Stored `Charge`; the jet field is not a recomputed PT-weighted jet charge |

  Properties and flags always follow the **same PT-selected object**, including
  producer-order ties. Values that do not apply to a type, missing values, and
  nonfinite measurements are padded with `0` and a false flag. Mass, Ehad/Eem
  and counts additionally require nonnegative values; counts and charge require
  integer values. Isolation is preserved if finite, without imposing a new
  positivity or isolation-selection requirement. The charged fraction requires
  both valid counts and a positive finite denominator. Two zero counts give
  `charged_fraction=0, charged_fraction_valid=0`; zero charged particles with
  positive neutral count give `0, 1`. A genuine zero lepton D0 similarly gives
  `abs_d0=0, abs_d0_valid=1`.

  **Schema-linked validity flags automatically guard measurement cuts.** For
  example, `abs_d0 > threshold` also requires `abs_d0_valid == 1` on the same
  object. An inapplicable jet isolation value cannot satisfy an isolation cut.
  The flexible ERL runner does not independently search linked flags unless
  `--allow-validity-cuts` is enabled. `obj_feature_schema` describes each column's kind and units,
  applicable types, raw sources, definition, and associated validity column.
  All applicable raw fields must exist and align with their collection's PT,
  including fields in empty collections. No D0 significance, jet truth flavour,
  or tagging label is added.
- `pf [N,1,1]`: a **zero compatibility placeholder**, not reconstructed PF
  candidates. The existing `scripts/make_reference_anomalous.py` preserves
  both the real `obj` tensor and event features unchanged when mixing.

Existing conversion commands now produce the real `obj` tensor automatically;
no new flag is needed. Event-only runners ignore `obj`. The current flexible
ERL runner loads its columns and schema dynamically; `--basic-object-features`
selects the four base columns. Legacy runners hardcoded to an older layout
are not interchangeable with schema-aware loading.
The public `build_object_tensor(records)` function in `objects.py` can also
be used independently on flat raw Awkward records.

Zero-object events are retained. A missing required column is an error, not an
assumed zero multiplicity. Input files must share a compatible raw schema.
Convert background and signal with the same settings before mixing. To keep
converted schemas consistent, regenerate both classes (and then their mixtures)
to use the newly added features; existing `.pt` files are not automatically updated.
Keep the source sidecars with the original `.pt` files: a mixed file's `source_files`
identifies those `.pt` inputs, whose sidecars resolve the raw provenance.
Existing `.pt` files and source sidecars are overwritten by default, with a
warning for each existing file. Add `--no-overwrite` to refuse replacement
(or pass `overwrite=False` to the Python function). Input files and output
symbolic links are protected. Serialization finishes before files are replaced.

## Expand

The core function is `convert_raw_parquet_to_pt` in `convert.py`. Default
multiplicity definitions and the lightweight `EventFeature` live in
`features.py`. Additional required fields are loaded automatically:

```python
import awkward as ak
import numpy as np
from raw_event_conversion import EventFeature, convert_raw_parquet_to_pt

def sum_jet_pt(records):
    pt = records["L1T_JetPuppiAK4_PT"]
    valid = ak.fill_none(np.isfinite(pt) & (pt > 30), False)
    return ak.sum(ak.where(valid, pt, 0), axis=1)

activity = EventFeature(
    "ht_jets_pt30", ("L1T_JetPuppiAK4_PT",), sum_jet_pt,
    kind="continuous", units="GeV",
    definition={"pt_threshold_GeV": 30, "comparison": ">"},
)
convert_raw_parquet_to_pt(
    "raw.parquet", "events.pt", label=1,
    extra_features=[activity], thresholds={"muons": [3, 5, 15]},
)
```

`event_features` is float64, with per-column kinds retained in its schema.
The event-only runners honor these kinds. Extra
features must return one finite value per event; undefined pair quantities need
an explicit modeling/validity policy, not implicit zero-filling.

Synthetic verification (no production data or EOS access):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_raw_event_conversion.py -v
```
