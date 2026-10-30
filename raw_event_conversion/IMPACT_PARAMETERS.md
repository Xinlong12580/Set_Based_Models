# Impact parameters: source verification

Checked on 2026-10-04. This describes the COLLIDE2V production sources
identified in the adjacent trigger-study audit, not arbitrary Parquet files
or every production revision. No production data were converted or changed.

## Units and definition

- `L1T_PFPart_D0`: signed transverse impact parameter, **mm**.
- `L1T_PFPart_ErrorD0`: corresponding uncertainty, **mm**, not a variance.
- `abs(D0) / ErrorD0`: dimensionless, defined only for finite D0 and finite,
  strictly positive ErrorD0 with a finite ratio.
- D0 uses the producer beam spot, falling back to the coordinate origin.
  It is not recomputed with respect to the event's reconstructed primary
  vertex, nor is it a secondary-vertex decay radius.

The converter uses these values directly; no cm-to-mm conversion, uncertainty
floor, epsilon, clipping, or inferred detector resolution is applied.

The same mm convention and beam-spot/origin reference apply to
`L1T_Electron_D0` and `L1T_MuonTight_D0`. The ranked-lepton features store
their absolute values with validity flags; they do not add significance,
load lepton ErrorD0, or recompute D0 relative to a reconstructed PV.

## Pinned source chain

The simulation revision is `d10b609b92f3c188ccf9c6c7459eee610f2b30d9`;
the Delphes fork revision is `4b2903bede33b2d882174ce8184075497669241c`.

1. [ParticlePropagator.cc](https://github.com/fermilab-hep-ai/delphes/blob/4b2903bede33b2d882174ce8184075497669241c/modules/ParticlePropagator.cc):
   positions are converted by `1e-3` to meters; the transverse closest-approach
   calculation subtracts the beam-spot position, and `D0` is stored with
   `1e3`, restoring millimeters. An unavailable beam spot defaults to zero.
2. [TrackSmearing.cc](https://github.com/fermilab-hep-ai/delphes/blob/4b2903bede33b2d882174ce8184075497669241c/modules/TrackSmearing.cc):
   when enabled, Gaussian D0 smearing uses `d0Error` directly and assigns
   `ErrorD0 = d0Error`; both therefore have the D0 length unit.
   [DelphesClasses.cc](https://github.com/fermilab-hep-ai/delphes/blob/4b2903bede33b2d882174ce8184075497669241c/classes/DelphesClasses.cc)
   also constructs the D0 covariance diagonal as `ErrorD0**2`.
3. [TreeWriter.cc](https://github.com/fermilab-hep-ai/delphes/blob/4b2903bede33b2d882174ce8184075497669241c/modules/TreeWriter.cc):
   `ProcessParticleFlowCandidates`, `ProcessElectrons`, and `ProcessMuons`
   copy both D0 and ErrorD0 directly.
4. [process_root_parquet.py](https://github.com/fermilab-hep-ai/simulation/blob/d10b609b92f3c188ccf9c6c7459eee610f2b30d9/process_root_parquet.py):
   `PFPart` reads `EFlow.D0`/`EFlow.ErrorD0`, with the `L1T` prefix for the
   L1 view. It sorts/trims candidates and saves these fields without any
   length rescaling; neither is in `SAFE_FLOAT16`. Electron and tight-muon
   D0 likewise come directly from their corresponding producer collections.

## Important limitation of the checked production

The [COLLIDE2V v1.0 card](https://github.com/fermilab-hep-ai/simulation/blob/d10b609b92f3c188ccf9c6c7459eee610f2b30d9/cards/CMS_PhaseII_200PU_COLLIDE2V_v1.tcl)
does not put `TrackSmearing` on its execution path. The L1 tracking chain
uses tracking efficiencies plus momentum/energy smearing. Its raw PF output
is `L1TEFlowMerger/eflow`, containing the L1 calorimeter's eflow tracks plus
photons and neutral hadrons. Candidate ErrorD0 initializes to zero;
momentum smearing clones candidates without assigning an impact-parameter
uncertainty. D0 is geometry-derived rather than calibrated L1 impact-parameter
resolution smearing in this checked chain.

A read-only check used the first 32 events of:

```text
../Trigger/data/minbias/minbias-NEVENT10000-RS59000002.parquet
```

Only PF PT, Charge, D0 and ErrorD0 were read. Among 1365 finite charged
candidates with PT > 2 GeV, all D0 values were finite and all ErrorD0 values
were zero; none had positive finite ErrorD0. This is a limited availability
check, not evidence that all shards/revisions have the same values and not
the basis for deriving the units.

Consequently, the new significance observables are implemented for inputs
with meaningful uncertainties, but are undefined for these checked candidates.
They contribute to the saved invalid-significance count, not to any significance
bin. Conversion warns when eligible measurements are invalid. Zero significance
bin counts in this situation are not a physics statement.

For a realistic L1 displaced-trigger study, validate the producer's displaced
tracking efficiency, D0 resolution and uncertainties, reference convention,
and candidate truncation. Fixing/regenerating simulation is outside this
preprocessing change; an arbitrary uncertainty must not be substituted here.

## Longitudinal impact parameters

DZ was checked against the same pinned production sources on 2026-10-28.
`L1T_Electron_DZ` and `L1T_MuonTight_DZ` are signed lengths in **mm**.
`ParticlePropagator` computes `dz = zd - bsz`: the longitudinal coordinate
at transverse closest approach to the z axis, minus the beam-spot z coordinate.
It stores `DZ = dz * 1e3`, converting its internal meters back to millimeters.
The checked card configures no beam-spot producer, so the reference falls
back to the coordinate origin. This is not DZ relative to a reconstructed PV.

The object tensor appends `dz` and `dz_valid`; the pair tensor copies both
members and appends `abs_delta_dz = abs(dz_1 - dz_2)` with its own guard.
Only finite DZ on a present electron/muon is valid. Invalid or inapplicable
values are zero-filled with false guards; a physical zero remains valid.
No absolute value is applied before the difference. `ErrorDZ` is not read,
and no significance, resolution smearing, vertex reassignment or rescaling
is introduced. The event tensor is unchanged.

The checked execution path has no `TrackSmearing`; momentum/energy smearing
preserves DZ, and `TreeWriter` copies it to output objects. Therefore these
DZ values are truth-trajectory-derived, without realistic DZ resolution.
A limited read-only check of the first 32 events of the minbias file above
found zero `ErrorDZ` for all stored PF/PUPPI candidates and the one present
tight muon. It does not establish availability across all shards or revisions.
