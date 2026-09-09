# Local data directory

This directory is a local scaffold only. The research corpus contains
human-participant audio, video, annotations, and derived participant-level
features, so no data files are distributed through this repository.

Git tracks this README and empty `.gitkeep` placeholders. Everything else
below `data/` is ignored.

| Directory | Local contents |
|---|---|
| `raw/` | Original recordings and calibration assets |
| `annotations/` | Identity, transcript, timing, and review artefacts |
| `processed/` | Synthetic data and locally assembled model-ready tables |

## Synthetic data

Synthetic datasets can be regenerated without private inputs:

```powershell
hri-turn-taking vap-generate
hri-turn-taking generate-synthetic
```

The first command creates the participant-wise VAP schema. The second creates
the frozen legacy frame-level TRP ablation dataset.

## Real-data boundary

Real sessions are represented on a 25 Hz grid using columns such as
`P1_head_yaw_deg`, `P1_pose_valid`, and `P1_vad`. Participant VAD is gold-tier:
it may define targets and evaluation events but must never be selected as an
input feature.

The real-data path is deliberately fail-closed:

```powershell
hri-turn-taking vap-intake <delivery-dir>
hri-turn-taking vap-assemble <intake.csv> <assembled.csv> --feature <feature.csv>
hri-turn-taking vap-audit --data <assembled-file-or-directory> --data-kind real --require-ready
```

Each locally supplied feature file needs the provenance manifest required by
the relevant contract under `docs/`. Intake and assembly verify participant
mapping, time alignment, exact keys, source hashes, column ownership, causal
feature declarations, and readiness for the selected feature set. Missing real
observations must remain explicitly missing or invalid; they must not be
silently converted into genuine zero-valued measurements.

Access to the corpus and its manifests must be obtained through the project's
authorised data-governance process. A clone of this repository alone is not
sufficient to run the formal real-data experiments.
