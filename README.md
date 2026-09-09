# Predictive Turn-Taking for Multi-Party HRI

This repository contains a research implementation of participant-wise voice
activity projection (VAP) for three- and four-person human-robot interaction.
The model estimates each participant's future speaking activity and supports
offline comparison of predictive and reactive floor-entry policies.

The project was developed for the MSc dissertation *Learning from Humans: A
Hybrid Multimodal Framework for Predictive Turn-Taking in Multi-Party HRI*.
This repository contains the software, tests, configuration, and technical
documentation. It intentionally excludes the dissertation source, presentation
files, recordings, annotations, trained checkpoints, and generated results.

## What is implemented

- A participant-structured causal Transformer or GRU with shared per-person
  weights, absent-slot masking, and four future voice-activity bins.
- Causal acoustic features and head-pose inputs, with explicit feature masks
  for audio-only, head-pose-only, and multimodal comparisons.
- Fail-closed real-data intake, assembly, provenance, and readiness checks.
- Group-disjoint nested leave-one-group-out evaluation and saved-prediction
  aggregation.
- Event-level evaluation for shift/hold, shift prediction, next speaker,
  overlap resolution, and robot-entry opportunities.
- Dual-threshold predictive policies, a causal reactive baseline, matched
  replay, previous-VAD oracle, gaze, and late-fusion diagnostics.
- A frozen frame-level TRP baseline under `src/hri_turn_taking/legacy/`.

The architecture is described in [ARCHITECTURE.md](ARCHITECTURE.md). The main
implementation is under `src/hri_turn_taking/vap/`.

## Repository layout

```text
configs/                 Default configuration and reproducibility metadata
data/                    Empty local-data scaffold; contents are ignored
docs/                    Methodology and executable-contract documentation
src/hri_turn_taking/     Package source
tests/                   Unit and integration tests
tools/                   Audited data and experiment entry points
```

## Installation

Python 3.10 or newer is required. The core package uses PyTorch and runs on CPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras are available for annotation, vision, audio capture, and
Torchaudio integrations:

```powershell
python -m pip install -e ".[annotation]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[audio]"
python -m pip install -e ".[ml]"
```

## Synthetic quick start

The synthetic route exercises the model without access to the private corpus:

```powershell
hri-turn-taking vap-generate
hri-turn-taking vap-train --data-kind synthetic
hri-turn-taking vap-predict data/processed/multiparty_synthetic.csv --data-kind synthetic
hri-turn-taking vap-demo --data-kind synthetic
```

Synthetic results validate software behaviour and pipeline wiring only. They
must not be interpreted as evidence for performance on real conversations.

## Real-data workflow

Real inputs use a 25 Hz wide table with participant columns named
`P{i}_<feature>`. Gold `P{i}_vad` columns are used only for targets and event
evaluation; they are never model inputs.

```powershell
hri-turn-taking vap-intake <delivery-dir>
hri-turn-taking vap-assemble <intake.csv> <assembled.csv> --feature <feature.csv>
hri-turn-taking vap-audit --data <assembled-file-or-directory> --data-kind real --require-ready
hri-turn-taking vap-train --data <assembled-directory> --data-kind real --feature-set audio_plus_head_pose
hri-turn-taking vap-predict <assembled-directory> --data-kind real --model <checkpoint.pt> --validation-only
```

The real-data commands intentionally require provenance sidecars and exact
source bindings. They fail closed when targets, selected inputs, participant
mapping, time alignment, or group provenance are incomplete. See
[data/README.md](data/README.md) and [docs/methodology.md](docs/methodology.md)
for the data boundary and evaluation protocol.

Set `HRI_CORPUS_ROOT` and, when needed, `HRI_HPE_DELIVERY_ROOT` to authorised
local source directories. These operator-controlled roots are never read from
an artifact manifest; leaving them unset uses the local `data/raw` scaffold and
the documented sibling HPE layout.

The recordings and participant-level artifact files are not distributed in
this repository. Run the commands above from the repository root. The formal
real-data experiments cannot be reproduced from a fresh clone without
separately authorised access to the source corpus and its manifests.

## Evaluation status

The completed real-data study used an eight-scene, four-group evaluation view
and nested leave-one-group-out testing. It reports balanced frame metrics and
event-level measures rather than raw accuracy, because future speech labels are
strongly imbalanced.

The completed limited head-pose comparison did not support H1, and the matched
offline predictive-versus-reactive replay did not support H2. A later
single-seed gaze diagnostic was suggestive but remained post-hoc and did not
change either conclusion. No physical Furhat deployment or live TTS latency
study was completed. The code therefore remains research software, not a
production robot controller.

Technical claim boundaries and experimental contracts are retained under
`docs/`; generated metrics and dissertation artefacts are deliberately omitted.

## Tests and linting

Run both checks before changing a model, data contract, or experiment runner:

```powershell
python -m pytest
python -m ruff check .
```

The suite covers causal feature extraction, future-perturbation resistance,
participant permutation equivariance, absent-slot isolation, group-level
splits, provenance replay, artifact aggregation, and experiment-specific
guards.

## Data and ethics

The underlying corpus contains human-participant audio and video. Raw media,
annotations, participant-level features, checkpoints, logs, and generated
outputs must remain local and are excluded by `.gitignore`. Do not commit them
to forks or derivative repositories. Complete the
[public release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md) before changing
the remote repository's visibility.
