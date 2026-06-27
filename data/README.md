# data/

Local datasets for the project. **Everything in this folder is git-ignored**
except this README and the `.gitkeep` placeholders (see the repo `.gitignore`):
raw A/V, extracted features, and annotations must never be committed or leave the
local machine (ethics — see `AGENTS.md` §9).

## Layout

| Subdir | Contents |
|---|---|
| `raw/` | Untouched recordings: monochrome + RGB-D video, lavalier + ReSpeaker-array audio, calibration data. |
| `processed/` | Derived, model-ready artifacts: the 25 fps wide feature table (`P{i}_<feature>` schema) and synthetic datasets. |
| `annotations/` | ELAN annotation files, including the required overlap-type tier (`backchannel` vs `competitive`). |

## Regenerating synthetic data

The processed CSVs are reproducible from the CLI and are not stored in git:

```powershell
hri-turn-taking vap-generate          # -> processed/multiparty_synthetic.csv (VAP schema)
hri-turn-taking generate-synthetic    # -> processed/synthetic_turn_taking.csv (legacy ablation)
```
