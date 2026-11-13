# Track B implementation

This guide covers saved-result inspection, local setup, and the classification,
evidence, and feedback pipeline. For the method and final results, see
[the project overview](../README.md). All commands below run from `trackb/`.

## Saved results

The summary command needs Python 3.10 or later:

```bash
python scripts/summarize_results.py
```

The final report bundle is in
[`docs/validation/gpu_final_2026-10-18/`](docs/validation/gpu_final_2026-10-18/README.md).
The summary command uses the Python standard library and reads existing files.

| Cumulative feedback labels | Accuracy | Balanced accuracy | Muscle–stroma mutual error |
| ---: | ---: | ---: | ---: |
| 200 | 98.33% | 97.96% | 5.52% |
| 300 | 99.44% | 99.26% | 1.38% |
| 400 | 99.53% | 99.35% | 1.18% |

The final classifier makes 17 errors among 3,590 evaluation tiles. Regional
explanations pass the stored format and grounding checks for 1,959 tiles
(54.6%). The final stage updates the classifier and reuses the earlier
200-label Qwen checkpoint. Feedback labels come from dataset annotations;
the query pool and evaluation split share the CRC-VAL-HE-7K domain.

[Recorded stage metrics](docs/validation/gpu_final_2026-10-18/final_4080_v1/results/vlm_summary.csv)

## CPU development

Use Python 3.11 or 3.12:

```bash
python -m pip install -r requirements-cloud.txt
python -m pytest -q tests
```

`bash scripts/setup_cloud.sh` also installs CPU PyTorch and runs
`scripts/validate_cpu.py`. Tests that require PyTorch skip when it is absent.

## Pipeline and application

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py --stage prepare --config configs/final_4080.yaml --dry-run
python -m streamlit run app.py -- --config configs/final_4080.yaml
```

Set local data, model, and output paths in the configuration before a full run.
The Streamlit application reads model outputs and image paths from that run.
The checked-in report bundle contains metrics and provenance; runtime weights,
embedding caches, and images remain external.

The final experiment uses `prepare`, `cache`, `probe`, `feedback`, `explain`,
and `report`: cache whole-tile features, fit and update the linear probe, then
produce explanations with the inherited Qwen checkpoint and region guide.
`evidence` calibrates the region guide; `zeroshot`, `r1`, `r2`, and `r3` support
the earlier model-training workflow. See the [final run sequence](docs/FINAL_GPU.md)
for stage order and required inherited artifacts.

## Data and models

`data/roles.csv` fixes 16,200 train_fit, 1,800 train_dev, 3,590 pool, and 3,590 eval
tiles. Manifests record image identities separately from the configurable image
root. Feedback records preserve the query model version and label source.

The default explanation model is Qwen2-VL-2B with language LoRA and frozen vision
weights. CONCH and MUSK supply whole-tile and regional representations.

## Experiment documentation

- [Final configuration and execution](docs/FINAL_GPU.md)
- [Windows GPU setup](docs/WINDOWS_RUNBOOK.md)
- [Earlier regional-evidence and feedback experiment](docs/FOLLOWUP_GPU.md)
- [Final recorded run](docs/validation/gpu_final_2026-10-18/README.md)
- [Implementation scope](docs/TRACK_B.md)

## Code map

| Path | Purpose |
| --- | --- |
| [`tb/`](tb/) | Features, classifiers, evidence, feedback, and training |
| [`app.py`](app.py) | Prediction review and feedback interface |
| [`run_pipeline.py`](scripts/run_pipeline.py) | Stage-level experiment entry point |
| [`final_4080.yaml`](configs/final_4080.yaml) | Final experiment configuration |
| [`docs/validation/`](docs/validation/) | Saved experiments and evaluation records |
