# Contributing

Contributions are welcome — especially new experiments, better datasets, or deployment improvements.

## Adding a New Experiment

1. Copy the closest existing script, e.g. `cp experiments/01_baseline_ciou.py experiments/10_my_experiment.py`
2. Change `EXP_NAME = 'my_experiment'`
3. Add your modification inside `main()` before the `model.train(...)` call
4. Add the estimated runtime to `APPROX_TRAIN_HOURS` in `src/config_shared.py`
5. Open a Pull Request with your results JSON attached

## Code Style

- Black formatting: `pip install black && black .`
- Max line length: 100
- All new modules go in `src/models/` or `src/utils/`
- Every new model class needs a docstring citing its paper

## Reporting Issues

Please include:
- Your GPU model and VRAM
- Python and PyTorch versions (`python --version`, `python -c "import torch; print(torch.__version__)"`)
- The full error traceback
- Which experiment script failed
