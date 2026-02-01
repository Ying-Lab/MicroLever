# MicroLever

**A dynamic model of longitudinal microbial ecology predicts fecal microbiota transplantation efficacy**

This repository contains code for **MicroLever**, a topology-aware deep learning framework for predicting fecal microbiota transplantation (FMT) efficacy from recipient longitudinal dynamics and donor compositions.

## Two-part layout (simulation vs model)

- `simulation/` — MATLAB scripts for gLV-based synthetic FMT simulation.
- `src/microlever/` — Python package (model + data loading).
- `scripts/` — runnable entry points (train / interpret / real-data prediction / merge metrics).
- `configs/` — example JSON configs.
- `data/` — datasets (**not tracked** in git).
- `experiments/` — outputs (**not tracked** in git).

## Quick start (model)

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py --config configs/train_default.json
```

Outputs:
```
experiments/<run_name>/<timestamp>/
  config.json
  logs/
  models/
  results/
```

## Interpretability

```bash
python scripts/interpret_attention_advanced.py --best_model /path/to/best_*.pth
python scripts/interpret_film_attention.py --best_model /path/to/best_*.pth
```

## Simulation (MATLAB)

See `simulation/README.md`.

## Paper abstract (excerpt)

Fecal microbiota transplantation (FMT) reconstitutes gut microbial ecology via donor microbiota transfer; however, clinical outcomes exhibit significant heterogeneity, where transplants from the same donor often yield divergent therapeutic efficacy in different recipients. Existing predictors predominantly depend on static taxonomic profiles collected from a single time point, implicitly assuming that if different recipients share the same taxonomic abundance profile captured in a single snapshot, they will exhibit the same treatment efficacy under transplantation from the same donor. However, in our study, utilizing a generalized Lotka–Volterra framework parameterized by clinical time-series data, we simulated FMT perturbations and identified “statically indistinguishable pairs” (SIPs)—re...

## Real cohort prediction (example)

```bash
python scripts/predict_real_clr_spline.py --config configs/real_test_default.json
```
