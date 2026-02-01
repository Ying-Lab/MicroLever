# MicroLever

**A dynamic model of longitudinal microbial ecology predicts fecal microbiota transplantation efficacy**

This repository contains code for **MicroLever**, a deep learning framework for predicting FMT efficacy from longitudinal microbiome dynamics that models FMT as an ecological perturbation.

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

## Paper abstract 

Fecal microbiota transplantation (FMT) reconstitutes gut microbial ecology via donor microbiota transfer; however, clinical outcomes exhibit significant heterogeneity, where transplants from the same donor often yield divergent therapeutic efficacy in different recipients. Existing predictors predominantly depend on static taxonomic profiles collected from a single time point, implicitly assuming that if different recipients share the same taxonomic abundance profile captured in a single snapshot, they will exhibit the same treatment efficacy under transplantation from the same donor. However, in our study, we utilized a generalized Lotka–Volterra (gLV) framework to model the FMT process, which is essentially the dynamic changes in the microbial community perturbed by the transplanted donor microbiota. We found that communities with highly similar static profiles (Bray–Curtis distance < 0.15) can transition to markedly different post-perturbation steady states under an identical perturbation. We identified “statically indistinguishable pairs” (SIPs): recipients with near-identical pre-FMT compositions that nevertheless exhibit opposite outcomes following transplantation from the same donor. Consequently, longitudinal community data are essential because they capture change trends instead of static snapshots, thereby improving the reliability and accuracy of predicting responses to perturbation. We therefore introduce MicroLever, a deep learning framework for predicting FMT efficacy from longitudinal microbiome dynamics that models FMT as an ecological perturbation. The framework captures recipient temporal dynamics via a Recipient Dynamics Encoder, integrates microbial interaction topology using a Graph Convolutional Network (GCN), and conditions donor inputs on the recipient’s ecological context through a Donor-to-Recipient Modulator. MicroLever achieved 92.86% accuracy in zero-shot evaluations on independent clinical cohorts. Notably, the model maintains superior performance under sparse sampling and observational noise. Furthermore, interpretability analyses highlight donor-derived Lachnospiraceae and Streptococcus-associated interaction signals linked to competitive suppression of Clostridioides difficile, consistent with colonization-resistance mechanisms. Overall, MicroLever provides an accurate, robust, and interpretable framework for predicting FMT efficacy, reinforcing the necessity of longitudinal dynamics for overcoming the limitations of snapshot-based inference.

## Real cohort prediction (example)

```bash
python scripts/predict_real_clr_spline.py --config configs/real_test_default.json
```
