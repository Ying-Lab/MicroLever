# Simulation (MATLAB)

This folder contains MATLAB scripts that generate ecologically grounded synthetic FMT datasets.

## Included scripts

- `matlab/sim40_global.m`
- `matlab/twins_test_2.m`

## Required external files (not included)

Both scripts reference additional dependencies that are not in this repository yet:

1) **Interaction matrix**
- `fake_A_53_dHOMO.csv` (or your inferred interaction matrix)

2) **MATLAB helper functions**
- a `func/` folder (e.g., RK4 solver, gLV helpers, labeling utilities)

Place them next to the `.m` files (same directory) or update the `addpath(...)` and file paths in the scripts.

## Outputs

The scripts write:
- `FMT_structured_data.h5`
- `csv_reports/` including:
  - `donor_final_abundance.csv`
  - `Cdiff_labels_table.csv`
  - (and other reports used in your Python pipeline)

Then copy/rename outputs into `data/` and set paths in `configs/train_default.json`.
