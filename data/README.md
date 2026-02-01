# Data folder (not committed)

Recommended structure:

```
data/
  adjacency/
    fake_A_53_dHOMO.csv
  sim/
    train/
      FMT_structured_data.h5
      csv_reports/
        donor_final_abundance.csv
        Cdiff_labels_table.csv
        recipient_final_abundance.csv
        transplant_cdiff_real.csv
        recipient_growth_rates.csv
```

Edit `configs/train_default.json` to point to your files.


For real cohort prediction:

```
data/real/
  FMT_real_data.h5
  Cdiff_labels_table.csv
  donor_final_abundance.csv
```
