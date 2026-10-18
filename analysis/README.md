# Analysis

Exploratory analysis on the official training file only (`data/train_prices_15min.csv`).
It informs the feature design but is not part of training or inference.

| Path | Content |
|---|---|
| [notebooks/EDA_final.ipynb](notebooks/EDA_final.ipynb) | EDA notebook |
| [reports/EDA_analysis.md](reports/EDA_analysis.md) | Written EDA findings, linking to `figures/eda_final/` |
| [reports/FEATURE_ENGINEERING.md](reports/FEATURE_ENGINEERING.md) | Each feature tied to the EDA finding that motivates it |
| `figures/eda_final/` | Figures produced by the notebook |
| `figures/september_report/` | Figures from the September hold-out assessment (`tools/report_assessment.py`) |

```bash
python -m pip install -r requirements.txt
jupyter notebook analysis/notebooks/EDA_final.ipynb
```
