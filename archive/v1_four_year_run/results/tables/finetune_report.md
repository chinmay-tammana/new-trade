# Model Training, Fine-Tuning, and Ablation Report

## 1. XGBoost Fine-Tuning Strategies (Evaluated on Validation)
| Strategy | Val Log-Loss | Val Brier Score | Raw Val Accuracy |
| :--- | :--- | :--- | :--- |
| S2: Refit Base+FT | 0.68953 | 0.24820 | 51.50% |
| S3: Warm Start (r=50, lr=0.01) | 0.69244 | 0.24965 | 50.50% |
| S1: Base Tuned | 0.69297 | 0.24991 | 50.66% |
| S0: Base Default | 0.70176 | 0.25396 | 53.32% |

## 2. Conventional Machine Learning Comparators
| Model | Hyperparameters | Val Log-Loss | Val Accuracy | Fit Time (s) | Latency (µs) | Size (KB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | `C=0.001` | 0.69109 | 50.50% | 0.006 | 395.5 | 3.2 |
| RandomForest | `{'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 10}` | 0.69092 | 52.49% | 0.156 | 1578.1 | 174.8 |
| HistGradientBoosting | `{'learning_rate': 0.02, 'max_depth': 2, 'max_iter': 100}` | 0.69276 | 51.50% | 0.748 | 1607.6 | 98.5 |

## 3. XGBoost Ablation Studies
| Ablation Experiment | Features Used | Val Log-Loss | Delta vs Base | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| Full Baseline (S2) | 22 | 0.68953 | +0.00000 | Neutral |
| Without Time Features | 19 | 0.69227 | +0.00274 | Helped |
| Without Context (SPY/XLF) | 17 | 0.69506 | +0.00553 | Helped |
| Without RVOL | 21 | 0.69247 | +0.00294 | Helped |
| Without Class Balance | 22 | 0.68856 | -0.00097 | Hurt |
| Label Threshold = 5 bps | 22 | 0.69262 | +0.00309 | Helped |
| Label Threshold = 10 bps | 22 | 0.68953 | +0.00000 | Neutral |
| Label Threshold = 15 bps | 22 | 0.69139 | +0.00186 | Helped |
| Label Threshold = 20 bps | 22 | 0.69148 | +0.00195 | Helped |

## 4. Final Deployment Decision
- **Deployed Architecture**: `XGBoost_S2: Refit Base+FT`
- **Selected Tuned Hyperparameters**:
```json
{
  "subsample": 1.0,
  "reg_lambda": 3.0,
  "n_estimators": 400,
  "min_child_weight": 3,
  "max_depth": 3,
  "learning_rate": 0.05,
  "gamma": 1.5,
  "colsample_bytree": 0.7,
  "base_weight": 0.3
}
```
- **Verification**: No Validation or Test rows were utilized during parameter fitting. All fit operations recorded in `results/logs/fit_ledger.json`.