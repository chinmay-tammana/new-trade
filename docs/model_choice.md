# Model Selection Justification: Candidate Architectures and Benchmarks

This document details the machine learning architectures evaluated for next-hour Goldman Sachs direction forecasting, justifying why each model is included in accordance with IMPACT guidelines.

---

## 1. Primary Model: Extreme Gradient Boosting (XGBoost)
- **Role**: Primary candidate for production deployment.
- **Justification**:
  1. **Tabular State-of-the-Art**: Decision-tree ensembles consistently outperform deep learning architectures on small-to-medium sized noisy tabular financial datasets (Gu et al., 2020).
  2. **Regularization Framework**: XGBoost incorporates second-order Taylor expansion approximations of the loss function alongside explicit \(L_1\) (`reg_alpha`) and \(L_2\) (`reg_lambda`) tree leaf regularization, preventing overfitting in low signal-to-noise intraday regimes.
  3. **Fine-Tuning Support**: XGBoost provides native `xgb_model` warm-start boosting, enabling continuous boosting on newly arrived fine-tuning (FT) chunks without discarding historical trees.
  4. **Sample Weighting**: Seamlessly supports non-uniform instance weighting (`base_weight`) and inverse-frequency class balancing.
  5. **Low Latency**: Inference executes in \(< 1.0\) millisecond per candle, easily satisfying real-time chart updating requirements.

---

## 2. Benchmark Comparator 1: Regularized Logistic Regression
- **Role**: Interpretable linear benchmark.
- **Pipeline**: `InFoldWinsorizer(0.5, 99.5)` \(\rightarrow\) `StandardScaler` \(\rightarrow\) `LogisticRegression(penalty='l2', solver='lbfgs')`.
- **Justification**:
  1. **Linearity Baseline**: Serves as the fundamental test of whether non-linear decision boundaries and feature interactions add genuine predictive edge over an optimal linear combination.
  2. **Direct Probability Calibration**: Produces native logit outputs without requiring separate calibration layers.
  3. **Overfitting Resistance**: With strong \(L_2\) regularization (tuned via hyperparameter \(C\)), linear models are highly resilient to regime shifts and extreme noise.

---

## 3. Benchmark Comparator 2: Random Forest
- **Role**: Non-linear bagging ensemble benchmark.
- **Model**: `RandomForestClassifier(n_estimators, max_depth, min_samples_leaf, class_weight='balanced')`.
- **Justification**:
  1. **Bagging vs. Boosting**: While boosting reduces bias sequentially, Random Forest reduces variance via bootstrap aggregation of de-correlated decision trees.
  2. **Outlier Invariance**: Independent tree voting prevents localized outlier candles from disproportionately distorting the global ensemble.
  3. **Baseline Comparison**: Tests whether parallel variance reduction outperforms sequential gradient boosting on financial returns.

---

## 4. Benchmark Comparator 3: Histogram Gradient Boosting (HistGradientBoosting)
- **Role**: Modern alternative gradient boosting benchmark.
- **Model**: `HistGradientBoostingClassifier(learning_rate, max_depth, max_iter, class_weight='balanced')`.
- **Justification**:
  1. **Implementation Cross-Validation**: Provides an independent, scikit-learn native implementation of histogram-binned gradient boosted trees (inspired by LightGBM).
  2. **Binning Regularization**: Integer binning of continuous features acts as an inherent regularizer against microstructural noise.
  3. **Deployment Comparator**: Evaluates whether XGBoost's specific split heuristics offer tangible gains over standard histogram boosting.

---

## 5. Comparator Switching Protocol
In accordance with the evaluation protocol, the deployed model will be the top-performing XGBoost fine-tuning strategy (S0–S3) **UNLESS** an independent comparator from Stage C outperforms it on Validation log-loss by at least `switch_margin_logloss = 0.002`. In that case, the comparator is deployed to avoid confirmation bias.
