# Literature Survey: Machine Learning for Intraday Directional Forecasting

## 1. Overview and Theoretical Background
Forecasting financial asset prices has evolved from the baseline Efficient Market Hypothesis (Fama, 1970)—which states that security prices fully reflect all available information—to quantitative and machine learning paradigms that search for local, non-linear informational regularities. In modern intraday equity markets, signals operate under heavy noise, non-stationarity, and severe microstructural transaction friction. Recent empirical asset pricing literature demonstrates that regularized machine learning models (gradient-boosted decision trees, deep sequence models, and penalized linear models) capture subtle interactions that classical statistical models miss (Gu et al., 2020; Chen & Guestrin, 2016).

However, high classification accuracy alone does not guarantee economic utility: uncalibrated raw scores lead to erratic sizing and false confidence, while failure to account for transaction costs and prevailing drift often renders theoretical edges negative in practice (Lopez de Prado, 2018; Niculescu-Mizil & Caruana, 2005).

---

## 2. IMPACT Literature Comparison Table

| Study | Dataset | Method | Evaluation | Gap / Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Fama (1970)**<br>*Journal of Finance* | Historical US equity price series (daily/monthly). | Serial correlation, runs tests, filter rules under EMH. | Statistical independence and risk-adjusted excess returns. | Assumes frictionless markets; does not address non-linear, multi-asset intraday interaction or modern ML formulations. |
| **Platt (1999)**<br>*Adv. Large Margin Classifiers* | Benchmark ML datasets (synthetic & UCI repositories). | Logistic sigmoid scaling on margin distance (\(\text{logit}(p) = a f(x) + b\)). | Log-likelihood, calibration curves, Brier score. | Evaluated primarily on static classification domains; does not model financial market regime shifts or asymmetric selective classification. |
| **Niculescu-Mizil & Caruana (2005)**<br>*ICML* | Eight binary benchmark datasets from UCI and KDD Cup. | Comparison of Platt scaling vs. isotonic regression across SVM, GBDT, neural nets, Naive Bayes. | Root Mean Squared Error (squared error), calibration error, log-loss. | Standard cross-validation on IID tabular benchmarks; does not consider non-IID temporal dependencies or intraday financial market friction. |
| **Chen & Guestrin (2016)**<br>*ACM KDD* | Benchmark classification, ranking, and CTR datasets (Higgs, Allstate, Yahoo LTR). | Scalable end-to-end tree boosting (XGBoost) with sparsity awareness and weighted quantile sketch. | AUC, NDCG, classification accuracy, training wall-clock time. | Generic tabular optimization; raw probabilities from tree leaf sums are frequently over-confident and uncalibrated without post-hoc scaling. |
| **Lopez de Prado (2018)**<br>*Advances in Fin. ML* | High-frequency futures and equity tick / order-book data. | Financial ML framework: purged & embargoed CV, fractional differentiation, meta-labeling, triple-barrier method. | Sharpe ratio, Information Ratio, deflated Sharpe, out-of-sample backtests. | Complex high-frequency infrastructure; primary focus on tick/event barriers rather than session-anchored hourly equity decision systems with abstention. |
| **Fischer & Krauss (2018)**<br>*Eur. J. Oper. Res.* | S&P 500 constituents from 1992 to 2015 (daily close data). | Long Short-Term Memory (LSTM) networks vs. Random Forest, GBDT, and Logistic Regression. | Daily out-of-sample directional accuracy and annualized long-short portfolio return. | Daily closing prices only; ignores intraday hourly dynamics and assumes frictionless rebalancing without execution slippage models. |
| **Gu, Kelly, & Xiu (2020)**<br>*Rev. Financ. Stud.* | Monthly US equities (CRSP/Compustat, 1957–2016; 30,000 stocks, 900+ predictors). | Comprehensive survey: OLS, ElasticNet, Partial Least Squares, Random Forests, GBDT, Deep Neural Networks. | Out-of-sample \(R^2_{\text{OOS}}\), Sharpe ratio, factor attribution. | Monthly macro/fundamental rebalancing horizon; macro-scale asset pricing does not directly translate to intraday (1h) trade-level directional execution. |
| **Wilson (1927)**<br>*JASA* | Theoretical binomial sampling distributions. | Score confidence interval for binomial proportions with finite-sample continuity correction. | Coverage probability under small sample sizes and extreme probabilities. | Statistical inference tool; requires integration into directional evaluation pipelines to prevent premature declarations of edge on small test counts. |

---

## 3. Identified Methodological Gap and Project Placement
While existing quantitative research provides mature boosting algorithms (Chen & Guestrin, 2016) and general warnings regarding financial data leakage and overfitting (Lopez de Prado, 2018), a critical gap remains in practical intraday trading support systems:

1. **Uncalibrated Model Confidence**: Standard implementations of tree ensembles produce raw logits that do not reflect true posterior probabilities of price continuation, resulting in distorted risk-reward assessments.
2. **Asymmetric Drift & Failure to Abstain**: In equity markets subject to long-term upward drift, symmetric decision thresholds frequently collapse into degenerate "always-long" models or suffer high churn in sideways regimes.
3. **Flawed Intraday Aggregation & Leakage**: Intraday systems frequently leak information across overnight session gaps (e.g. attempting to predict next-morning open from previous close without separate treatment) or fit scaling transformations across entire datasets.

**Project Placement**: This project addresses these gaps by implementing a fully leak-free, session-anchored hourly forecasting pipeline on Goldman Sachs (`GS`) equity. It integrates market context (`SPY`, `XLF`), strictly in-fold preprocessing, walk-forward fine-tuning across four chronological eras, post-hoc Platt probability calibration, and a selective abstention rule with asymmetric confidence margins (\(d_{\text{up}}, d_{\text{down}}\)) evaluated under realistic transaction friction (5 bps) and honest Wilson confidence intervals.
