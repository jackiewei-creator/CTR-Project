# CTR Modeling and Evaluation (Notebook 4)

This notebook focuses on developing, evaluating, and visualizing models for **Click-Through Rate (CTR) prediction**, using the merged ad–feed dataset from previous preprocessing stages.  
It establishes a logistic regression baseline, visualizes key metrics, interprets feature effects, and refines results using Random Forest and LightGBM with cross-validation.

---

## 4.3 Performance Visualization (ROC & PR Curves)

**Goal:** Evaluate model discrimination and calibration beyond scalar metrics.

**Steps:**
- Computed **ROC (Receiver Operating Characteristic)** and **Precision–Recall** curves using validation predictions.  
- Compared logistic regression performance against random guessing.

**Findings:**
- The baseline logistic regression achieved **ROC-AUC ≈ 0.75**, showing good separation power.  
- PR curve revealed moderate precision under high recall — typical for imbalanced CTR datasets.  
- These plots confirmed the model learned meaningful behavioral signals.

---

## 4.4 Feature Importance Analysis (Baseline Model)

**Goal:** Interpret how each feature influences CTR probability.

**Steps:**
- Extracted standardized coefficients from logistic regression.  
- Ranked features by absolute weight and visualized top 15 using `seaborn.barplot`.

**Key Insights:**
- Positive features (increase CTR): `f_refresh_sum`, `f_up_mean`, `slot_id`, `f_up_sum`.  
- Negative features (decrease CTR): `f_cat_uniq`, `f_refresh_mean`, `f_browser_life`.  
- Behavioral engagement indicators dominate model influence, while broad interest diversity may reduce click tendency.

---

## 4.5 Random Forest Baseline

**Goal:** Provide a nonlinear benchmark capturing feature interactions.

**Steps:**
- Trained a **Random Forest Classifier** with balanced class weights.  
- Computed Gini-based feature importances and identified top 30 predictors.  
- Evaluated multiple metrics including ROC-AUC, PR-AUC, LogLoss, and F1.

**Performance:**
| Metric | Value |
|--------|--------|
| ROC-AUC | 0.798 |
| PR-AUC | 0.122 |
| LogLoss | 0.493 |
| F1 | 0.092 |

**Insights:**
- Nonlinear modeling improved generalization over logistic regression.  
- Features like `f_refresh_sum` and `f_up_mean` remained top predictors.  
- Random Forest importance ranking guided feature selection for LightGBM.

---

## 4.6 LightGBM with Cross-Validation (Top-30 Features)

**Goal:** Boost model performance and stability using gradient boosting.

**Steps:**
- Applied **LightGBM** with 5-fold stratified CV and early stopping (150 rounds).  
- Used the **Top-30 most informative features** identified from Random Forest.  
- Tracked ROC-AUC, PR-AUC, LogLoss, Precision, Recall, and F1 across folds.

**Cross-Validation Results:**
| Metric | Mean | Notes |
|--------|------|--------|
| ROC-AUC | 0.812 | Consistent improvement |
| PR-AUC | 0.139 | Smoother PR curve |
| LogLoss | 0.4182 | Lower error |
| F1 | 0.10 | Higher recall balance |

**Insights:**
- LightGBM outperformed both baselines in all metrics.  
- Key behavioral features (`f_refresh_sum`, `f_up_mean`, `slot_id`) retained high importance.  
- Boosting provided better recall and reduced overfitting, suitable for CTR prediction.

---

## Outputs

| Output | Description |
|--------|--------------|
| **Model metrics** | Evaluation tables for Logistic Regression, Random Forest, and LightGBM (AUC, PR-AUC, LogLoss, F1). |
| **Feature importance** | Random Forest and LightGBM rankings for interpretability. |
| **Curves** | ROC & PR curves visualized inline. |
| **Selected features** | `top_feats` (Top-30) and `safe_feats` (Top-27) used for final training. |

---

## Summary

LightGBM achieved the best balance between accuracy and interpretability, reaching  
**ROC-AUC ≈ 0.814**, **PR-AUC ≈ 0.142**, and **F1 ≈ 0.10**.  
Across all models, engagement-based behavioral signals proved to be the most consistent predictors of ad clicks.
