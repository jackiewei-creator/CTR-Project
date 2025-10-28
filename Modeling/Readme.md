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

## 4.7 Discussion: Threshold Tuning and Class Rebalancing

**Goal:**  
Refine model performance beyond the LightGBM baseline by tuning classification thresholds and adjusting class weights to better handle severe data imbalance.

---

### Step 1 — Decision Threshold Expansion

**Method:**  
The initial LightGBM used a range (0.01 - 0.5) to decide whether a prediction is a “click.”  
Here, the threshold was expanded across a broader range (0.01–0.95) to locate the point that maximizes F1-score.

**Findings:**  
- The optimal F1 was reached at a higher threshold (~0.86).  
- This threshold indicates that the model only predicts a click when it is very confident.  
- Consequently:  
  - **Precision increased sharply**, reducing false positives.  
  - **Recall dropped substantially**, as many true clicks were no longer detected.  
  - Overall **F1 improved moderately** (≈ +0.13 over baseline).

**Interpretation:**  
This pattern reflects the classic **precision–recall trade-off**:  
as the model becomes more conservative, its accuracy per prediction improves,  
but it identifies fewer positive cases.

---

### Step 2 — Rebalancing Class Weights

**Method:**  
Positive sample weights were increased (via `scale_pos_weight`)  
to emphasize the minority class during training.  
This approach encourages the model to predict more positives and better compensate for the dataset’s strong imbalance.

**Results:**

| Metric | Before | After Rebalancing |
|:--------|:--------|:----------------|
| Precision | ~0.09 | **0.24** |
| Recall | ~0.63 | **0.25** |
| F1-score | ~0.10 | **0.24** |
| ROC-AUC | 0.81 | 0.81 |
| PR-AUC | 0.14 | 0.14 |

**Insights:**  
- Precision and F1 improved significantly, indicating more reliable click predictions.  
- Recall decreased as the model favored certainty over coverage.  
- Overall, this adjustment produced a **better-calibrated classifier** without compromising ranking metrics (AUC values remained stable).

---

### Step 3 — Practical Interpretation

Such trade-offs are expected and realistic in **CTR prediction tasks**,  
where click events represent only ~1–2% of all impressions.  
In both research and production systems, **F1-scores around 0.20–0.25**  
are considered strong baselines for this problem scale.

In real advertising systems:
- Higher precision means users see fewer irrelevant ads — improving engagement and satisfaction.  
- Lower recall implies missing some potential clicks,  
  but this is acceptable when prioritizing ad quality and user experience.

Hence, the observed **“precision-up, recall-down”** trend  
aligns with production-level CTR model behavior.

---

### Step 4 — Broader Evaluation Perspective

Although F1 provides a simple balance between precision and recall,  
it depends on an arbitrary threshold and fluctuates under class imbalance.  

For both academic and industrial evaluation, **ranking-based metrics** are more stable and informative:

- **ROC-AUC** — measures how well the model ranks positive samples above negatives.  
- **PR-AUC** — summarizes the overall trade-off between precision and recall across all thresholds.

Leading CTR systems, such as **Google’s Wide & Deep**, **Facebook’s DCN**, and **Tencent’s DeepFM**,  
consistently report **AUC** and **LogLoss** as their primary metrics  
(Wang et al., *DCN V2: Improved Deep & Cross Network*, WWW 2021).

**Conclusion:**  
While threshold tuning and reweighting notably improved F1 and precision,  
the true measure of model quality lies in **ROC-AUC (≈ 0.81)** and **PR-AUC (≈ 0.14)** —  
demonstrating that the model effectively captures ranking and probability calibration essential for CTR prediction.
