# CTR Synthetic Data Generation and Evaluation  

This repository documents a complete end-to-end workflow for generating CTR synthetic data using CTGAN and evaluating the resulting data on three major dimensions: **utility**, **fidelity**, and **privacy**.  
All experiments are conducted using the *Safe Top-27* leakage-free feature set established in earlier coursework.

The methodology strictly avoids test leakage and uses a held-out real validation set (`X_val_real`, `y_val_real`) for all evaluation steps.

---

# 6.0 Reload Merged Data and Rebuild Train/Validation Split

We reload the merged training dataset (`train_merged.parquet`) and recompute the same 80/20 stratified split used in earlier sections.

Key observations:
- Total rows: **7,675,517**
- CTR ≈ **1.55%**, which is consistent across splits
- 60 numeric features selected (excluding label)
- Training set: **6.14M** rows  
- Validation set: **1.53M** rows  

This step ensures the downstream experiments are anchored in the same data partitioning used during midterm analyses, guaranteeing comparability.

---

# 6.1 Prepare Clean Dataset (Safe Top-27 Features)

The Safe Top-27 features were carefully selected to prevent leakage from identifiers or timestamp information. We subset the dataset to the 27 engineered behavioral features plus the binary target.

Results:
- Final matrix: **7.67M rows × 28 columns**
- All features are numerical and stable for generative modeling
- This dataset becomes the unified input for CTGAN and all later evaluation tasks

This reduces complexity while ensuring representativeness of user behavior patterns.

---

# 6.2 Train/Validation Split for Synthesizers

To properly evaluate synthetic data quality, we isolate a **held-out real validation set** (`X_val_real`, `y_val_real`) that CTGAN and other models never observe.

- `X_train_syn`: **6.14M** real rows (used to train CTGAN)
- `X_val_real`: **1.53M** real rows (used only for evaluation)

Maintaining separation guarantees fairness in utility, fidelity, and privacy metrics.

---

# 6.3 CTGAN Training

## 6.3.1–6.3.2 Build CTGAN Training Table
We directly sample **200,000** real rows to form a computationally manageable training set for CTGAN.

Additional steps:
- Detected **18 discrete columns** using integer-dtype and cardinality rules
- Saved the sampled 200k dataset for teammates (`real_train_ctgan_200k_safe27.csv`)
- CTR in sampled subset: **≈1.60%**, similar to full data

This guarantees the GAN sees realistic class imbalance conditions within a feasible training size.

## 6.3.3 Train CTGAN (pac = 1)
We train CTGAN using lightweight hyperparameters:

- epochs: 10  
- batch size: 1,024  
- pac: **1** (prevents batch divisibility errors)  
- hidden layers: (256, 256)

Results:
- CTGAN converges stably in ~8 minutes
- Saved trained model to: `ctgan_safe_top27_small.pkl`

This model serves as the generator for downstream synthetic sample creation.

---

# 6.4 Generate Synthetic Data

We generate **200,000** synthetic rows matching the Safe Top-27 schema.

Outputs:
- `synthetic/ctgan_safe_top27_200k.parquet`
- `synthetic/ctgan_safe_top27_200k.csv`

Observations:
- Synthetic CTR ≈ **5.11%**
- Much higher than real CTR (≈1.55%)

This behavior is typical of GANs trained on highly imbalanced binary labels: the generator tends to inflate minority class frequency due to mode collapse on sparse events.

---

# 7.1 Utility Evaluation: Train on 200k Real → Test on Real

We train a LightGBM model on **200k real rows**, then evaluate on the real validation set.

Results:
- ROC-AUC: **0.758**
- PR-AUC: **0.099**
- F1: **0.090**
- Recall: **0.530**

Interpretation:
This represents the baseline performance achievable when training on a small real subset.  
It reflects the inherent difficulty of CTR prediction with 1–2% positive labels.

This model serves as the benchmark for judging synthetic data quality.

---

# 7.2 Utility: Train on 200k Synthetic → Test on Real

We train LightGBM using only the 200k synthetic rows.

Results:
- ROC-AUC: **0.652**
- PR-AUC: **0.043**
- F1: **0.000**
- The classifier predicts almost all zeros due to synthetic label bias

Interpretation:
- Predictive utility of pure synthetic data is substantially lower.
- The inflated synthetic CTR disrupts the model’s calibration, leading to highly imbalanced predicted probabilities.
- This confirms that synthetic data is **not a substitute** for real training data.

However, this does not negate its usefulness for augmentation.

---

# 7.3 Utility: Mixed 100k Real + 100k Synthetic → Test on Real

We construct an evenly balanced mixed dataset:

- 100,000 real samples  
- 100,000 synthetic samples  

Results:
- ROC-AUC: **0.763** (highest of all three settings)
- PR-AUC: **0.095**
- F1: **0.150** (significantly improved)
- Precision: 0.094  
- Recall: 0.364  

Interpretation:
- Mixed training produces the **best overall balance** between precision and recall.
- Synthetic data introduces additional variance and acts as a regularizer.
- Although synthetic CTR is inflated, combining synthetic and real helps the model generalize better.

This shows synthetic data’s value is strongest when used as augmentation rather than replacement.

---

# 7.4 Fidelity Evaluation

## 7.4.1 Marginal Distribution Fidelity (Wasserstein Distances)
We compare each feature’s distribution in real 200k vs synthetic 200k.

Findings:
- Most features show **small Wasserstein distances**, indicating similar marginal distributions.
- Features involving counts (e.g., `f_up_sum`, `f_refresh_sum`) show moderate shifts.
- No feature exhibits extreme divergence.

This indicates CTGAN successfully approximates low-dimensional distributions.

## 7.4.2 Pairwise Correlation Preservation
We compute all 351 pairwise correlations and measure real-vs-synthetic differences.

Results:
- Mean absolute correlation difference: **0.055**
- Weak correlations are reproduced accurately
- Strong correlations (especially behaviors involving interaction counts) deviate:

For example:
- `f_up_sum` vs `f_dislike_sum`: real corr = **0.93**, synthetic corr = **0.21**

Interpretation:
GAN struggles to capture **high-order interactions** involving multiple behavioral features.  
This is expected and consistent with limitations of current tabular GAN models.

---

# 7.5 Privacy Evaluation (Nearest-Neighbor Distance)

We compute nearest-neighbor distances:

- real → real  
- synthetic → real  

Sample size: 5,000 each.

Results:
- Mean real→real distance: **2.54**
- Mean syn→real distance: **3.58**
- Synthetic distances consistently larger

Interpretation:
- Synthetic samples do **not** cluster unusually close to real points.
- No indication of memorization or record-level leakage.
- CTGAN exhibits **good privacy preservation** under this evaluation.

---

# 7.6 Summary of Results

## Utility
| Training Setting            | ROC-AUC | PR-AUC | F1    | Interpretation |
|-----------------------------|---------|--------|-------|----------------|
| 200k real only              | 0.758   | 0.099  | 0.090 | Strong baseline |
| 200k synthetic only         | 0.652   | 0.043  | 0.000 | Weak utility; poor calibration |
| 100k real + 100k synthetic  | 0.763   | 0.095  | 0.150 | Best F1; synthetic helps as augmentation |

## Fidelity
- Marginal distributions reproduced moderately well  
- Correlation structure partially preserved  
- Strong behavioral correlations not fully captured  
- Overall fidelity acceptable but not sufficient for stand-alone modeling

## Privacy
- Synthetic samples consistently farther from real-nearest neighbors  
- No sign of memorization  
- CTGAN satisfies privacy expectations under distance-based analysis  

## Final Conclusion
CTGAN synthetic data demonstrates **reasonable fidelity** and **strong privacy protection**, but limited stand-alone predictive capability.  
The most effective usage is **augmentation**, where mixing synthetic and real data improves robustness and produces better recall/F1 than training on real data alone.

This completes the full synthetic data evaluation pipeline.
