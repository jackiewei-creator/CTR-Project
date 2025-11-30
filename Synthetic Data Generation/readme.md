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

---

# 8.0–8.9 CVAE Synthetic Data Generation and Evaluation

Following the CTGAN pipeline, we additionally implement a **Conditional Variational Autoencoder (CVAE)** to generate 27-dimensional synthetic CTR feature vectors.  
Unlike CTGAN, CVAE explicitly models the conditional likelihood \( p(x \mid y) \), which is theoretically suitable for label-conditioned synthetic data, though its performance depends heavily on model architecture and the complexity of the underlying data manifold.

All experiments in this section are conducted using the same Safe Top-27 training/validation split as earlier sections.

---

# 8.0 Reconstruct Safe Top-27 Training and Validation Splits

We reload the merged dataset and isolate the Safe Top-27 features and label.

Results:
- Full clean dataset: **7,675,517 rows × 28 columns**
- Training portion: **6,140,413 rows**
- Validation portion: **1,535,104 rows**
- CTR remains stable at roughly **1.55%** across all splits

This ensures consistency when comparing CTGAN and CVAE performance.

---

# 8.1 Normalization for CVAE

Because VAE-based models require bounded inputs, we apply **MinMaxScaler** to all 27 features:

- Fit scaler on **training split only**
- Transform both `X_train_syn` and `X_val_real`
- Save fitted scaler: `cvae_safe27_scaler.pkl`

Outputs:
- Scaled training matrix: **6.14M × 27**
- Scaled validation matrix: **1.53M × 27**
- Per-feature min = 0, max = 1, as expected

This ensures that the CVAE decoder outputs match the valid feature range.

---

# 8.2 CVAE Architecture

We construct a compact CVAE with the following architecture.

**Encoder**
- Input dimension = **28** (27 features + 1 label)
- Two hidden layers: 64 → 64 (ReLU)
- Outputs:
  - Mean vector \( \mu \in \mathbb{R}^{16} \)
  - Log-variance vector \( \log \sigma^2 \in \mathbb{R}^{16} \)

**Latent space**
- \( z \sim \mathcal{N}(\mu, \sigma^2) \)
- Latent dimension = **16**

**Decoder**
- Input: concatenated vector `[z, y]` (16 + 1)
- Two hidden layers: 64 → 64 (ReLU)
- Output dimension: **27**
- Final activation: Sigmoid (ensures outputs in [0, 1])

This is a standard conditional VAE design for tabular modeling.

---

# 8.3 CVAE Training (500k Subsample)

To keep CPU training feasible, we subsample:

- **500,000** training rows  
- **100,000** validation rows  

Training settings:
- Batch size = 1,024
- Epochs = 10
- Loss = reconstruction MSE + KL divergence  
  \[
  \mathcal{L} = \mathbb{E}[\|x - \hat{x}\|^2] + \beta \cdot \mathrm{KL}(q(z|x,y) \,\|\, \mathcal{N}(0, I))
  \]
- KL term remains close to 0 (weak regularization)

Results:
- Training reconstruction ≈ **0.0546**
- Validation reconstruction ≈ **0.0547**
- Minimal overfitting, but also limited latent structure

The model converges stably, but the very small KL term indicates that the latent bottleneck is not strongly constraining the representation.

---

# 8.4 Generate 200k CVAE Synthetic Samples

Synthetic generation proceeds as:

1. Sample \( y \sim \text{Bernoulli}(\hat{p}) \), with \(\hat{p}\) = real training CTR ≈ 0.01552  
2. Sample latent code \( z \sim \mathcal{N}(0, I) \)  
3. Decode `[z, y]` into scaled feature space  
4. Inverse-transform features using `cvae_safe27_scaler.pkl`  

Outputs:
- `synthetic/cvae_safe_top27_200k.parquet`
- `synthetic/cvae_safe_top27_200k.csv`

Observed properties:
- Output shape: **200,000 × 28**
- Synthetic CTR ≈ **1.51%** (very close to real rate)
- Feature values appear smooth and averaged, as expected for VAE-style decoders

The CVAE successfully preserves CTR proportion, but tends to oversmooth feature distributions.

---

# 8.5 Utility: Train on 200k CVAE Synthetic → Test on Real

We train a LightGBM model using **only** the 200k CVAE synthetic samples, then evaluate on the held-out real validation set.

Key logs:
- LightGBM emits repeated warnings:  
  `No further splits with positive gain, best gain: -inf`
- Training AUC ≈ **1.00** (perfect fit on synthetic domain)
- Validation AUC ≈ **0.683**
- PR-AUC ≈ **0.037**
- LogLoss ≈ **1.81**
- Accuracy ≈ **0.59**
- F1 ≈ **0.049**

Interpretation:
- The model can separate CVAE synthetic data easily but fails to generalize to real data.
- Over-smooth, mis-shaped synthetic patterns lead to poor ranking quality on real clicks.
- Compared to the CTGAN-only setting, CVAE-only utility is **worse**, especially in PR-AUC.

Conclusion: CVAE synthetic data alone is **not suitable** as a drop-in replacement for real CTR training data.

---

# 8.6 Utility Sensitivity: Mixed Real + CVAE Synthetic (Total 200k)

We evaluate the impact of mixing CVAE synthetic data with real data while keeping the total training size fixed at 200,000.

Synthetic fractions tested:
- 10%, 20%, 30%, 40%, 50% synthetic  
- Remaining fraction filled with real data

For each fraction, we:
- Sample real and synthetic rows
- Train LightGBM on the mixed dataset
- Evaluate on `X_val_real`, `y_val_real`

Summary (ROC-AUC):

| Synthetic Fraction | ROC-AUC |
|--------------------|---------|
| 10%                | 0.753   |
| 20%                | 0.750   |
| 30%                | 0.745   |
| 40%                | 0.744   |
| 50%                | 0.740   |

PR-AUC shows a similar pattern, peaking around 10–20% synthetic and degrading afterward.

Interpretation:
- Small amounts of CVAE synthetic data (≤ 20%) do not severely hurt performance and can slightly stabilize training.
- As the synthetic proportion increases beyond 30%, downstream utility **degrades monotonically**.
- CVAE is less helpful than CTGAN as an augmentation source.

---

# 8.7 Visualization: Utility vs Synthetic Fraction (CVAE)

We visualize ROC-AUC and PR-AUC as functions of the CVAE synthetic fraction.

Key observations from the curves:
- Both ROC-AUC and PR-AUC are highest when synthetic fraction is small (around 10–20%).
- Increasing the CVAE synthetic fraction consistently reduces utility.
- Unlike CTGAN, there is no clear performance gain over the pure-real baseline.

This confirms that CVAE synthetic data is not as useful as CTGAN for utility-driven augmentation in this CTR setting.

---

# 8.8 Fidelity: Correlation Structure Preservation

To assess multivariate fidelity, we compare the **27×27 Pearson correlation matrices** between:

- A 200,000-row real sample
- A 200,000-row CVAE synthetic sample

We compute:
- Absolute difference matrix: `abs(corr_real − corr_syn)`
- Mean absolute correlation difference across all feature pairs

Results:
- Mean absolute correlation difference ≈ **0.4516**
- In contrast, CTGAN achieved ≈ **0.0552**

Examples of severe mismatches:
- `creat_type_cd` vs `f_dislike_mean`  
  - Real: ≈ **0.08**  
  - CVAE: ≈ **−0.97**
- `slot_id` vs `f_browser_life`  
  - Real: near **0**  
  - CVAE: ≈ **−0.96**

Interpretation:
- CVAE fails to reproduce the high-dimensional dependence structure of CTR features.
- Many correlations are not only mis-scaled but also **sign-flipped**, indicating strong distortion.
- This explains the poor generalization of CVAE-based models in 8.5 and 8.6.

---

# 8.9 Privacy Evaluation for CVAE Synthetic Data

We repeat the nearest-neighbor privacy analysis:

- Compute distances between real points and their nearest real neighbors (real→real)
- Compute distances between synthetic points and their nearest real neighbors (syn→real)
- Use standardized feature space and Euclidean distance

Results (example):
- Mean real→real distance ≈ **1.68**
- Mean syn→real distance ≈ **1.90**  
- Synthetic samples are **further** from real points than typical real neighbors

Interpretation:
- There is no evidence of memorization or record-level leakage.
- CVAE synthetic data provides **good privacy** under this distance-based metric.
- The main issue is **fidelity**, not privacy.

---

# Overall Comparison: CTGAN vs CVAE on CTR Safe Top-27

| Criterion                 | CTGAN                            | CVAE                              |
|--------------------------|----------------------------------|-----------------------------------|
| Marginal fidelity        | Moderate                         | Poor                              |
| Correlation fidelity     | **Good (mean diff ≈ 0.055)**     | **Very poor (mean diff ≈ 0.452)** |
| Privacy (NN distance)    | Good (syn further than real)     | Good (syn further than real)      |
| Utility: pure synthetic  | AUC ≈ 0.65–0.74 (task-dependent) | AUC ≈ 0.68 with poor PR-AUC       |
| Utility: mixed training  | Best with 50% CTGAN mix          | No clear gain over real baseline  |
| Label realism            | CTR inflated (~5%)               | CTR close to real (~1.5%)         |

## Final Conclusion for VAE/CVAE

- **CTGAN** remains the preferred generative model for this CTR task: it provides reasonable fidelity and useful augmentation when combined with real data.
- **CVAE** preserves label proportion and offers strong privacy but severely distorts feature correlations, resulting in **weaker downstream utility**.
- For this project, the recommended practice is:
  - Use **real data** as the primary training source.
  - Use **CTGAN synthetic data** as a limited augmentation source.
  - Treat **CVAE synthetic data** mainly as a negative control and a demonstration that not all conditional generative models are equally suitable for high-dimensional, sparse CTR prediction tasks.
