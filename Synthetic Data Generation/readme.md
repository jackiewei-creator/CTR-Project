# CTR Synthetic Data Generation and Evaluation


This repository presents the full experimental pipeline for generating CTR tabular synthetic data using CTGAN and evaluating the results from the perspectives of utility, fidelity, and privacy. All processing is based on the Safe Top-27 non-leakage features identified earlier in the midterm stage of the project.

The workflow includes data reconstruction, CTGAN training, synthetic sample generation, LightGBM utility comparison, distribution and correlation fidelity checks, and nearest-neighbor privacy analysis.

---

## 6.0 Reload Merged Data and Rebuild Train/Validation Split
The project begins by reloading the merged training dataset (`train_merged.parquet`) and reconstructing the same 80/20 stratified split used in earlier sections.

Key results:
- Loaded 7.67M rows, CTR ≈ 1.55%.
- Selected 60 numeric features.
- Train/valid partition:  
  - Training: 6.14M rows  
  - Validation: 1.53M rows  
- Validation CTR preserved exactly.

---

## 6.1 Prepare Clean Dataset (Safe Top-27)
We select the previously identified Safe Top-27 features (after removing identifiers and timestamp-related leakage variables).  
The cleaned dataset consists of the 27 features plus the binary label.

Key results:
- Final dataset: 7,675,517 rows × 28 columns.
- All features are numeric, leakage-free, and appropriate for generative modeling.

---

## 6.2 Train/Validation Split for Generative Models
To ensure unbiased evaluation, a separate split is created specifically for synthesizer training:

- `X_train_syn` (6.14M rows) is used for CTGAN.
- `X_val_real` (1.53M rows) is held out for fidelity, utility, and privacy evaluation only.

Both splits maintain the original CTR rate (≈1.55%).

---

## 6.3 CTGAN Training

### 6.3.1–6.3.2 Build CTGAN Training Table
We directly sample 200,000 real rows from `df_clean` to form a manageable CTGAN training subset.  
Discrete columns are automatically identified based on integer dtype and limited cardinality.

Key results:
- CTGAN training subset: 200,000 rows.
- 18 columns identified as discrete.
- Saved for teammates as: `real_train_ctgan_200k_safe27.csv`.

### 6.3.3 Train CTGAN (pac = 1)
CTGAN is trained using:
- 10 epochs
- batch size = 1024
- generator/discriminator hidden layers = (256, 256)
- pac = 1 (to avoid batch divisibility issues)

Key results:
- Training completed stably.
- Saved model: `ctgan_safe_top27_small.pkl`.

---

## 6.4 Generate Synthetic Data
Using the trained CTGAN model, we generate 200,000 synthetic samples with the same schema as the Safe Top-27 feature set plus the label column.

Outputs:
- Parquet: `synthetic/ctgan_safe_top27_200k.parquet`
- CSV: `synthetic/ctgan_safe_top27_200k.csv`

Key observations:
- Synthetic CTR ≈ 5.11%, higher than real CTR.  
  This is expected due to the difficulty of reproducing extreme class imbalance.

---

## 7.1 Utility Evaluation: Train on 200k Real → Test on Real
We train LightGBM using 200,000 real samples and evaluate on the real validation set.  
This acts as the baseline for comparison.

Results:
- ROC-AUC: 0.758  
- PR-AUC: 0.099  
- F1: 0.090  

This represents the expected performance upper bound for models trained on synthetic data.

---

## 7.2 Utility: Train on 200k Synthetic → Test on Real
We train LightGBM solely on the 200,000 synthetic samples and evaluate on the same real validation set.

Results:
- ROC-AUC: 0.652  
- PR-AUC: 0.044  
- F1: 0.000  

Pure synthetic data does not achieve competitive predictive performance, which is consistent with limitations of GAN-based tabular modeling in extreme class imbalance scenarios.

---

## 7.3 Utility: Mixed Training (100k Real + 100k Synthetic)
We construct a 50/50 mixed training set and train LightGBM.

Results:
- ROC-AUC: 0.763  
- PR-AUC: 0.095  
- F1: 0.150  

The mixed model matches or slightly exceeds the performance of the real-only baseline, especially in recall-related metrics.  
This suggests that synthetic data can enhance robustness when combined with real data.

---

## 7.4 Fidelity Evaluation

### 7.4.1 Marginal Distribution Fidelity (Wasserstein Distance)
We compute the Wasserstein distance for each of the 27 features between real (200k) and synthetic (200k) samples.

Observations:
- Most features have small distribution differences.
- A few high-variance behavioral features exhibit larger divergence.

### 7.4.2 Pairwise Correlation Preservation
We compute all 351 unique pairwise correlations in:

- real 200k
- synthetic 200k  

Then compute the absolute differences.

Key results:
- Mean absolute correlation difference: 0.055.
- Weak and medium correlations are preserved well.
- Strong behavior-related correlations (e.g., `f_up_sum`, `f_dislike_sum`, `f_rows`) show the largest differences.

Synthetic data captures global structure but misses some high-order feature interactions.

---

## 7.5 Privacy Evaluation (Nearest-Neighbor Distance)
We compare:

- distances from 5,000 real samples → nearest real sample  
- distances from 5,000 synthetic samples → nearest real sample  

Key results:
- Mean real→real distance: 2.54  
- Mean synthetic→real distance: 3.58  

Synthetic samples are consistently farther from real data than real points are from each other.  
This indicates no evidence of memorization and good privacy preservation.

---

## 7.6 Summary of Findings

### Utility
| Training Setting | ROC-AUC | PR-AUC | F1 | Notes |
|------------------|---------|--------|----|-------|
| 200k real only | 0.758 | 0.099 | 0.090 | Strong baseline |
| 200k synthetic only | 0.652 | 0.044 | 0.000 | Weak utility |
| 100k real + 100k synthetic | 0.763 | 0.095 | 0.150 | Best overall; synthetic helps augment |

### Fidelity
- Marginal feature distributions generally close.
- Correlation structure partly preserved; mean abs difference = 0.055.
- Several strong behavioral couplings not captured accurately.

### Privacy
- Nearest-neighbor analysis shows synthetic samples are farther from real points.
- No indication of overfitting or record-level memorization.

### Overall Conclusion
CTGAN-generated synthetic data offers reasonable fidelity and strong privacy protection, but its standalone predictive utility is limited.  
However, when combined with real data, synthetic samples can improve model robustness and yield competitive performance relative to real-only training.

---
