# CTR Project – Data Processing Phase

This repository documents the **data processing workflow** for ad click-through rate (CTR) prediction based on the [Digix Global AI Challenge – Digital Marketing Dataset](https://www.kaggle.com/datasets/xiaojiu1414/digix-global-ai-challenge).

The objective of this phase is to prepare cross-domain behavioral data from the ads (target) and news feeds (source) domains through three steps—(1) clean and standardize all raw tables, (2) aggregate feeds logs into per-user profiles, and (3) merge these profiles back into ads records—producing a consistent, analysis-ready dataset for EDA and downstream CTR modeling.

---

## Overview of the Data Processing Phase

The data processing stage is divided into three sequential notebooks:

| Notebook | Title | Description |
|-----------|--------|-------------|
| **1_DataCleaning.ipynb** | Data Cleaning and Preprocessing | Cleans and normalizes raw ads and feeds datasets to ensure structural consistency and type integrity. |
| **2_FeedsAggregation.ipynb** | Feed-Level User Aggregation | Aggregates news feed logs per user to create user-level behavioral profiles (planned). |
| **3_AdsMerge_Modeling.ipynb** | Ads–Feeds Integration and Feature Alignment | Merges aggregated user profiles with ads data to form the final model input (planned). |



---

## 1. Data Cleaning and Preprocessing

### Objective
To prepare the four raw datasets —  
`train_data_ads.csv`, `test_data_ads.csv`, `train_data_feeds.csv`, and `test_data_feeds.csv` —  
into clean, type-consistent, and analysis-ready tables.

### Main Steps
1. **Dataset inspection**  
   Verified structure, column names, and dimensions; identified `label` in `train_data_ads` as the target variable.

2. **Standardized cleaning rules**  
   - Missing values: numeric → `-1`, categorical → `"unknown"`, multi-value strings → `""`.  
   - Data types: IDs/enums → `category`; numeric fields retained as `int` or `float`.  
   - Temporal features: extracted `t_hour`, `t_wday`, and `t_is_weekend` from timestamps (`pt_d` and `e_et`).  
   - Multi-value fields: computed summary statistics (`_len`, `_uniq`) for caret-separated entries.

3. **Ads domain cleanup**  
   Generated 17 derived columns, including temporal and behavioral summaries.  
   Ensured consistent schema between training and test datasets.  
   Observed class imbalance (CTR ≈ 1.55%).

4. **Feeds domain cleanup**  
   Created 11 derived columns representing user activity and interest diversity.  
   Standardized schema between train/test splits.  

5. **Outputs**  
   Cleaned data were exported in Parquet format for the next phase

## 2. Feeds Aggregation and User Profiling

### Objective
To transform feed-level behavior logs into **per-user profiles** that summarize user activity, interests, feedback, and temporal patterns.  
This enables cross-domain integration between user behavior (feeds) and advertising responses (ads).

### Main Steps
1. **Dataset loading and validation**  
   Loaded `feeds_train_clean.parquet` and `feeds_test_clean.parquet`.  
   Verified structural consistency and key identifiers (`u_userId`, `t_hour`, `i_entities_len`).

2. **User-level aggregation logic**  
   Grouped by `u_userId` to compute:
   - **Activity level:** feed count (`f_rows`), refresh frequency (`f_refresh_mean`, `f_refresh_sum`).  
   - **Interest breadth:** unique news categories (`f_cat_uniq`), average entity count (`f_entities_len_mean`).  
   - **Feedback behavior:** upvotes (`f_up_mean`, `f_up_sum`) and dislikes (`f_dislike_mean`, `f_dislike_sum`).  
   - **Temporal preference:** average and peak browsing hours (`f_avg_hour`, `f_peak_hour`).  
   - **Device traits:** median phone price and most frequent browser mode.

3. **Feature enhancement (⭐ Innovation)**  
   Introduced **cyclic time features** to capture daily periodicity:
   - `f_hour_sin = sin(2π × f_avg_hour / 24)`  
   - `f_hour_cos = cos(2π × f_avg_hour / 24)`  
   These features help models understand continuous time-of-day patterns (e.g., night vs. morning users).

4. **Light EDA for validation**  
   Visualized user-level feature distributions:
   - User activity and interest diversity showed heavy-tailed patterns.  
   - Peak browsing hours concentrated between 19:00–22:00.  
   - Device prices were stable (median ≈ 13–14).  
   These results confirmed data quality and realistic user behavior.

5. **Outputs**  
   Exported aggregated datasets for model integration:
   - `feeds_user_agg_train.parquet`  
   - `feeds_user_agg_test.parquet`  

## 3. Merge and Exploratory Data Analysis (EDA)

### Objective
To integrate ad-level data (`train_data_ads`) with user-level features aggregated from feeds (`feeds_user_agg`), enabling a cross-domain analysis of how user behaviors relate to ad click-through rates (CTR).

### Main Steps
**Data merging**
- Merged `train_data_ads` with `feeds_user_agg_train` and `test_data_ads` with `feeds_user_agg_test`.
- Joined on the user identifier (`user_id` ↔ `u_userId`).
- Verified column alignment and ensured no missing values were introduced.

**Post-merge verification**
- Confirmed consistent feature types and dataset dimensions:
  - Training set: 7.68 M rows × 68 columns  
  - Test set: 0.98 M rows × 67 columns
- Inspected key numerical features (f_rows, f_refresh_mean, f_up_sum, etc.) — no anomalies detected.

**Exploratory Data Analysis (EDA)**
- **Correlation heatmap:** Identified feature clusters with strong internal correlation (e.g., `_len` vs. `_uniq`, `_sum` vs. `_mean`).
- **CTR correlations:** Found positive correlations between CTR (`label`) and feed-based behavior metrics such as:
  - `f_up_mean` (+0.08): users with more upvotes are more likely to click ads.
  - `f_dislike_mean` (+0.06): even active dislikers show higher CTR, indicating general engagement.
  - `f_hour_cos` (+0.03): users active at certain hours show slightly higher CTR.
- **Boxplots:** Visualized behavioral and device-related differences between clicked (1) and non-clicked (0) samples, revealing distinct patterns in upvote and refresh behavior.
- **Data export:** Final merged datasets were saved in `outputs_merged/` for modeling.

### Outputs
- `train_merged.parquet` — merged training dataset (ads + feeds features)
- `test_merged.parquet`   — merged testing dataset ready for inference

---

