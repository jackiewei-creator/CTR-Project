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
