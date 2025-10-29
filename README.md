# CTR-Project
*Cross-Domain Click-Through Rate Prediction based on Ads and Feeds Behavioral Data*

## 1. Overview  
This project develops a **CTR prediction pipeline**, integrating user behavioral data from two domains — **Ads (target)** and **Feeds (source)** — to model and understand the relationship between user engagement patterns and advertising click-through probability.  

Data Source: [CTR Prediction - 2022 DIGIX Global AI Challenge](https://www.kaggle.com/datasets/xiaojiu1414/digix-global-ai-challenge)


## 2. Presentation Slides  

📎 [Statistics C261 Midterm Slides-The Regressers](https://docs.google.com/presentation/d/1ROU-wOxUzBlG5TPCH9s2_08X0SAMlbQT/edit?usp=sharing&ouid=105892605078401013159&rtpof=true&sd=true)  

## 3. Workflow
1. **Data Processing** — Cleaning, aggregation, and feature preparation.  
   Prepare cross-domain datasets and user-level aggregation.  
   Main notebooks:  
   [Data Cleaning Notebook](./Data%20Processing/Data%20Cleaning.ipynb)  
   [Feeds Aggregation Notebook](./Data%20Processing/Feeds_Aggregation.ipynb)  
   [Merge and EDA Notebook](./Data%20Processing/merge_and_EDA.ipynb)  
   
3. **Modeling**
   - Baseline CTR models with Logistic Regression and Random Forest
   - Optimization Model: LightGBM, XGBoost

4. **Evaluation & Future Work**
   - Model performance and expansion plans.  
   - Analyze feature importance, interpret user engagement, and extend to deep models.
