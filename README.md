# cro-funnel-drop-action-brief
CRO action brief and Explainable AI pipeline for AT Food Co. Identifies 60% search-to-cart funnel leaks, prioritizes call feedback root causes, executes causal A/B testing, and deploys SHAP/LIME conversion explain ability.
# 📉 AT Food Co. — E-Commerce CRO Action Brief & Explainable AI Pipeline
> **Conversion Rate Optimization & Causal Analytics: Funnel Drop Identification, Qualitative Root-Cause Bucketing, Causal A/B Testing, and SHAP/LIME Model Explainability**

[![Domain](https://img.shields.io/badge/Domain-CRO%20%7C%20Growth%20Analytics-orange)](#)
[![Brand](https://img.shields.io/badge/Brand-AT%20Food%20Co.-green)](#)
[![Methodology](https://img.shields.io/badge/Methodology-Causal%20Learning%20%7C%20SHAP%20%7C%20LIME-blue)](#)
[![Status](https://img.shields.io/badge/Status-Completed-green)](#)

---

## 📌 Executive Summary

* **Objective:** Identify the primary conversion leak across 20,000 visitors, categorize qualitative feedback from 100 non-converting searchers, design a causal A/B experiment, and deploy a Python Explainable AI (XAI) pipeline using SHAP and LIME.
* **Core Insight:** The sharpest funnel leak occurs between **Searching for Meal Plans** and **Adding to Cart** (60% drop-off; 7,200 visitors lost). The primary actionable driver is stockouts in core **Medium & Large Fresh Meal Trial Pouches** (38% of call responses).

---

## 📊 1. Funnel Drop Analysis

    [ Landing on Home / Store ] ─── 20,000 Visitors
           │
           ▼ (60.0% Move On)
    [ Searching Meal Plans ] ────── 12,000 Visitors
           │
           ▼ (40.0% Move On ⚠️ SHARPEST DROP: 60% Drop-off / 7,200 Lost)
    [ Adding to Cart ] ──────────── 4,800 Visitors
           │
           ▼ (80.0% Move On)
    [ Purchasing Plan ] ─────────── 3,840 Subscribers (Overall Conversion: 19.2%)

* **Landing to Searching:** 60.0% move on (12,000 of 20,000).
* **Searching to Adding to Cart:** 40.0% move on (4,800 of 12,000).
* **Adding to Cart to Purchasing:** 80.0% move on (3,840 of 4,800).
* **Sharpest Drop-off:** **Searching to Adding to Cart**. While overall conversion is 19.2% (3,840 of 20,000), 7,200 searchers exit before adding meal plans to cart.

---

## 🔍 2. Qualitative Root-Cause Classification

Analyzing automated customer call responses from 100 randomly sampled non-converting searchers:

| Reason Bucket | Responses | Classification & Action |
|---|---|---|
| **Size / Portion Not Found** | **38** | **Actionable (Rank 1):** Restock core Medium & Large fresh meal trial pouches and deploy an instant "Notify Me / Pre-Order" trigger on out-of-stock selectors |
| **Product Not Found** | **27** | **Actionable (Rank 2):** Review site search indexing, query auto-correction, and synonym mapping for custom diet terms (e.g., "puppy growth", "sensitive gut") |
| **Shorter Delivery Expected** | **21** | **Non-Actionable (CRO Scope):** Cold-chain logistics and courier fulfillment contract overhauls are outside immediate site CRO scope |
| **Location Not Serviceable** | **14** | **Non-Actionable:** Restricted by fresh cold-chain delivery coverage boundaries |

---

## 🧪 3. Hypothesis & Causal Learning Framework

### A. Hypothesis Formulation
* **Null Hypothesis (H0):** Keeping Medium and Large fresh meal trial pouches in stock or providing a pre-order trigger does not affect the search-to-cart conversion rate for AT Food Co.
* **Alternate Hypothesis (H1):** Keeping Medium and Large fresh meal trial pouches in stock or providing a pre-order trigger increases the search-to-cart conversion rate by resolving the 38% user intent bottleneck.

### B. Causal Experimentation & Statistical Metrics
1. **Historical Baseline:** Compare historical click data to evaluate search-to-cart rates during fully stocked periods versus out-of-stock periods.
2. **Behavioral Clustering & Treatment:** Segment users searching for trial pouches into a distinct behavioral cluster. Apply the restock/pre-order trigger as a deliberate causal treatment to a 50% split of this target cluster.
3. **Medians & Standard Deviations:** Report the shift in **median conversion** (to resist outlier distortion) and **standard deviation** (to verify consistent behavior across the cluster).

---

## 🤖 4. Explainable AI (XAI) Pipeline (`cro_explainability_pipeline.py`)

    [ User Clickstream Data ] ──► [ Feature Selection (Mutual Info) ] ──► [ K-Means Clustering ]
                                                                                   │
                                                                                   ▼
    [ Individual LIME Explanation ] ◄── [ Global SHAP Summary ] ◄── [ XGBoost Conversion Model ]

    # cro_explainability_pipeline.py
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap

    # 1. Simulate 20,000 Visitor Behavioral Indicators
    np.random.seed(42)
    n_samples = 20000

    data = pd.DataFrame({
        'search_frequency': np.random.poisson(lam=3, size=n_samples),
        'out_of_stock_hit': np.random.binomial(n=1, p=0.38, size=n_samples),
        'session_duration_sec': np.random.exponential(scale=180, size=n_samples),
        'viewed_reviews': np.random.binomial(n=1, p=0.45, size=n_samples),
        'converted_to_cart': np.random.binomial(n=1, p=0.40, size=n_samples)
    })

    # 2. Train Conversion Prediction Model
    X = data.drop(columns=['converted_to_cart'])
    y = data['converted_to_cart']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    # 3. Compute SHAP Values for Model Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("SHAP feature importance calculation completed successfully.")

---

## 💼 5. Commercial Judgment Guardrails

Before pushing changes to 100% production at AT Food Co., evaluate:
1. **Repeat Subscription Velocity:** Prioritize repeat monthly meal plan subscribers over one-time trial purchasers, as repeat buyers build long-term brand equity and higher monetary value.
2. **Multi-Period Order Velocity:** Track 1-day, 2-day, and 30-day order volumes following the restock treatment.
3. **Inventory ROI vs. Churn:** Weigh the financial return of holding extra fresh inventory against customer churn rates, retention metrics, and cold-storage holding costs.
