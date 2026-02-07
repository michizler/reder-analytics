# Reder Analytics — Customer Churn Prediction

<div align="center">

**Predicting and Preventing Customer Attrition with Machine Learning**

_A proof-of-concept solution for Reder Telecommunications_

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)

</div>

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Business Context](#business-context)
- [The Challenge](#the-challenge)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
  - [Source Data](#source-data)
  - [Data Cleaning & Normalization](#data-cleaning--normalization)
  - [Feature Engineering](#feature-engineering)
  - [Encoding Strategy](#encoding-strategy)
- [Model Development](#model-development)
  - [Feature Selection](#feature-selection)
  - [Baseline Model](#baseline-model)
  - [Optimized Model](#optimized-model)
  - [Evaluation Results](#evaluation-results)
- [API Service](#api-service)
- [Presentation Layer](#presentation-layer)
- [Cloud Deployment Estimates](#cloud-deployment-estimates)
- [Roadmap](#roadmap)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Executive Summary

Reder Telecommunications, an Oslo-based internet and mobile services provider with over 2,000 active customers, is experiencing a **15% annual churn rate** — a figure that is trending upward. Customer acquisition costs run **5–7× higher** than retention costs, making each departure a compounding financial loss.

This project delivers a **machine learning-powered churn prediction system** that identifies at-risk customers before they leave. The trained model achieves **96.9% accuracy** with balanced precision and recall, and is deployed through a lightweight **FastAPI REST endpoint** capable of returning real-time churn probabilities in milliseconds.

The business objective: **reduce churn by 10% within 12 months** by shifting from reactive, one-size-fits-all retention to proactive, data-driven engagement strategies.

---

## Business Context

| Detail                 | Value                                              |
| ---------------------- | -------------------------------------------------- |
| **Company**            | Reder Telecommunications                           |
| **Headquarters**       | Oslo, Norway                                       |
| **Founded**            | 2005                                               |
| **Services**           | Internet (broadband/fiber) and mobile (voice/data) |
| **Customer Base**      | 2,000+ active subscribers                          |
| **Current Churn Rate** | ~15% annually (increasing)                         |

Reder operates in a competitive Scandinavian telecom market where customers have multiple alternatives and switching costs are low. The company needed a way to move beyond aggregate reporting and understand churn at the individual customer level.

**Industry precedent supports this approach.** Companies like Netflix, Spotify, and Amazon have long used machine learning to predict attrition and personalize retention — the same principles applied here at a scale appropriate to Reder's operations.

---

## The Challenge

Three external drivers are accelerating customer departure:

1. **Intense Competition** — Multiple providers offer comparable services at competitive or lower price points, creating a volatile, price-sensitive market where differentiation is difficult.

2. **Network Quality Issues** — Intermittent service interruptions and inconsistent coverage directly erode customer confidence and satisfaction.

3. **Shifting Customer Expectations** — Today's subscribers expect personalized service, transparent pricing, and proactive communication — standards Reder has not fully met.

These external pressures are compounded by four internal obstacles:

| Obstacle                     | Description                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Limited Customer Insight** | Only aggregate churn rates were tracked — no per-segment or per-customer analytics existed.                    |
| **Pricing Inconsistencies**  | Customers with similar usage patterns were on different plans, creating perceptions of unfairness.             |
| **Network Quality Gaps**     | No predictive monitoring system was in place to anticipate or address outages before impact.                   |
| **Reactive Retention**       | Interventions only occurred after a customer signaled intent to leave — by which point recovery rates are low. |

The core problem: **Reder was losing customers it could have saved, because it didn't know who was at risk until it was too late.**

---

## Solution Approach

The solution follows a standard end-to-end machine learning workflow:

```
Raw Data → Cleaning → Feature Engineering → Feature Selection → Model Training → API Deployment
```

**Offline Pipeline (batch):**
Ingest raw customer data → parse nested JSON columns → normalize into relational structure → engineer RFM features → encode categorical variables → select top features via mutual information → train and tune a Decision Tree classifier → serialize the model.

**Online Pipeline (real-time):**
Customer attributes arrive via POST request → Pydantic validates the schema → the pre-trained model scores the input → a churn prediction (binary) and probability (0–1) are returned.

---

## Project Structure

```
reder-analytics/
│
├── source-data/
│   └── Dataset.xlsx - Sheet1.csv      # Raw dataset (2,000+ records, 20+ columns)
│
├── preprocessing/
│   └── clean.ipynb                    # Data cleaning & feature engineering notebook
│
├── model/
│   ├── model.ipynb                    # Model training & evaluation notebook
│   ├── model.pkl                      # Serialized trained model (Decision Tree)
│   ├── data.json                      # Feature column list for inference
│   └── churn_data_clean.csv           # Cleaned, encoded dataset (12,483 × 55)
│
├── reder-presentation/                # Interactive slide deck (React + Vite)
│   ├── src/
│   │   ├── App.jsx                    # 23-slide presentation component
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── app.py                             # FastAPI prediction endpoint
├── requirements.txt                   # Python dependencies
├── .env                               # Environment configuration
└── README.md                          # This file
```

---

## Data Pipeline

### Source Data

The raw dataset (`Dataset.xlsx - Sheet1.csv`) contains **2,000+ customer records** with the following attributes:

| Category       | Fields                                                   |
| -------------- | -------------------------------------------------------- |
| **Identity**   | CustomerID, Name, Age, Gender                            |
| **Contact**    | Email, Phone, Address, Location                          |
| **Account**    | Segment, SubscriptionDetails, PurchaseHistory            |
| **Behavior**   | ServiceInteractions, WebsiteUsage, ClickstreamData       |
| **Financial**  | PaymentHistory                                           |
| **Engagement** | EngagementMetrics, Feedback, NPS, MarketingCommunication |
| **Target**     | ChurnLabel (binary: 0 = retained, 1 = churned)           |

**Key challenge:** Nine of these columns violate first normal form (1NF) — they store nested JSON structures (lists of dictionaries or single dictionaries) as string literals within cells. This required a multi-step normalization process before any analysis could begin.

### Data Cleaning & Normalization

All preprocessing is documented in `preprocessing/clean.ipynb`.

**Step 1 — Parse string literals into Python objects**

The nested columns were stored as string representations of Python data structures. The `ast.literal_eval` function safely converts them:

```python
import ast

def convert_string(col):
    if isinstance(col, str):
        return ast.literal_eval(col)
    return col

cols_to_extract = [
    'PurchaseHistory', 'SubscriptionDetails', 'ServiceInteractions',
    'PaymentHistory', 'WebsiteUsage', 'ClickstreamData',
    'EngagementMetrics', 'Feedback', 'MarketingCommunication'
]

for col in cols_to_extract:
    customer_data[col] = customer_data[col].apply(convert_string)
```

**Step 2 — Normalize nested columns into separate DataFrames**

A reusable `normalize_column()` function handles both data shapes:

- **List of dictionaries** (one-to-many) — explodes the list, then applies `pd.json_normalize` to flatten each dictionary into columns.
- **Single dictionary** (one-to-one) — directly applies `pd.json_normalize`.

Each normalized column becomes its own DataFrame, linked back to the parent record via `CustomerID`:

```python
sub_details  = normalize_column(customer_data, "SubscriptionDetails")
srv_interac  = normalize_column(customer_data, "ServiceInteractions")
payment_hist = normalize_column(customer_data, "PaymentHistory")
web_usage    = normalize_column(customer_data, "WebsiteUsage")
clk_stream   = normalize_column(customer_data, "ClickstreamData")
eng_metrics  = normalize_column(customer_data, "EngagementMetrics")
feedback     = normalize_column(customer_data, "Feedback")
mkt_com      = normalize_column(customer_data, "MarketingCommunication")
```

**Step 3 — Merge all features into a single analytical DataFrame**

All normalized tables are joined back together via left merges on `CustomerID`:

```python
customer_main = demographic_features \
    .merge(purchase_hist, on='CustomerID', how='left') \
    .merge(sub_details,   on='CustomerID', how='left') \
    .merge(srv_interac,   on='CustomerID', how='left') \
    .merge(payment_hist,  on='CustomerID', how='left') \
    .merge(web_usage,     on='CustomerID', how='left') \
    .merge(clk_stream,    on='CustomerID', how='left') \
    .merge(eng_metrics,   on='CustomerID', how='left') \
    .merge(feedback,      on='CustomerID', how='left') \
    .merge(mkt_com,       on='CustomerID', how='left')

customer_main['ChurnLabel'] = customer_data['ChurnLabel']
```

### Feature Engineering

**RFM Analysis** (Recency, Frequency, Monetary) segments customers based on behavioral value:

```python
# Calculate raw RFM values
present_date = customer_main["last_interaction_date"].max()
customer_main['recency']   = present_date - customer_main['last_interaction_date']
customer_main['frequency'] = customer_main['total_frequency'] * customer_main['Logins']
customer_main['monetary']  = customer_main['total_purchase_value']

# Quintile binning (scored 1–5)
R_label = range(5, 0, -1)   # Lower recency = higher score
F_label = range(1, 6)       # Higher frequency = higher score
M_label = range(1, 6)       # Higher monetary = higher score

customer_main['recency']   = pd.qcut(customer_main['recency'],   5, labels=R_label)
customer_main['frequency'] = pd.qcut(customer_main['frequency'], 5, labels=F_label)
customer_main['monetary']  = pd.qcut(customer_main['monetary'],  5, labels=M_label)

# Composite RFM score and customer segments
RFM_Score = R + F + M
bins       = [1, 4, 10, 15]
bin_labels = ['risk', 'engaged', 'loyalist']
customer_main['customer_segment'] = pd.cut(RFM_Score, bins=bins, labels=bin_labels, include_lowest=True)
```

This produces three customer tiers — **risk** (low engagement, high recency), **engaged** (moderate activity), and **loyalist** (frequent, high-value, recent) — which serve as both a feature and a business-interpretable segmentation.

### Encoding Strategy

Different encoding approaches were applied based on the nature of each categorical variable:

| Variable Type                   | Method          | Columns                                          |
| ------------------------------- | --------------- | ------------------------------------------------ |
| **Nominal** (no order)          | OneHotEncoder   | `Gender`, `Segment`, `nps_category`              |
| **Ordinal** (natural order)     | LabelEncoder    | `age_group`, `customer_segment`                  |
| **Cardinal** (high cardinality) | Target Encoding | `Plan` (mean of `total_purchase_value` per plan) |

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Nominal — one-hot
ohc = OneHotEncoder(sparse_output=False)
for col in nominal_columns:
    encoded_col = ohc.fit_transform(customer_main[[col]])
    enc_df = pd.DataFrame(encoded_col, columns=ohc.get_feature_names_out([col]))
    customer_main = pd.concat([customer_main, enc_df], axis=1)
    customer_main.drop(columns=[col], inplace=True)

# Ordinal — label
le = LabelEncoder()
for col in ordinal_columns:
    customer_main[col] = le.fit_transform(customer_main[col])

# Cardinal — target encoding
for col in cardinal_columns:
    customer_main[col] = customer_main.groupby(col)['total_purchase_value'].transform('mean')
```

**Final output:** `churn_data_clean.csv` — **12,483 rows × 55 columns**, ready for model training.

---

## Model Development

All modeling work is documented in `model/model.ipynb`.

### Feature Selection

**Mutual information** was used to rank features by their statistical dependency with the target variable (`ChurnLabel`). This non-parametric method captures both linear and non-linear relationships:

```python
from sklearn.feature_selection import mutual_info_classif

target   = churn_data['ChurnLabel']
features = churn_data.drop(columns=['CustomerID', 'ChurnLabel'])

mi_scores = mutual_info_classif(features, target)
mi_scores_sorted = mi_scores_df.sort_values('Scores', ascending=False).iloc[:20]
```

**Top 10 features by mutual information score:**

| Rank | Feature                  | MI Score      |
| ---- | ------------------------ | ------------- |
| 1    | `late_payment_rate`      | Highest       |
| 2    | `payment_risk_score`     | Very High     |
| 3    | `total_late_payments`    | Very High     |
| 4    | `total_interactions`     | High          |
| 5    | `TimeSpent(minutes)`     | High          |
| 6    | `NPS`                    | Moderate–High |
| 7    | `engagement_intensity`   | Moderate      |
| 8    | `engagement_ratio`       | Moderate      |
| 9    | `nps_category_Detractor` | Moderate      |
| 10   | `nps_category_Passive`   | Moderate      |

**Key insight:** Payment behavior features dominate the top three positions, confirming that late payments and financial risk are the strongest individual predictors of churn. Engagement and NPS features provide complementary signal.

### Baseline Model

Before hyperparameter tuning, a Logistic Regression baseline was established:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
```

**Baseline results: 97.0% accuracy** — confirming that the selected features carry strong predictive signal and that the dataset is well-suited for classification.

**Class distribution check:** 6,314 churned vs. 6,169 retained — near-perfectly balanced, so no resampling or class weighting was necessary.

### Optimized Model

A **Decision Tree Classifier** was selected for the final model due to its interpretability (important for non-technical stakeholders) and strong performance on tabular data. Hyperparameters were optimized using **RandomizedSearchCV** with 4-fold cross-validation:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

model = DecisionTreeClassifier(random_state=42)

param_dist = {
    "criterion":        ["gini", "entropy", "log_loss"],
    "splitter":         ["best", "random"],
    "max_depth":        [100, 200, 300, 500],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":     ["auto", "sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    cv=4, n_jobs=1, random_state=42
)
random_search.fit(X_train, y_train)
```

**Best parameters found:**

| Parameter           | Value      |
| ------------------- | ---------- |
| `criterion`         | `log_loss` |
| `splitter`          | `random`   |
| `max_depth`         | `300`      |
| `min_samples_split` | `6`        |
| `min_samples_leaf`  | `4`        |
| `max_features`      | `log2`     |

### Evaluation Results

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 96.9% |
| **Precision** | ~97%  |
| **Recall**    | ~97%  |
| **F1-Score**  | ~97%  |

**Confusion matrix** (on 2,497 test samples):

|                      | Predicted: No Churn | Predicted: Churn |
| -------------------- | ------------------- | ---------------- |
| **Actual: No Churn** | 1,226 (TN)          | 38 (FP)          |
| **Actual: Churn**    | 36 (FN)             | 1,197 (TP)       |

The model correctly identifies 97% of actual churners (recall) while maintaining 97% precision — meaning very few false alarms. This balance is critical for retention programs, where both missed opportunities (false negatives) and wasted outreach (false positives) carry costs.

**Model persistence:**

```python
import pickle, json

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save expected feature columns for inference
with open("data.json", "w") as file:
    json.dump(X_train.columns.to_list(), file)
```

---

## API Service

The prediction model is served via a **FastAPI** REST endpoint (`app.py`), providing real-time inference:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd, pickle, json, os, uvicorn

app = FastAPI(title="Reder API", version="1.0.0")

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        example=[{
            "payment_risk_score": 400.0,
            "total_late_payments": 40,
            "late_payment_rate": 10.00,
            "NPS": 3,
            "engagement_ratio": 0.3,
            # ... all 20 features
        }]
    )

def load_model():
    model_path    = os.path.join('model', 'model.pkl')
    features_path = os.path.join('model', 'data.json')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'r') as f:
        features = json.load(f)
    return model, features

@app.post('/predict')
def model_predict(req: PredictionRequest):
    df = pd.DataFrame(req.records)
    model, features = load_model()
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)
    pred_proba = model.predict_proba(df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "prediction_probability": float(pred_proba[0])
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)
```

**How it works:**

1. A POST request sends customer attributes as a JSON array.
2. Pydantic validates the request schema against `PredictionRequest`.
3. The pre-trained model is loaded from `model.pkl`; expected features are loaded from `data.json`.
4. Input features are reindexed to match training columns (missing features default to 0).
5. The model returns both a binary prediction (0 or 1) and a continuous probability score (0.0–1.0).

**Example request:**

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"payment_risk_score": 400, "total_late_payments": 40, "late_payment_rate": 10.0, "NPS": 3, "engagement_ratio": 0.3, "total_interactions": 4, "TimeSpent(minutes)": 15, "engagement_intensity": 735}]}'
```

**Example response:**

```json
{
  "prediction": 1,
  "prediction_probability": 0.87
}
```

A prediction of `1` with probability `0.87` means the model is 87% confident this customer will churn — a strong candidate for proactive retention outreach.

---

## Presentation Layer

An interactive **23-slide React presentation** is included in `reder-presentation/`, built with Vite and styled with a dark navy/teal telecom aesthetic.

**Running the presentation locally:**

```bash
cd reder-presentation
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser. Navigate with arrow keys or the dot navigation bar.

**Slide coverage:** Company overview, business challenge, project rationale, objectives, data pipeline, model training, solution architecture, cloud deployment costs, expected impact, roadmap, and closing.

---

## Cloud Deployment Estimates

For production deployment of this lightweight FastAPI + Decision Tree solution (CPU-only inference, no GPU required), the following monthly cost estimates apply:

| Component             | AWS                   | Azure                      | GCP                              |
| --------------------- | --------------------- | -------------------------- | -------------------------------- |
| Compute               | EC2 t3.medium: $30–34 | App Service B1: $13–35     | Cloud Run (0.5 vCPU): $8–18      |
| Alternative Compute   | ECS Fargate: $18–25   | Container Instance: $15–22 | Compute Engine e2-medium: $25–30 |
| Storage               | S3: $1–3              | Blob Storage: $1–3         | Cloud Storage: $1–2              |
| Monitoring            | CloudWatch: $5–10     | Azure Monitor: $5–8        | Cloud Monitoring: $0–5           |
| Load Balancer         | ELB: $18–22           | App Gateway: $20–25        | Cloud LB: $18–20                 |
| Data Transfer (10 GB) | $1–2                  | $1–2                       | $1–2                             |
| **Estimated Total**   | **$55–95/mo**         | **$45–90/mo**              | **$30–75/mo**                    |

**Provider comparison:**

|                | AWS                                        | Azure                                             | GCP                                                         |
| -------------- | ------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| **Best for**   | Broadest ecosystem, mature ML tooling      | Microsoft/enterprise integration, hybrid benefit  | Container-native workloads, pay-per-request                 |
| **Strengths**  | Largest community, most regions, SageMaker | Hybrid Benefit saves up to 40%, strong compliance | Cloud Run scales to zero, automatic sustained-use discounts |
| **Trade-offs** | Complex pricing, egress fees compound      | SKU complexity, less intuitive console            | Smaller enterprise footprint, fewer regions                 |

**Recommendation:** For this POC, **GCP Cloud Run** offers the lowest cost with pay-per-request scaling. For enterprise environments with existing Microsoft infrastructure, **Azure App Service** integrates naturally. For maximum flexibility and ecosystem breadth, **AWS ECS Fargate** provides the best balance.

---

## Roadmap

| Phase               | Timeline   | Focus                                | Key Activities                                                                                                                          |
| ------------------- | ---------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **1 — Validation**  | Month 1–2  | Prove the model works with live data | Validate predictions against real outcomes, A/B test retention interventions, refine feature engineering, obtain stakeholder sign-off   |
| **2 — Integration** | Month 3–5  | Connect into existing systems        | Integrate API with CRM, build monitoring layer, automate data ingestion pipelines, establish model retraining schedule                  |
| **3 — Scale**       | Month 6–12 | Full production and expansion        | Deploy to production, enable real-time scoring at scale, expand to upsell/cross-sell predictions, measure KPI outcomes against baseline |

**Long-term vision:** Evolve from a single-purpose churn predictor into a comprehensive **customer intelligence platform** — combining churn prediction, upsell scoring, and lifetime value optimization into a unified decision engine.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the presentation)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd reder-analytics

# Set up Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API
python app.py
# → API available at http://localhost:3000
# → Interactive docs at http://localhost:3000/docs

# (Optional) Run the presentation
cd reder-presentation
npm install
npm run dev
# → Presentation at http://localhost:5173
```

### Reproducing the Pipeline

1. **Data cleaning:** Open `preprocessing/clean.ipynb` in Jupyter and run all cells. This reads the raw CSV and produces `model/churn_data_clean.csv`.

2. **Model training:** Open `model/model.ipynb` and run all cells. This trains the classifier and outputs `model/model.pkl` and `model/data.json`.

3. **Inference:** Start the API with `python app.py` and send POST requests to `/predict`.

---

## Tech Stack

| Layer                   | Technology                                 | Purpose                                       |
| ----------------------- | ------------------------------------------ | --------------------------------------------- |
| **Data Processing**     | Python, Pandas, NumPy                      | Data ingestion, cleaning, transformation      |
| **Notebooks**           | Jupyter                                    | Interactive development and documentation     |
| **Machine Learning**    | Scikit-learn                               | Feature selection, model training, evaluation |
| **Feature Selection**   | Mutual Information (`mutual_info_classif`) | Non-parametric feature ranking                |
| **Model Serialization** | Pickle, JSON                               | Model and feature persistence                 |
| **API Framework**       | FastAPI                                    | REST endpoint with automatic OpenAPI docs     |
| **Schema Validation**   | Pydantic                                   | Request/response type safety                  |
| **ASGI Server**         | Uvicorn                                    | High-performance async serving                |
| **Presentation**        | React, Vite, Lucide Icons                  | Interactive slide deck                        |

---

## License

This project was developed as a proof-of-concept for Reder Telecommunications. Names were changed. All rights reserved.

---
