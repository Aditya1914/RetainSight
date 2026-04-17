# RetainSight

**A predictive analytics platform that analyzes customer behavior, identifies churn risk, and recommends data-driven retention strategies with dollar-value ROI estimates.**

RetainSight doesn't stop at "here's a prediction." It goes all the way to *"here's what you should do about it, what it'll cost, and what it's worth."*

---

## The Problem

A SaaS company is losing customers. Leadership wants to know: **who's likely to leave, what should we do about it, and is it worth the cost?**

RetainSight answers all three. It connects customer data to business decisions — running analytics to understand what's happening, ML models to predict what's coming, and a decision engine to recommend what to do about it with dollar-value ROI estimates.

---

## What It Does

**Data layer** — A synthetic SaaS company: 2,000 customers, subscription histories, purchase transactions, and 690K+ behavioral events. Churned customers show realistically declining engagement before they leave.

**Analytics layer** — 10 SQL queries that answer the questions a data team actually gets asked: monthly revenue with MoM growth, cohort retention matrices, conversion funnels, LTV estimation, and churn rate breakdowns across segments.

**ML layer** — Three models trained on 18+ engineered features (tenure, spend, engagement, recency):
- Churn prediction (Logistic Regression & Random Forest, AUC ~0.97)
- Customer segmentation (KMeans, 4 behavioral clusters)
- Lifetime value estimation (Gradient Boosting, R² ~0.80)

**Decision layer** — Takes model predictions and produces per-customer recommendations: risk tier, recommended intervention, cost, expected savings, and ROI. Aggregates into an executive summary of total revenue at risk vs. intervention budget vs. net return.

**Dashboard** — 5-page Streamlit app:

| Page | What It Shows |
|------|--------------|
| **Executive Overview** | KPIs, revenue trend, conversion funnel, churn by channel |
| **Cohort & Retention** | Retention heatmap, DAU trend, active vs churned engagement |
| **Churn Risk Explorer** | Risk distribution, spend vs churn scatter, filterable customer table |
| **Decisions & ROI** | Revenue at risk, intervention costs, expected savings, recommendation table |
| **Customer Deep Dive** | Individual customer profile with risk assessment and action plan |

**EDA notebook** — Exploratory analysis (`notebooks/01_eda_and_churn_analysis.ipynb`) covering data distributions, churn investigation, and feature correlations that informed the model design.

---

## Quick Start

```bash
# Clone and enter the project
cd RetainSight

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate data + train all models
python setup.py

# Launch the dashboard
streamlit run app/dashboard.py
```

---

## Project Structure

```
RetainSight/
├── app/
│   └── dashboard.py               # 5-page Streamlit dashboard
├── src/
│   ├── data_generation/
│   │   ├── schema.py              # SQLite schema (4 tables, indexes)
│   │   └── generate.py            # Synthetic data generator (2,000 customers)
│   ├── analytics/
│   │   └── queries.py             # 10 SQL analytics functions
│   ├── ml/
│   │   ├── feature_engineering.py # 18+ feature customer matrix
│   │   └── models.py             # Churn, segmentation, LTV models
│   └── decision_engine/
│       └── engine.py              # Recommendations + ROI calculations
├── notebooks/
│   └── 01_eda_and_churn_analysis.ipynb  # Exploratory analysis
├── data/                          # Generated DB + trained models (gitignored)
├── setup.py                       # One-command setup
└── requirements.txt
```

---

## Architecture

```
Raw Tables (SQLite)
    │
    ├──▶ SQL Analytics (queries.py)         → Revenue, cohorts, funnels, segments
    │
    ├──▶ Feature Engineering                → 18+ features per customer
    │         │
    │         ▼
    │    ML Models (models.py)              → Churn risk, segments, LTV
    │         │
    │         ▼
    │    Decision Engine (engine.py)        → Actions, costs, expected ROI
    │
    └──▶ Streamlit Dashboard (dashboard.py) → 5 interactive pages
```

Every layer is callable as standalone Python functions — the dashboard is just one consumer. The same functions could power an API, a scheduled report, or an AI orchestration layer.

---

## Data Schema

| Table | Records | Description |
|-------|---------|-------------|
| `customers` | 2,000 | Demographics, signup date, acquisition channel, churn status |
| `subscriptions` | ~2,800 | Plan history (free/starter/pro/enterprise) with MRR, upgrades, downgrades |
| `transactions` | ~13,400 | Purchases across 4 categories (subscription, addon, service, one-time) |
| `events` | ~691,000 | Behavioral signals: logins, page views, feature usage, support tickets, feedback |

Churned customers are programmatically designed to show declining event frequency in the second half of their lifecycle — mimicking real-world disengagement patterns.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | SQLite |
| Analytics | SQL + Pandas |
| ML | scikit-learn |
| Visualization | Plotly + Streamlit |
| Decision Engine | Python |

---

## Adapting to Real Data

RetainSight is built on synthetic data, but the architecture is schema-driven. To point it at a real SaaS company's data:

1. Replace `data/retainsight.db` with a database matching the 4-table schema (customers, subscriptions, transactions, events)
2. Re-run `python setup.py` to retrain models on the new data
3. The dashboard, analytics, and decision engine work without code changes

The same function layer (`queries.py`, `models.py`, `engine.py`) could also back an API, a scheduled report, or a natural language interface.

---

## License

MIT
