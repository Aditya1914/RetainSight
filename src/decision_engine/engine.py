"""
RetainSight Decision Engine.

Translates ML predictions + analytics into actionable business
recommendations with ROI estimates.

Core logic:
  1. Score every customer for churn risk
  2. Estimate cost-of-churn per customer (based on MRR/LTV)
  3. Recommend an action (discount, outreach, upsell, nothing)
  4. Estimate ROI of that action
"""

import pandas as pd
import numpy as np

from src.ml.models import predict_churn
from src.data_generation.schema import get_connection


# Intervention costs and expected success rates
INTERVENTIONS = {
    "High": {
        "action": "Personal outreach + 20% discount",
        "cost_per_customer": 50.0,
        "success_rate": 0.35,
    },
    "Medium": {
        "action": "Automated email + 10% discount",
        "cost_per_customer": 15.0,
        "success_rate": 0.25,
    },
    "Low": {
        "action": "No intervention needed",
        "cost_per_customer": 0.0,
        "success_rate": 0.0,
    },
}


def _get_customer_context() -> pd.DataFrame:
    """Fetch customer name, country, channel for display purposes."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT customer_id, name, country, acquisition_channel, signup_date
        FROM customers
    """, conn)
    conn.close()
    return df


def generate_recommendations() -> pd.DataFrame:
    """
    Produce a recommendation for every customer.

    Columns:
      - customer_id, name, country, risk_tier, churn_probability
      - monthly_value (MRR proxy)
      - annual_risk_value: revenue at risk if customer churns
      - recommended_action
      - intervention_cost
      - expected_save_value: probability-weighted revenue saved
      - expected_roi: (save - cost) / cost
    """
    churn_df = predict_churn()
    context_df = _get_customer_context()

    df = churn_df.merge(context_df, on="customer_id", how="left")

    df["monthly_value"] = np.where(
        df["current_mrr"] > 0,
        df["current_mrr"],
        df["total_spend"] / np.maximum(df["tenure_days"] / 30, 1),
    )
    df["annual_risk_value"] = (
        df["monthly_value"] * 12 * df["churn_probability"]
    ).round(2)

    actions, costs, success_rates = [], [], []
    for tier in df["risk_tier"]:
        tier_str = str(tier)
        intervention = INTERVENTIONS.get(tier_str, INTERVENTIONS["Low"])
        actions.append(intervention["action"])
        costs.append(intervention["cost_per_customer"])
        success_rates.append(intervention["success_rate"])

    df["recommended_action"] = actions
    df["intervention_cost"] = costs
    df["expected_save_value"] = (
        df["annual_risk_value"] * pd.Series(success_rates, index=df.index)
    ).round(2)
    df["expected_roi"] = np.where(
        df["intervention_cost"] > 0,
        ((df["expected_save_value"] - df["intervention_cost"])
         / df["intervention_cost"]).round(2),
        0.0,
    )

    return df.sort_values("annual_risk_value", ascending=False)


def summary_report() -> dict:
    """
    Executive summary: aggregate ROI across all recommended interventions.
    """
    df = generate_recommendations()
    actionable = df[df["risk_tier"].isin(["High", "Medium"])]

    total_customers_at_risk = len(actionable)
    total_revenue_at_risk = actionable["annual_risk_value"].sum()
    total_intervention_cost = actionable["intervention_cost"].sum()
    total_expected_savings = actionable["expected_save_value"].sum()
    net_roi = (
        (total_expected_savings - total_intervention_cost)
        / max(total_intervention_cost, 1)
    )

    high_risk = df[df["risk_tier"] == "High"]
    medium_risk = df[df["risk_tier"] == "Medium"]

    return {
        "total_customers": len(df),
        "high_risk_count": len(high_risk),
        "medium_risk_count": len(medium_risk),
        "low_risk_count": len(df[df["risk_tier"] == "Low"]),
        "total_revenue_at_risk": round(total_revenue_at_risk, 2),
        "total_intervention_cost": round(total_intervention_cost, 2),
        "total_expected_savings": round(total_expected_savings, 2),
        "net_roi_pct": round(net_roi * 100, 2),
        "high_risk_avg_value": round(high_risk["annual_risk_value"].mean(), 2) if len(high_risk) else 0,
        "medium_risk_avg_value": round(medium_risk["annual_risk_value"].mean(), 2) if len(medium_risk) else 0,
    }
