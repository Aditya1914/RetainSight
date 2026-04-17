"""
RetainSight — Streamlit Dashboard

Pages:
  1. Executive Overview — KPIs, revenue trend, funnel
  2. Cohort & Retention — heatmap, engagement comparison
  3. Churn Risk Explorer — risk distribution, customer-level drill-down
  4. Decision Panel — recommendations with ROI
  5. Customer Deep Dive — search and inspect individual customers
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="RetainSight",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.title("🔭 RetainSight")
st.sidebar.markdown("*Predictive Business Analytics*")

page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Cohort & Retention", "Churn Risk Explorer",
     "Decisions & ROI", "Customer Deep Dive"],
)

st.sidebar.markdown("---")
st.sidebar.caption("RetainSight v1.0")


# ── Data Loading (cached) ───────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_monthly_revenue():
    from src.analytics.queries import monthly_revenue
    return monthly_revenue()

@st.cache_data(ttl=300)
def load_revenue_by_plan():
    from src.analytics.queries import revenue_by_plan
    return revenue_by_plan()

@st.cache_data(ttl=300)
def load_revenue_by_category():
    from src.analytics.queries import revenue_by_category
    return revenue_by_category()

@st.cache_data(ttl=300)
def load_cohort_retention():
    from src.analytics.queries import cohort_retention
    return cohort_retention()

@st.cache_data(ttl=300)
def load_conversion_funnel():
    from src.analytics.queries import conversion_funnel
    return conversion_funnel()

@st.cache_data(ttl=300)
def load_dau():
    from src.analytics.queries import daily_active_users
    return daily_active_users()

@st.cache_data(ttl=300)
def load_engagement_comparison():
    from src.analytics.queries import engagement_by_churn_status
    return engagement_by_churn_status()


@st.cache_data(ttl=300)
def load_customer_segments():
    from src.analytics.queries import customer_segments
    return customer_segments()

@st.cache_data(ttl=300)
def load_churn_predictions():
    from src.ml.models import predict_churn
    return predict_churn()

@st.cache_data(ttl=300)
def load_recommendations():
    from src.decision_engine.engine import generate_recommendations
    return generate_recommendations()

@st.cache_data(ttl=300)
def load_summary_report():
    from src.decision_engine.engine import summary_report
    return summary_report()


# ── Helpers ──────────────────────────────────────────────────────────────

def metric_card(label: str, value, delta=None, delta_color="normal"):
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# =====================================================================
# PAGE 1: Executive Overview
# =====================================================================

if page == "Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level business health at a glance.")

    rev = load_monthly_revenue()
    funnel = load_conversion_funnel()
    segments = load_customer_segments()
    rev_plan = load_revenue_by_plan()

    # KPI row
    total_revenue = rev["revenue"].sum()
    latest_month_rev = rev.iloc[-1]["revenue"] if len(rev) else 0
    latest_growth = rev.iloc[-1]["mom_growth_pct"] if len(rev) else 0
    total_customers = segments["total_customers"].sum()
    avg_churn = segments["churn_rate_pct"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Total Revenue", f"${total_revenue:,.0f}")
    with k2:
        metric_card("Latest Month", f"${latest_month_rev:,.0f}",
                     delta=f"{latest_growth:+.1f}% MoM")
    with k3:
        metric_card("Total Customers", f"{total_customers:,}")
    with k4:
        metric_card("Avg Churn Rate", f"{avg_churn:.1f}%",
                     delta_color="inverse")

    st.markdown("---")

    # Revenue trend
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Monthly Revenue Trend")
        fig = px.area(rev, x="month", y="revenue",
                      labels={"month": "Month", "revenue": "Revenue ($)"},
                      color_discrete_sequence=["#4F46E5"])
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Revenue by Plan")
        fig = px.pie(rev_plan, names="plan", values="total_mrr",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    # Funnel + segments
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Conversion Funnel")
        fig = go.Figure(go.Funnel(
            y=funnel["stage"],
            x=funnel["users"],
            textinfo="value+percent initial",
            marker=dict(color=["#4F46E5", "#7C3AED", "#A78BFA", "#C4B5FD", "#DDD6FE"]),
        ))
        fig.update_layout(funnelmode="stack")
        st.plotly_chart(fig, width="stretch")

    with col4:
        st.subheader("Churn Rate by Acquisition Channel")
        fig = px.bar(segments, x="segment", y="churn_rate_pct",
                     color="segment",
                     labels={"segment": "Channel", "churn_rate_pct": "Churn Rate (%)"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")


# =====================================================================
# PAGE 2: Cohort & Retention
# =====================================================================

elif page == "Cohort & Retention":
    st.title("📈 Cohort & Retention Analysis")

    cohort_df = load_cohort_retention()
    engagement = load_engagement_comparison()
    dau = load_dau()

    # Cohort heatmap
    st.subheader("Monthly Cohort Retention Heatmap")
    pivot = cohort_df.pivot_table(
        index="cohort", columns="period", values="retention_rate",
    )
    pivot = pivot[[c for c in sorted(pivot.columns) if c <= 12]]

    fig = px.imshow(
        pivot,
        labels=dict(x="Months Since Signup", y="Signup Cohort", color="Retention %"),
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=".0f",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Active Users")
        fig = px.line(dau, x="date", y="dau",
                      labels={"date": "Date", "dau": "DAU"},
                      color_discrete_sequence=["#4F46E5"])
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Engagement: Active vs Churned")
        if len(engagement):
            eng_melted = engagement.melt(
                id_vars="status",
                value_vars=["avg_logins", "avg_feature_uses", "avg_support_tickets"],
                var_name="metric",
                value_name="average",
            )
            fig = px.bar(eng_melted, x="metric", y="average", color="status",
                         barmode="group",
                         color_discrete_map={"Active": "#4F46E5", "Churned": "#EF4444"},
                         labels={"metric": "Metric", "average": "Avg Count"})
            st.plotly_chart(fig, width="stretch")


# =====================================================================
# PAGE 3: Churn Risk Explorer
# =====================================================================

elif page == "Churn Risk Explorer":
    st.title("⚠️ Churn Risk Explorer")

    churn_df = load_churn_predictions()

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with col_f2:
        prob_threshold = st.slider(
            "Min Churn Probability", 0.0, 1.0, 0.0, 0.05,
        )

    filtered = churn_df[
        (churn_df["risk_tier"].isin(risk_filter))
        & (churn_df["churn_probability"] >= prob_threshold)
    ]

    # KPIs
    k1, k2, k3 = st.columns(3)
    tier_counts = churn_df["risk_tier"].value_counts()
    with k1:
        metric_card("🔴 High Risk", tier_counts.get("High", 0))
    with k2:
        metric_card("🟡 Medium Risk", tier_counts.get("Medium", 0))
    with k3:
        metric_card("🟢 Low Risk", tier_counts.get("Low", 0))

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Distribution")
        fig = px.histogram(
            churn_df, x="churn_probability", nbins=40,
            color="risk_tier",
            color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
            labels={"churn_probability": "Churn Probability", "count": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Spend vs Churn Probability")
        fig = px.scatter(
            filtered, x="total_spend", y="churn_probability",
            color="risk_tier",
            color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
            hover_data=["customer_id", "current_mrr"],
            labels={"total_spend": "Total Spend ($)", "churn_probability": "Churn Prob"},
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"Customer List ({len(filtered):,} customers)")
    st.dataframe(
        filtered.sort_values("churn_probability", ascending=False),
        width="stretch",
        height=400,
    )


# =====================================================================
# PAGE 4: Decisions & ROI
# =====================================================================

elif page == "Decisions & ROI":
    st.title("🎯 Decisions & ROI")
    st.markdown("Actionable recommendations backed by expected return on investment.")

    report = load_summary_report()
    recs = load_recommendations()

    # Executive summary cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Revenue at Risk", f"${report['total_revenue_at_risk']:,.0f}")
    with k2:
        metric_card("Intervention Cost", f"${report['total_intervention_cost']:,.0f}")
    with k3:
        metric_card("Expected Savings", f"${report['total_expected_savings']:,.0f}")
    with k4:
        metric_card("Net ROI", f"{report['net_roi_pct']:.0f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Tier Breakdown")
        tier_data = pd.DataFrame({
            "Tier": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [report["high_risk_count"], report["medium_risk_count"], report["low_risk_count"]],
        })
        fig = px.bar(tier_data, x="Tier", y="Count",
                     color="Tier",
                     color_discrete_map={
                         "High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#10B981"
                     })
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("ROI by Customer (Top 50)")
        top_roi = recs[recs["expected_roi"] > 0].nlargest(50, "expected_roi")
        fig = px.bar(top_roi, x="customer_id", y="expected_roi",
                     color="risk_tier",
                     color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B"},
                     labels={"customer_id": "Customer", "expected_roi": "Expected ROI (x)"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Recommendation table
    st.subheader("Full Recommendation Table")
    risk_filter = st.multiselect(
        "Filter by Risk",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"],
        key="rec_risk_filter",
    )
    display_recs = recs[recs["risk_tier"].isin(risk_filter)]
    display_cols = [
        "customer_id", "name", "country", "risk_tier", "churn_probability",
        "monthly_value", "annual_risk_value", "recommended_action",
        "intervention_cost", "expected_save_value", "expected_roi",
    ]
    st.dataframe(
        display_recs[display_cols].sort_values("annual_risk_value", ascending=False),
        width="stretch",
        height=400,
    )


# =====================================================================
# PAGE 5: Customer Deep Dive
# =====================================================================

elif page == "Customer Deep Dive":
    st.title("🔍 Customer Deep Dive")

    recs = load_recommendations()

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        tier_filter = st.selectbox(
            "Filter by Risk Tier",
            options=["All", "High", "Medium", "Low"],
            key="dive_tier_filter",
        )
    filtered_recs = recs if tier_filter == "All" else recs[recs["risk_tier"] == tier_filter]

    customer_options = filtered_recs.sort_values("churn_probability", ascending=False)
    display_labels = {
        row["customer_id"]: f"{row['customer_id']} — {row['name']} ({row['risk_tier']} risk)"
        for _, row in customer_options.iterrows()
    }

    with col_f2:
        selected_id = st.selectbox(
            "Select a Customer",
            options=list(display_labels.keys()),
            format_func=lambda x: display_labels[x],
        )

    if selected_id:
        rec = recs[recs["customer_id"] == selected_id].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Name:** {rec['name']}")
            st.markdown(f"**Country:** {rec['country']}")
            st.markdown(f"**Signed up:** {rec['signup_date']}")
        with col2:
            st.markdown(f"**Channel:** {rec['acquisition_channel']}")
            st.markdown(f"**Total Spend:** ${rec['total_spend']:,.2f}")
            st.markdown(f"**Monthly Value:** ${rec['monthly_value']:,.2f}")
        with col3:
            tier_str = str(rec["risk_tier"])
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(tier_str, "⚪")
            st.markdown(f"**Status:** {color} {tier_str} Risk")
            st.markdown(f"**Churn Prob:** {rec['churn_probability']:.1%}")
            st.markdown(f"**MRR:** ${rec['current_mrr']:,.2f}")

        st.markdown("---")
        st.subheader("Risk Assessment")

        k1, k2, k3 = st.columns(3)
        with k1:
            tier_str = str(rec["risk_tier"])
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(tier_str, "⚪")
            metric_card(f"{color} Risk Tier", tier_str)
        with k2:
            metric_card("Churn Probability", f"{rec['churn_probability']:.1%}")
        with k3:
            metric_card("Annual Revenue at Risk", f"${rec['annual_risk_value']:,.0f}")

        st.markdown("---")
        st.subheader("Recommendation")
        action_color = {"High": "error", "Medium": "warning", "Low": "info"}.get(tier_str, "info")
        getattr(st, action_color)(f"**Action:** {rec['recommended_action']}")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            metric_card("Intervention Cost", f"${rec['intervention_cost']:,.0f}")
        with rcol2:
            metric_card("Expected Savings", f"${rec['expected_save_value']:,.0f}")
        with rcol3:
            roi_val = rec['expected_roi']
            metric_card("Expected ROI", f"{roi_val:.1f}x")
