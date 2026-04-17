"""
SQL analytics engine for RetainSight.

Every function runs a real SQL query against the database and returns
a pandas DataFrame. These cover the core business analytics that
interviewers expect: cohorts, revenue, funnels, LTV, segments.
"""

import sqlite3

import pandas as pd

from src.data_generation.schema import get_connection


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Revenue Analytics
# ---------------------------------------------------------------------------

def monthly_revenue() -> pd.DataFrame:
    """Monthly revenue with MoM growth rate."""
    return _query("""
        WITH monthly AS (
            SELECT
                strftime('%Y-%m', transaction_date) AS month,
                SUM(amount)                         AS revenue,
                COUNT(DISTINCT customer_id)         AS paying_customers
            FROM transactions
            GROUP BY 1
        )
        SELECT
            month,
            revenue,
            paying_customers,
            ROUND(revenue / NULLIF(paying_customers, 0), 2) AS avg_revenue_per_customer,
            ROUND(
                (revenue - LAG(revenue) OVER (ORDER BY month))
                / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100,
            2) AS mom_growth_pct
        FROM monthly
        ORDER BY month
    """)


def revenue_by_plan() -> pd.DataFrame:
    """Total and average MRR by subscription plan."""
    return _query("""
        SELECT
            s.plan,
            COUNT(DISTINCT s.customer_id)   AS customers,
            ROUND(SUM(s.mrr), 2)            AS total_mrr,
            ROUND(AVG(s.mrr), 2)            AS avg_mrr
        FROM subscriptions s
        WHERE s.status = 'active'
        GROUP BY s.plan
        ORDER BY total_mrr DESC
    """)


def revenue_by_category() -> pd.DataFrame:
    """Revenue breakdown by product category."""
    return _query("""
        SELECT
            product_category,
            COUNT(*)                AS transaction_count,
            ROUND(SUM(amount), 2)  AS total_revenue,
            ROUND(AVG(amount), 2)  AS avg_transaction
        FROM transactions
        GROUP BY product_category
        ORDER BY total_revenue DESC
    """)


# ---------------------------------------------------------------------------
# Cohort & Retention
# ---------------------------------------------------------------------------

def cohort_retention() -> pd.DataFrame:
    """
    Monthly signup cohort retention matrix.
    Returns columns: cohort, period (months since signup), retained_customers, cohort_size, retention_rate.
    """
    return _query("""
        WITH cohort AS (
            SELECT
                customer_id,
                strftime('%Y-%m', signup_date) AS cohort
            FROM customers
        ),
        activity AS (
            SELECT DISTINCT
                customer_id,
                strftime('%Y-%m', event_date) AS active_month
            FROM events
        ),
        joined AS (
            SELECT
                c.cohort,
                a.active_month,
                c.customer_id,
                CAST(
                    (CAST(strftime('%Y', a.active_month || '-01') AS INTEGER) -
                     CAST(strftime('%Y', c.cohort || '-01') AS INTEGER)) * 12
                    + (CAST(strftime('%m', a.active_month || '-01') AS INTEGER) -
                       CAST(strftime('%m', c.cohort || '-01') AS INTEGER))
                AS INTEGER) AS period
            FROM cohort c
            JOIN activity a ON c.customer_id = a.customer_id
        ),
        cohort_size AS (
            SELECT cohort, COUNT(DISTINCT customer_id) AS cohort_size
            FROM cohort
            GROUP BY cohort
        )
        SELECT
            j.cohort,
            j.period,
            COUNT(DISTINCT j.customer_id) AS retained_customers,
            cs.cohort_size,
            ROUND(COUNT(DISTINCT j.customer_id) * 100.0 / cs.cohort_size, 2) AS retention_rate
        FROM joined j
        JOIN cohort_size cs ON j.cohort = cs.cohort
        WHERE j.period >= 0 AND j.period <= 12
        GROUP BY j.cohort, j.period
        ORDER BY j.cohort, j.period
    """)


# ---------------------------------------------------------------------------
# Customer Analytics
# ---------------------------------------------------------------------------

def top_customers(limit: int = 20) -> pd.DataFrame:
    """Top customers by total lifetime spend."""
    return _query("""
        SELECT
            c.customer_id,
            c.name,
            c.country,
            c.signup_date,
            c.acquisition_channel,
            ROUND(SUM(t.amount), 2) AS total_spend,
            COUNT(t.transaction_id) AS num_transactions,
            ROUND(SUM(t.amount) / NULLIF(COUNT(t.transaction_id), 0), 2) AS avg_order_value,
            CASE WHEN c.is_churned = 1 THEN 'Churned' ELSE 'Active' END AS status
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id
        ORDER BY total_spend DESC
        LIMIT ?
    """, (limit,))


def customer_lifetime_value() -> pd.DataFrame:
    """
    Estimated LTV per customer:
    LTV = avg_monthly_revenue * avg_lifespan_months
    """
    return _query("""
        WITH monthly_spend AS (
            SELECT
                customer_id,
                strftime('%Y-%m', transaction_date) AS month,
                SUM(amount) AS monthly_revenue
            FROM transactions
            GROUP BY customer_id, month
        ),
        customer_stats AS (
            SELECT
                customer_id,
                AVG(monthly_revenue)   AS avg_monthly_revenue,
                COUNT(DISTINCT month)  AS active_months
            FROM monthly_spend
            GROUP BY customer_id
        )
        SELECT
            c.customer_id,
            c.name,
            c.signup_date,
            c.acquisition_channel,
            ROUND(cs.avg_monthly_revenue, 2)                        AS avg_monthly_revenue,
            cs.active_months,
            ROUND(cs.avg_monthly_revenue * cs.active_months, 2)     AS estimated_ltv,
            CASE WHEN c.is_churned = 1 THEN 'Churned' ELSE 'Active' END AS status
        FROM customers c
        JOIN customer_stats cs ON c.customer_id = cs.customer_id
        ORDER BY estimated_ltv DESC
    """)


def customer_segments() -> pd.DataFrame:
    """Segment customers by acquisition channel with key metrics."""
    return _query("""
        SELECT
            c.acquisition_channel                       AS segment,
            COUNT(DISTINCT c.customer_id)               AS total_customers,
            SUM(c.is_churned)                           AS churned,
            ROUND(SUM(c.is_churned) * 100.0 /
                  COUNT(DISTINCT c.customer_id), 2)     AS churn_rate_pct,
            ROUND(AVG(t.total_spend), 2)                AS avg_spend
        FROM customers c
        LEFT JOIN (
            SELECT customer_id, SUM(amount) AS total_spend
            FROM transactions
            GROUP BY customer_id
        ) t ON c.customer_id = t.customer_id
        GROUP BY c.acquisition_channel
        ORDER BY avg_spend DESC
    """)


# ---------------------------------------------------------------------------
# Conversion Funnel
# ---------------------------------------------------------------------------

def conversion_funnel() -> pd.DataFrame:
    """
    Behavioral funnel:
    signup → login → page_view → feature_use → transaction
    Shows drop-off at each stage.
    """
    return _query("""
        WITH stages AS (
            SELECT 1 AS stage_order, 'Signed Up' AS stage,
                   COUNT(DISTINCT customer_id) AS users
            FROM customers

            UNION ALL
            SELECT 2, 'Logged In',
                   COUNT(DISTINCT customer_id)
            FROM events WHERE event_type = 'login'

            UNION ALL
            SELECT 3, 'Viewed Pages',
                   COUNT(DISTINCT customer_id)
            FROM events WHERE event_type = 'page_view'

            UNION ALL
            SELECT 4, 'Used Features',
                   COUNT(DISTINCT customer_id)
            FROM events WHERE event_type = 'feature_use'

            UNION ALL
            SELECT 5, 'Made Purchase',
                   COUNT(DISTINCT customer_id)
            FROM transactions
        )
        SELECT
            stage_order,
            stage,
            users,
            ROUND(users * 100.0 /
                  FIRST_VALUE(users) OVER (ORDER BY stage_order), 2) AS pct_of_signups
        FROM stages
        ORDER BY stage_order
    """)


# ---------------------------------------------------------------------------
# Engagement & Activity
# ---------------------------------------------------------------------------

def daily_active_users() -> pd.DataFrame:
    """Daily active users (DAU) based on events."""
    return _query("""
        SELECT
            event_date          AS date,
            COUNT(DISTINCT customer_id) AS dau
        FROM events
        GROUP BY event_date
        ORDER BY event_date
    """)


def engagement_by_churn_status() -> pd.DataFrame:
    """Compare average engagement between churned and active customers."""
    return _query("""
        SELECT
            CASE WHEN c.is_churned = 1 THEN 'Churned' ELSE 'Active' END AS status,
            ROUND(AVG(e.event_count), 2)  AS avg_events,
            ROUND(AVG(e.login_count), 2)  AS avg_logins,
            ROUND(AVG(e.feature_count), 2) AS avg_feature_uses,
            ROUND(AVG(e.ticket_count), 2)  AS avg_support_tickets
        FROM customers c
        LEFT JOIN (
            SELECT
                customer_id,
                COUNT(*)                                             AS event_count,
                SUM(CASE WHEN event_type = 'login' THEN 1 ELSE 0 END)         AS login_count,
                SUM(CASE WHEN event_type = 'feature_use' THEN 1 ELSE 0 END)   AS feature_count,
                SUM(CASE WHEN event_type = 'support_ticket' THEN 1 ELSE 0 END) AS ticket_count
            FROM events
            GROUP BY customer_id
        ) e ON c.customer_id = e.customer_id
        GROUP BY c.is_churned
    """)


def churn_by_segment() -> pd.DataFrame:
    """Churn rate breakdown by country, channel, and plan."""
    return _query("""
        SELECT
            c.country,
            c.acquisition_channel,
            s.plan,
            COUNT(DISTINCT c.customer_id) AS total,
            SUM(c.is_churned) AS churned,
            ROUND(SUM(c.is_churned) * 100.0 / COUNT(DISTINCT c.customer_id), 2) AS churn_rate
        FROM customers c
        LEFT JOIN subscriptions s ON c.customer_id = s.customer_id
            AND s.status IN ('active', 'cancelled')
        GROUP BY c.country, c.acquisition_channel, s.plan
        HAVING total >= 5
        ORDER BY churn_rate DESC
    """)
