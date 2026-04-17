"""
Feature engineering for ML models.

Builds a feature matrix from raw SQL data — each row is one customer
with aggregated behavioral, transactional, and subscription features.

For churned customers, features are computed as of a random point BEFORE
churn (simulating prediction on still-active customers). This prevents
temporal leakage from post-churn inactivity.
"""

import random

import pandas as pd

from src.data_generation.schema import get_connection


def build_feature_matrix() -> pd.DataFrame:
    """
    Build a customer-level feature matrix for churn prediction and segmentation.

    Features:
      - tenure_days: days since signup (as of observation point)
      - total_spend, avg_order_value, num_transactions
      - plan_encoded: numeric plan tier
      - current_mrr
      - total_events, login_count, feature_use_count, support_ticket_count
      - events_last_30d, events_last_7d: recency signals
      - days_since_last_event: inactivity indicator
      - avg_events_per_week: engagement intensity normalized by tenure
      - support_ticket_ratio: support tickets as fraction of total events
      - acquisition_channel_encoded, country_encoded
      - age
      - is_churned (target)
    """
    conn = get_connection()

    customers = pd.read_sql_query(
        "SELECT customer_id, signup_date, churn_date, is_churned FROM customers",
        conn,
    )

    sim_end = "2025-03-31"
    random.seed(123)
    obs_dates = {}
    for _, row in customers.iterrows():
        cid = row["customer_id"]
        if row["is_churned"] and row["churn_date"]:
            churn_dt = pd.Timestamp(row["churn_date"])
            signup_dt = pd.Timestamp(row["signup_date"])
            window = (churn_dt - signup_dt).days
            if window > 30:
                offset = random.randint(14, min(60, window // 2))
            else:
                offset = max(1, window // 3)
            obs_dt = churn_dt - pd.Timedelta(days=offset)
            obs_dates[cid] = obs_dt.strftime("%Y-%m-%d")
        else:
            obs_dates[cid] = sim_end

    obs_df = pd.DataFrame(
        list(obs_dates.items()), columns=["customer_id", "obs_date"]
    )

    obs_table_ddl = """
        CREATE TEMP TABLE obs_points (
            customer_id INTEGER PRIMARY KEY,
            obs_date DATE NOT NULL
        )
    """
    conn.execute("DROP TABLE IF EXISTS obs_points")
    conn.execute(obs_table_ddl)
    conn.executemany(
        "INSERT INTO obs_points (customer_id, obs_date) VALUES (?, ?)",
        obs_df.values.tolist(),
    )
    conn.commit()

    df = pd.read_sql_query("""
        WITH txn_agg AS (
            SELECT
                t.customer_id,
                COALESCE(SUM(t.amount), 0)     AS total_spend,
                COALESCE(AVG(t.amount), 0)     AS avg_order_value,
                COUNT(t.transaction_id)         AS num_transactions
            FROM transactions t
            JOIN obs_points o ON t.customer_id = o.customer_id
            WHERE t.transaction_date <= o.obs_date
            GROUP BY t.customer_id
        ),
        latest_sub AS (
            SELECT
                s.customer_id,
                s.plan,
                s.mrr,
                ROW_NUMBER() OVER (
                    PARTITION BY s.customer_id
                    ORDER BY s.start_date DESC
                ) AS rn
            FROM subscriptions s
            JOIN obs_points o ON s.customer_id = o.customer_id
            WHERE s.start_date <= o.obs_date
        ),
        event_agg AS (
            SELECT
                e.customer_id,
                COUNT(*)                                                        AS total_events,
                SUM(CASE WHEN e.event_type = 'login' THEN 1 ELSE 0 END)       AS login_count,
                SUM(CASE WHEN e.event_type = 'feature_use' THEN 1 ELSE 0 END) AS feature_use_count,
                SUM(CASE WHEN e.event_type = 'support_ticket' THEN 1 ELSE 0 END) AS support_ticket_count,
                SUM(CASE WHEN e.event_type = 'page_view' THEN 1 ELSE 0 END)   AS page_view_count,
                SUM(CASE WHEN e.event_date >= date(o.obs_date, '-30 days')
                         THEN 1 ELSE 0 END)  AS events_last_30d,
                SUM(CASE WHEN e.event_date >= date(o.obs_date, '-7 days')
                         THEN 1 ELSE 0 END)  AS events_last_7d,
                MAX(e.event_date)             AS last_event_date
            FROM events e
            JOIN obs_points o ON e.customer_id = o.customer_id
            WHERE e.event_date <= o.obs_date
            GROUP BY e.customer_id
        )
        SELECT
            c.customer_id,
            CAST(julianday(o.obs_date) - julianday(c.signup_date) AS INTEGER) AS tenure_days,
            c.age,
            c.country,
            c.acquisition_channel,
            COALESCE(t.total_spend, 0)          AS total_spend,
            COALESCE(t.avg_order_value, 0)      AS avg_order_value,
            COALESCE(t.num_transactions, 0)     AS num_transactions,
            COALESCE(ls.plan, 'free')           AS plan,
            COALESCE(ls.mrr, 0)                 AS current_mrr,
            COALESCE(e.total_events, 0)         AS total_events,
            COALESCE(e.login_count, 0)          AS login_count,
            COALESCE(e.feature_use_count, 0)    AS feature_use_count,
            COALESCE(e.support_ticket_count, 0) AS support_ticket_count,
            COALESCE(e.page_view_count, 0)      AS page_view_count,
            COALESCE(e.events_last_30d, 0)      AS events_last_30d,
            COALESCE(e.events_last_7d, 0)       AS events_last_7d,
            CASE
                WHEN e.last_event_date IS NOT NULL
                THEN CAST(julianday(o.obs_date) - julianday(e.last_event_date) AS INTEGER)
                ELSE 999
            END AS days_since_last_event,
            c.is_churned
        FROM customers c
        JOIN obs_points o ON c.customer_id = o.customer_id
        LEFT JOIN txn_agg t ON c.customer_id = t.customer_id
        LEFT JOIN latest_sub ls ON c.customer_id = ls.customer_id AND ls.rn = 1
        LEFT JOIN event_agg e ON c.customer_id = e.customer_id
    """, conn)

    conn.execute("DROP TABLE IF EXISTS obs_points")
    conn.close()

    plan_map = {"free": 0, "starter": 1, "pro": 2, "enterprise": 3}
    df["plan_encoded"] = df["plan"].map(plan_map).fillna(0).astype(int)

    df["avg_events_per_week"] = (
        df["total_events"] / (df["tenure_days"].clip(lower=7) / 7)
    ).round(2)

    df["support_ticket_ratio"] = (
        df["support_ticket_count"] / df["total_events"].clip(lower=1)
    ).round(4)

    channel_dummies = pd.get_dummies(df["acquisition_channel"], prefix="channel")
    country_dummies = pd.get_dummies(df["country"], prefix="country")
    df = pd.concat([df, channel_dummies, country_dummies], axis=1)

    df.drop(columns=["plan", "acquisition_channel", "country"], inplace=True)

    return df
