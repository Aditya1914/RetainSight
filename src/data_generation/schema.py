"""
RetainSight database schema.

Tables:
  - customers: demographics, signup info, subscription plan
  - subscriptions: plan history with start/end dates and MRR
  - transactions: individual purchases with amount and product category
  - events: behavioral signals (login, page_view, support_ticket, feature_use)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "retainsight.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id     INTEGER PRIMARY KEY,
        name            TEXT NOT NULL,
        email           TEXT NOT NULL UNIQUE,
        signup_date     DATE NOT NULL,
        country         TEXT NOT NULL,
        age             INTEGER NOT NULL,
        gender          TEXT NOT NULL,
        acquisition_channel TEXT NOT NULL,   -- organic, paid, referral, social
        is_churned      INTEGER NOT NULL DEFAULT 0,
        churn_date      DATE
    );

    CREATE TABLE IF NOT EXISTS subscriptions (
        subscription_id INTEGER PRIMARY KEY,
        customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
        plan            TEXT NOT NULL,       -- free, starter, pro, enterprise
        mrr             REAL NOT NULL,       -- monthly recurring revenue in USD
        start_date      DATE NOT NULL,
        end_date        DATE,               -- NULL = still active
        status          TEXT NOT NULL        -- active, cancelled, upgraded, downgraded
    );

    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id  INTEGER PRIMARY KEY,
        customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
        transaction_date DATE NOT NULL,
        amount          REAL NOT NULL,
        product_category TEXT NOT NULL,      -- subscription, addon, service, one_time
        payment_method  TEXT NOT NULL        -- credit_card, debit_card, paypal, bank_transfer
    );

    CREATE TABLE IF NOT EXISTS events (
        event_id        INTEGER PRIMARY KEY,
        customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
        event_date      DATE NOT NULL,
        event_type      TEXT NOT NULL,       -- login, page_view, feature_use, support_ticket, feedback
        event_detail    TEXT                 -- e.g. feature name, page name, ticket subject
    );

    CREATE INDEX IF NOT EXISTS idx_sub_customer ON subscriptions(customer_id);
    CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
    CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date);
    CREATE INDEX IF NOT EXISTS idx_evt_customer ON events(customer_id);
    CREATE INDEX IF NOT EXISTS idx_evt_date ON events(event_date);
    CREATE INDEX IF NOT EXISTS idx_evt_type ON events(event_type);
    """)
    conn.commit()
