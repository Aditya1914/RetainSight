"""
Synthetic data generator for RetainSight.

Generates ~2000 customers with realistic behavioral patterns:
  - Churned customers *tend* to have lower engagement, but with significant overlap
  - Some active customers are low-engagement; some churned were highly active
  - Churn probability influenced by channel, plan, and tenure (not just events)
  - Noise ensures the ML problem isn't trivially separable
"""

import random
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

from .schema import create_tables, get_connection

SEED = int(time.time()) % 100_000

COUNTRIES = ["US", "UK", "India", "Germany", "Canada", "Australia", "France", "Brazil"]
CHANNELS = ["organic", "paid", "referral", "social"]
PLANS = {
    "free":       0.0,
    "starter":    19.0,
    "pro":        49.0,
    "enterprise": 149.0,
}
PRODUCT_CATEGORIES = ["subscription", "addon", "service", "one_time"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
EVENT_TYPES = ["login", "page_view", "feature_use", "support_ticket", "feedback"]
FEATURE_NAMES = ["dashboard", "reports", "export", "integrations", "api", "settings"]
PAGE_NAMES = ["home", "pricing", "docs", "blog", "changelog", "profile"]

FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Alex", "Amit", "Ana", "Ben", "Carlos",
    "Chen", "David", "Elena", "Emma", "Fatima", "Grace", "Hans", "Isha",
    "James", "Julia", "Kai", "Liam", "Maria", "Mei", "Noah", "Olivia",
    "Priya", "Raj", "Sara", "Tom", "Uma", "Victor", "Wei", "Yuki", "Zara",
    "Arjun", "Diya", "Ethan", "Freya", "George", "Hannah", "Ivan",
]
LAST_NAMES = [
    "Smith", "Patel", "Garcia", "Mueller", "Wang", "Silva", "Johnson",
    "Kim", "Brown", "Singh", "Martin", "Lee", "Wilson", "Kumar", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris", "Clark", "Lewis",
    "Walker", "Hall", "Young", "Allen", "Wright", "Lopez", "Hill", "Scott",
]

SIM_START = date(2023, 1, 1)
SIM_END = date(2025, 3, 31)
NUM_CUSTOMERS = 2000
BASE_CHURN_RATE = 0.25

CHANNEL_CHURN_MODIFIER = {
    "organic": -0.05,
    "referral": -0.08,
    "paid": +0.06,
    "social": +0.04,
}
PLAN_CHURN_MODIFIER = {
    "free": +0.12,
    "starter": +0.02,
    "pro": -0.05,
    "enterprise": -0.10,
}


def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def _generate_customers() -> list[dict]:
    customers = []
    plan_list = list(PLANS.keys())
    plan_weights = [0.15, 0.35, 0.35, 0.15]

    for cid in range(1, NUM_CUSTOMERS + 1):
        signup = _rand_date(SIM_START, SIM_END - timedelta(days=60))
        channel = random.choice(CHANNELS)
        plan = random.choices(plan_list, weights=plan_weights, k=1)[0]

        tenure_days = (SIM_END - signup).days
        tenure_modifier = 0.05 if tenure_days < 120 else (-0.05 if tenure_days > 500 else 0.0)

        churn_prob = BASE_CHURN_RATE + CHANNEL_CHURN_MODIFIER[channel] + PLAN_CHURN_MODIFIER[plan] + tenure_modifier
        churn_prob = max(0.05, min(0.50, churn_prob))

        is_churned = 1 if random.random() < churn_prob else 0
        churn_date = None
        if is_churned:
            earliest_churn = signup + timedelta(days=14)
            if earliest_churn < SIM_END:
                churn_date = _rand_date(earliest_churn, SIM_END)
            else:
                is_churned = 0

        customers.append({
            "customer_id": cid,
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "email": f"user{cid}@example.com",
            "signup_date": signup.isoformat(),
            "country": random.choice(COUNTRIES),
            "age": random.randint(18, 65),
            "gender": random.choice(["M", "F", "Other"]),
            "acquisition_channel": channel,
            "initial_plan": plan,
            "is_churned": is_churned,
            "churn_date": churn_date.isoformat() if churn_date else None,
        })
    return customers


def _generate_subscriptions(customers: list[dict]) -> list[dict]:
    subs = []
    sid = 1
    plan_list = list(PLANS.keys())

    for c in customers:
        signup = date.fromisoformat(c["signup_date"])
        churn_dt = date.fromisoformat(c["churn_date"]) if c["churn_date"] else None
        end_boundary = churn_dt or SIM_END

        current_plan = c["initial_plan"]
        current_start = signup

        num_changes = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1], k=1)[0]
        for i in range(num_changes + 1):
            is_last = (i == num_changes)
            if is_last:
                end_dt = churn_dt
                status = "cancelled" if c["is_churned"] else "active"
            else:
                duration = random.randint(30, 180)
                end_dt = current_start + timedelta(days=duration)
                if end_dt >= end_boundary:
                    end_dt = churn_dt
                    status = "cancelled" if c["is_churned"] else "active"
                    is_last = True
                else:
                    old_idx = plan_list.index(current_plan)
                    if old_idx < len(plan_list) - 1 and random.random() < 0.6:
                        new_idx = old_idx + 1
                        status = "upgraded"
                    elif old_idx > 0:
                        new_idx = old_idx - 1
                        status = "downgraded"
                    else:
                        new_idx = old_idx + 1
                        status = "upgraded"

            mrr = PLANS[current_plan] * random.uniform(0.9, 1.1)

            subs.append({
                "subscription_id": sid,
                "customer_id": c["customer_id"],
                "plan": current_plan,
                "mrr": round(mrr, 2),
                "start_date": current_start.isoformat(),
                "end_date": end_dt.isoformat() if end_dt else None,
                "status": status,
            })
            sid += 1

            if is_last:
                break

            current_start = end_dt
            current_plan = plan_list[new_idx]

    return subs


def _generate_transactions(customers: list[dict]) -> list[dict]:
    txns = []
    tid = 1

    for c in customers:
        signup = date.fromisoformat(c["signup_date"])
        churn_dt = date.fromisoformat(c["churn_date"]) if c["churn_date"] else None
        end_boundary = churn_dt or SIM_END
        active_days = (end_boundary - signup).days
        if active_days <= 0:
            continue

        base_txns = random.randint(2, max(3, active_days // 30))
        if c["is_churned"]:
            reduction = random.choice([0.5, 0.6, 0.7, 0.8, 1.0])
            base_txns = max(1, int(base_txns * reduction))

        for _ in range(base_txns):
            txn_date = _rand_date(signup, end_boundary)
            category = random.choices(
                PRODUCT_CATEGORIES,
                weights=[0.4, 0.25, 0.2, 0.15],
                k=1,
            )[0]

            if category == "subscription":
                amount = random.uniform(15, 160)
            elif category == "addon":
                amount = random.uniform(5, 50)
            elif category == "service":
                amount = random.uniform(50, 500)
            else:
                amount = random.uniform(10, 200)

            txns.append({
                "transaction_id": tid,
                "customer_id": c["customer_id"],
                "transaction_date": txn_date.isoformat(),
                "amount": round(amount, 2),
                "product_category": category,
                "payment_method": random.choice(PAYMENT_METHODS),
            })
            tid += 1

    return txns


def _generate_events(customers: list[dict]) -> list[dict]:
    events = []
    eid = 1

    for c in customers:
        signup = date.fromisoformat(c["signup_date"])
        churn_dt = date.fromisoformat(c["churn_date"]) if c["churn_date"] else None
        end_boundary = churn_dt or SIM_END
        active_days = (end_boundary - signup).days
        if active_days <= 0:
            continue

        if c["is_churned"]:
            events_per_week = random.uniform(2, 10)
        else:
            events_per_week = random.uniform(3, 12)

        total_events = int(events_per_week * active_days / 7)
        total_events = max(2, min(total_events, 800))

        decline_rate = 0.0
        if c["is_churned"]:
            decline_rate = random.uniform(0.1, 0.35)

        for i in range(total_events):
            progress = i / max(total_events, 1)

            if c["is_churned"] and progress > 0.6:
                if random.random() < decline_rate:
                    continue

            evt_date = _rand_date(signup, end_boundary)

            if c["is_churned"] and progress > 0.7:
                evt_type = random.choices(
                    EVENT_TYPES,
                    weights=[0.25, 0.25, 0.1, 0.25, 0.15],
                    k=1,
                )[0]
            else:
                evt_type = random.choices(
                    EVENT_TYPES,
                    weights=[0.3, 0.3, 0.2, 0.1, 0.1],
                    k=1,
                )[0]

            if evt_type == "feature_use":
                detail = random.choice(FEATURE_NAMES)
            elif evt_type == "page_view":
                detail = random.choice(PAGE_NAMES)
            elif evt_type == "support_ticket":
                detail = random.choice(["billing", "bug", "feature_request", "how_to", "cancellation"])
            elif evt_type == "feedback":
                detail = random.choice(["positive", "neutral", "negative"])
            else:
                detail = None

            events.append({
                "event_id": eid,
                "customer_id": c["customer_id"],
                "event_date": evt_date.isoformat(),
                "event_type": evt_type,
                "event_detail": detail,
            })
            eid += 1

    return events


def _insert_batch(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def generate_all(db_path: Path | None = None, seed: int | None = None) -> Path:
    """Generate the full synthetic dataset and return the DB path."""
    actual_seed = seed if seed is not None else SEED
    random.seed(actual_seed)
    print(f"Using seed: {actual_seed}")

    if db_path:
        from .schema import DB_PATH as _  # noqa: F401
        import src.data_generation.schema as schema_mod
        schema_mod.DB_PATH = db_path

    from .schema import DB_PATH
    db_file = db_path or DB_PATH
    for suffix in ("", "-shm", "-wal"):
        p = db_file.parent / (db_file.name + suffix)
        if p.exists():
            p.unlink()

    conn = get_connection()
    create_tables(conn)

    print("Generating customers...")
    customers = _generate_customers()
    customer_rows = [{k: v for k, v in c.items() if k != "initial_plan"} for c in customers]
    _insert_batch(conn, "customers", customer_rows)

    print("Generating subscriptions...")
    subs = _generate_subscriptions(customers)
    _insert_batch(conn, "subscriptions", subs)

    print("Generating transactions...")
    txns = _generate_transactions(customers)
    _insert_batch(conn, "transactions", txns)

    print("Generating events...")
    events = _generate_events(customers)
    _insert_batch(conn, "events", events)

    conn.commit()
    conn.close()

    print(f"\nDatabase created at: {db_file}")
    print(f"  Customers:     {len(customers):,}")
    print(f"  Subscriptions: {len(subs):,}")
    print(f"  Transactions:  {len(txns):,}")
    print(f"  Events:        {len(events):,}")
    return DB_PATH


if __name__ == "__main__":
    generate_all()
