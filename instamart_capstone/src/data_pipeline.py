"""Part A: load, profile, clean, store in SQLite, validate (Pandas vs SQL)."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import DB_PATH, RAW_DIR

CITY_MAP = {c.lower(): c for c in ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]}
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
AGE_RANGE = (10, 100)          # ages outside this are impossible values (outliers)

CLEANING_RULES = """\
CLEANING RULES
customers/orders: rows with a blank customer_id / order_id / product_id are dropped (cannot be joined).
customers: city -> strip + lowercase, mapped to 6 canonical cities; gender (male/M/FEMALE/f/blank...) -> Male / Female / Unknown;
  age blank or outside 10-100 (outlier, e.g. -4, 0, 150, 999) -> NaN, then filled with the median age of the customer's city.
orders: exact duplicate rows dropped; if an order_id appears more than once (pending copy + final row) keep the LATEST order_ts;
  order_amount 'N/A'/'error' -> imputed as product price * quantity * (1 - discount_pct/100);
  negative delivery_minutes -> NaN; blank/NaN delivery of COMPLETED orders -> median delivery of that product category;
  delivery_minutes of cancelled orders stays empty (never delivered).
tickets: text stripped; empty/whitespace and over-long tickets are kept here and blocked later by the Part C guardrails.
"""


def load_raw(raw_dir=RAW_DIR):
    raw = Path(raw_dir)
    kw = dict(keep_default_na=False)   # keep blanks as '' so they can be counted and cleaned explicitly
    return (pd.read_csv(raw / "customers.csv", **kw), pd.read_csv(raw / "orders.csv", **kw),
            pd.read_csv(raw / "tickets.csv", **kw), pd.read_csv(raw / "products.csv"))


def profile(df, name, max_unique=40):
    """Shape, dtypes, blanks, and value counts with % for categorical columns."""
    blanks = df.astype(str).apply(lambda s: s.str.strip().eq("")).sum() + df.isna().sum()
    lines = [f"=== {name}: shape={df.shape} ===", "Data types:", str(df.dtypes), "Blank/null counts:", str(blanks)]
    for col in df.columns:
        n = df[col].nunique(dropna=False)
        if df[col].dtype == object and n <= max_unique:
            counts = df[col].value_counts(dropna=False)
            table = pd.DataFrame({"count": counts, "pct": (counts / len(df) * 100).round(2)})
            lines.append(f"\nValue counts for {col} ({n} unique):\n{table.to_string()}")
    return "\n".join(lines)


def audit_orders(orders):
    """Counts of the data problems found in raw orders (checkpoints)."""
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    return {"rows": len(orders), "exact_duplicate_rows": int(orders.duplicated().sum()),
            "duplicate_order_ids": int(orders["order_id"].duplicated().sum()),
            "pending_rows": int(orders["status"].eq("pending").sum()),
            "bad_order_amount": int(amount.isna().sum()), "negative_delivery": int((delivery < 0).sum())}


def drop_missing_ids(df, id_cols):
    """Rows with a blank/missing key cannot be joined to anything, so they are removed (documented rule)."""
    ok = df[id_cols].astype(str).apply(lambda s: s.str.strip().ne("") & s.str.lower().ne("nan")).all(axis=1)
    return df[ok].copy()


def clean_customers(df):
    df = drop_missing_ids(df.drop_duplicates(), ["customer_id"])
    df["city"] = df["city"].astype(str).str.strip().str.lower().map(CITY_MAP).fillna("Unknown")
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(*AGE_RANGE), "age"] = np.nan
    df["age"] = df["age"].fillna(df.groupby("city")["age"].transform("median")).fillna(df["age"].median()).round().astype(int)
    return df.reset_index(drop=True)


def clean_orders(df, products):
    df = drop_missing_ids(df.drop_duplicates(), ["order_id", "customer_id", "product_id"])
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    # same order_id twice (earlier pending copy + final row): the latest timestamp wins
    df = df.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").copy()
    for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[["order_amount", "delivery_minutes"]] = df[["order_amount", "delivery_minutes"]].astype(float)   # safe even if nothing needs imputing
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    price = products.set_index("product_id")["price_inr"]
    miss = df["order_amount"].isna()
    if miss.any():
        df.loc[miss, "order_amount"] = (df.loc[miss, "product_id"].map(price).astype(float) * df.loc[miss, "quantity"]
                                        * (1 - df.loc[miss, "discount_pct"] / 100)).round(2)
    category = df["product_id"].map(products.set_index("product_id")["category"])
    done = df["status"].eq("completed")
    cat_median = df[done].groupby(category[done])["delivery_minutes"].median()
    fill = done & df["delivery_minutes"].isna()
    if fill.any():
        df.loc[fill, "delivery_minutes"] = category[fill].map(cat_median).astype(float)
    df["delivery_minutes"] = df["delivery_minutes"].round().astype("Int64")
    df["quantity"] = df["quantity"].astype(int)
    return df.reset_index(drop=True)


def clean_tickets(df):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    return df


def clean_all(raw_dir=RAW_DIR):
    """Returns cleaned customers, orders, tickets, products, a profile report, the order audit, row counts before/after."""
    customers, orders, tickets, products = load_raw(raw_dir)
    audit = audit_orders(orders)
    raw_city_spellings = int(customers["city"].nunique())
    report = "\n\n".join([profile(customers, "customers"), profile(orders, "orders"), profile(tickets, "tickets")])
    before = {"customers": len(customers), "orders": len(orders), "tickets": len(tickets)}

    customers, orders, tickets = clean_customers(customers), clean_orders(orders, products), clean_tickets(tickets)
    after = {"customers": len(customers), "orders": len(orders), "tickets": len(tickets)}
    report += "\n\n" + "\n\n".join([profile(customers, "customers AFTER clean"), profile(orders, "orders AFTER clean"),
                                    profile(tickets, "tickets AFTER clean")])
    audit["raw_city_spellings"] = raw_city_spellings
    return customers, orders, tickets, products, report, audit, before, after


def to_sqlite(customers, orders, tickets, products, db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        orders.assign(order_ts=orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_sql(
            "orders", con, if_exists="replace", index=False)
        tickets.to_sql("tickets", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)


def validate(customers, orders, tickets, db_path=DB_PATH):
    """Cross-check Pandas results against SQL. Returns (report_text, city_table, all_passed)."""
    done = orders[orders["status"] == "completed"]
    checks = [
        ("customer rows", len(customers), "SELECT COUNT(*) FROM customers"),
        ("order rows", len(orders), "SELECT COUNT(*) FROM orders"),
        ("ticket rows", len(tickets), "SELECT COUNT(*) FROM tickets"),
        ("distinct cities", customers["city"].nunique(), "SELECT COUNT(DISTINCT city) FROM customers"),
        ("completed orders", len(done), "SELECT COUNT(*) FROM orders WHERE status='completed'"),
        ("pending orders", int((orders["status"] == "pending").sum()), "SELECT COUNT(*) FROM orders WHERE status='pending'"),
        ("sum completed amount", round(done["order_amount"].sum(), 2),
         "SELECT ROUND(SUM(order_amount),2) FROM orders WHERE status='completed'"),
        ("duplicate order_ids", int(orders["order_id"].duplicated().sum()),
         "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders"),
        ("unknown gender", int((customers["gender"] == "Unknown").sum()),
         "SELECT COUNT(*) FROM customers WHERE gender='Unknown'"),
        ("avg completed delivery", round(done["delivery_minutes"].astype(float).mean(), 3),
         "SELECT ROUND(AVG(delivery_minutes),3) FROM orders WHERE status='completed'"),
    ]
    results = []
    with sqlite3.connect(db_path) as con:
        for name, pandas_val, sql in checks:
            sql_val = con.execute(sql).fetchone()[0]
            passed = abs(float(pandas_val) - float(sql_val)) < 1e-6
            results.append((name, pandas_val, sql_val, "PASS" if passed else "FAIL"))
        city_table = pd.read_sql("SELECT city, COUNT(*) AS customers FROM customers GROUP BY city ORDER BY 2 DESC", con)
    report = "\n".join(f"{n:24s} pandas={p}  sql={s}  {r}" for n, p, s, r in results)
    return report, city_table, all(r[3] == "PASS" for r in results)


# ---------------------------------------------------------------- Part A: reports, charts, SQL analysis
def quality_snapshot(customers, orders, tickets):
    """Data-quality metrics that can be measured on BOTH the raw and the cleaned tables."""
    age = pd.to_numeric(customers["age"], errors="coerce")
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    done = orders["status"].astype(str).eq("completed")
    gender = customers["gender"].astype(str).str.strip()
    return {
        "customer rows": len(customers), "order rows": len(orders), "ticket rows": len(tickets),
        "exact duplicate order rows": int(orders.astype(str).duplicated().sum()),
        "repeated order_id rows": int(orders["order_id"].duplicated().sum()),
        "distinct city spellings": int(customers["city"].nunique()),
        "distinct gender spellings": int(gender.nunique()),
        "blank gender": int(gender.eq("").sum()),
        "impossible/blank age (outside 10-100)": int((~age.between(*AGE_RANGE)).sum()),
        "non-numeric order_amount": int(amount.isna().sum()),
        "negative delivery_minutes": int((delivery < 0).sum()),
        "completed orders without delivery time": int((done & delivery.isna()).sum()),
        "pending rows": int(orders["status"].astype(str).eq("pending").sum()),
        "blank customer/order ids": int(orders[["order_id", "customer_id"]].astype(str).apply(
            lambda s: s.str.strip().eq("")).any(axis=1).sum()),
    }


def iqr_outliers(orders):
    """IQR (1.5 x IQR) outlier counts for the numeric order columns; reported, not blindly deleted."""
    lines = []
    for col in ["order_amount", "delivery_minutes"]:
        x = pd.to_numeric(orders[col], errors="coerce").dropna()
        q1, q3 = x.quantile(.25), x.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        lines.append(f"{col}: IQR fence [{lo:.1f}, {hi:.1f}] -> {int(((x < lo) | (x > hi)).sum())} statistical outliers "
                     f"({((x < lo) | (x > hi)).mean() * 100:.1f}%), max={x.max():.0f}")
    return "\n".join(lines)


def cleaning_report(raw, clean):
    """Before/after data-quality table + decisions. raw/clean = (customers, orders, tickets)."""
    before, after = quality_snapshot(*raw), quality_snapshot(*clean)
    table = pd.DataFrame({"before": before, "after": after})
    table["fixed"] = table["before"] != table["after"]
    return ("DATA-CLEANING REPORT (before vs after)\n" + table.to_string() + "\n\n"
            "Outlier analysis on RAW orders (IQR method)\n" + iqr_outliers(raw[1]) + "\n"
            "Decision: high order amounts are real (electronics at INR 1,299-1,999 x quantity) so they are KEPT; "
            "only impossible values (age -4/0/150/999, negative delivery minutes, non-numeric amounts) are corrected.\n\n"
            + CLEANING_RULES)


def eda_charts(raw, clean, products, out_dir):
    """Write EDA PNGs: quality before/after, categorical distributions, numeric distributions, trends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (rc, ro, rt), (cc, co, ct) = raw, clean
    saved = []

    q = pd.DataFrame({"before": quality_snapshot(rc, ro, rt), "after": quality_snapshot(cc, co, ct)}).iloc[3:]
    ax = q.plot.barh(figsize=(9, 6), color=["#d93025", "#1e8e3e"])
    ax.set(title="Data quality issues: before vs after cleaning", xlabel="rows / distinct values")
    plt.tight_layout(); plt.savefig(out_dir / "eda_1_quality_before_after.png", dpi=120); plt.close(); saved.append("eda_1")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    co["status"].value_counts().plot.bar(ax=ax[0, 0], title="Order status (clean)", color="#fc8019")
    cc["city"].value_counts().plot.bar(ax=ax[0, 1], title="Customers per city (clean)", color="#3b82f6")
    cc["gender"].value_counts().plot.bar(ax=ax[1, 0], title="Gender (clean)", color="#8b5cf6")
    ct["true_category"].value_counts().plot.bar(ax=ax[1, 1], title="Ticket categories", color="#14b8a6")
    for a in ax.flat:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out_dir / "eda_2_categorical.png", dpi=120); plt.close(fig); saved.append("eda_2")

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    age_raw = pd.to_numeric(rc["age"], errors="coerce")
    ax[0, 0].hist(age_raw.dropna(), bins=40, color="#d93025"); ax[0, 0].set_title("Age RAW (outliers -4..999)")
    ax[1, 0].hist(cc["age"], bins=30, color="#1e8e3e"); ax[1, 0].set_title("Age CLEAN")
    dm_raw = pd.to_numeric(ro["delivery_minutes"], errors="coerce")
    ax[0, 1].hist(dm_raw.dropna(), bins=40, color="#d93025"); ax[0, 1].set_title("Delivery minutes RAW (negatives)")
    ax[1, 1].hist(co["delivery_minutes"].dropna().astype(float), bins=40, color="#1e8e3e"); ax[1, 1].set_title("Delivery minutes CLEAN")
    ax[0, 2].hist(pd.to_numeric(ro["order_amount"], errors="coerce").dropna(), bins=40, color="#d93025"); ax[0, 2].set_title("Order amount RAW")
    ax[1, 2].hist(co["order_amount"], bins=40, color="#1e8e3e"); ax[1, 2].set_title("Order amount CLEAN")
    fig.tight_layout(); fig.savefig(out_dir / "eda_3_numeric_distributions.png", dpi=120); plt.close(fig); saved.append("eda_3")

    done = co[co["status"] == "completed"].merge(products[["product_id", "category"]], on="product_id")
    monthly = done.groupby(done["order_ts"].dt.to_period("M")).agg(orders=("order_id", "count"), revenue=("order_amount", "sum"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    monthly["orders"].plot(ax=ax[0], marker="o", title="Completed orders per month"); ax[0].set_xlabel("")
    done.groupby("category")["order_amount"].sum().sort_values().plot.barh(ax=ax[1], title="Revenue by product category (INR)", color="#fc8019")
    fig.tight_layout(); fig.savefig(out_dir / "eda_4_trends_and_revenue.png", dpi=120); plt.close(fig); saved.append("eda_4")
    return saved


def load_sql_queries(path=None):
    """Read sql/queries.sql: blocks of '-- title' followed by one statement ending in ';'."""
    path = path or Path(__file__).resolve().parent.parent / "sql" / "queries.sql"
    blocks = [b.strip() for b in Path(path).read_text().split("\n-- ")[1:]]
    return {b.split("\n", 1)[0]: b.split("\n", 1)[1].strip().rstrip(";") for b in blocks}


SQL_QUERIES = load_sql_queries()


def sql_analysis(db_path=DB_PATH):
    """Run the join queries against SQLite; returns report text."""
    parts = []
    with sqlite3.connect(db_path) as con:
        for title, sql in SQL_QUERIES.items():
            parts.append(f"-- {title}\n{sql}\n{pd.read_sql(sql, con).to_string(index=False)}\n")
    return "\n".join(parts)
