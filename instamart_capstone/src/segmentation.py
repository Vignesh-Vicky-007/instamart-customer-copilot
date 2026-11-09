"""Part B: RFM + StandardScaler + K-Means (K=2..8), segments, and a lapse (churn) model."""
import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score, recall_score,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR, OUTPUT_DIR, SEED

SNAPSHOT = pd.Timestamp("2025-10-01")        # fixed snapshot: data window is 2025-04-01 .. 2025-09-30
LAPSE_CUTOFF = pd.Timestamp("2025-09-01")    # features use orders BEFORE this; label uses September
K_RANGE = range(2, 9)
RFM_COLS = ["recency", "frequency", "monetary"]
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery", "age", "tenure_days"]
CATEGORICAL = ["favourite_category", "city", "gender"]


def build_rfm(orders, snapshot=SNAPSHOT):
    """One row per customer with at least one COMPLETED order: recency (days), frequency, monetary (INR)."""
    done = orders[orders["status"] == "completed"]
    rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum")).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
    return rfm.drop(columns="last_order")[["customer_id", "recency", "frequency", "monetary"]]


def kmeans_sweep(rfm, out_dir=OUTPUT_DIR, k_range=K_RANGE, seed=SEED):
    """StandardScaler on RFM, fit K-Means for K=2..8, record inertia + silhouette, pick best K by silhouette."""
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[RFM_COLS])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    out_dir.mkdir(exist_ok=True, parents=True)
    scores.to_csv(out_dir / "kmeans_scores.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(scores["k"], scores["inertia"], marker="o")
    ax[0].set(title="Elbow (inertia)", xlabel="K")
    ax[1].plot(scores["k"], scores["silhouette"], marker="o", color="tab:green")
    ax[1].axvline(best_k, ls="--", color="grey")
    ax[1].set(title=f"Silhouette (best K={best_k})", xlabel="K")
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_k_selection.png", dpi=130)
    plt.close(fig)
    return scores, best_k, scaler, X


AT_RISK_DAYS = 60   # a segment whose average customer has not ordered for 60+ days is "at risk"


def label_segments(rfm):
    """Give each cluster a business name from its mean RFM profile (names made unique)."""
    prof = rfm.groupby("segment_id")[RFM_COLS].mean()
    names = {}
    for seg_id, row in prof.iterrows():
        if row["monetary"] >= prof["monetary"].quantile(.75) and row["frequency"] >= prof["frequency"].median():
            names[seg_id] = "Champions / Premium"
        elif row["recency"] > AT_RISK_DAYS:
            names[seg_id] = "At-Risk / Lapsing"
        elif row["recency"] <= prof["recency"].median() and row["frequency"] >= prof["frequency"].median():
            names[seg_id] = "Loyal / Active"
        else:
            names[seg_id] = "Low-Engagement / Occasional"
    seen = {}
    for seg_id, name in names.items():
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            names[seg_id] = f"{name} {seen[name]}"
    return rfm["segment_id"].map(names)


INTERPRETATION = {
    "Champions / Premium": "Order often AND spend the most - protect with premium service and early offers.",
    "Loyal / Active": "Order very frequently and recently - keep engaged with habit-building offers.",
    "At-Risk / Lapsing": "Have not ordered for 60+ days on average - target with win-back offers (retention list).",
    "Low-Engagement / Occasional": "Still ordering but infrequently and spending less - nudge with repeat-purchase offers.",
}


def interpret(name):
    base = name.rsplit(" ", 1)[0] if name[-1].isdigit() else name
    return INTERPRETATION.get(base, "")


def plot_clusters(rfm, scaler, out_dir=OUTPUT_DIR):
    """Cluster visualisations: PCA 2-D scatter, Recency-vs-Monetary scatter, segment sizes, mean RFM per segment."""
    from sklearn.decomposition import PCA
    out_dir.mkdir(exist_ok=True, parents=True)
    X = scaler.transform(rfm[RFM_COLS])
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(X)
    names = sorted(rfm["segment"].unique())
    colors = dict(zip(names, plt.cm.tab10.colors))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for n in names:
        m = (rfm["segment"] == n).to_numpy()
        ax[0, 0].scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=.6, color=colors[n], label=n)
        ax[0, 1].scatter(rfm.loc[m, "recency"], rfm.loc[m, "monetary"], s=8, alpha=.6, color=colors[n])
    ax[0, 0].set(title="Segments in PCA space (2 components of scaled RFM)", xlabel="PC1", ylabel="PC2")
    ax[0, 0].legend(fontsize=8, markerscale=2)
    ax[0, 1].set(title="Recency vs Monetary", xlabel="recency (days)", ylabel="monetary (INR)")
    rfm["segment"].value_counts().reindex(names).plot.bar(ax=ax[1, 0], color=[colors[n] for n in names], title="Customers per segment")
    ax[1, 0].tick_params(axis="x", rotation=20)
    mean = rfm.groupby("segment")[RFM_COLS].mean()
    (mean / mean.max()).reindex(names).plot.bar(ax=ax[1, 1], title="Mean RFM per segment (scaled to max = 1)")
    ax[1, 1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_visualisation.png", dpi=120)
    plt.close(fig)


def segment(rfm, out_dir=OUTPUT_DIR, models_dir=MODELS_DIR):
    """Segment customers with the best K. Returns (rfm + segment columns, K scores, best K, segment profile)."""
    scores, best_k, scaler, X = kmeans_sweep(rfm, out_dir)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    rfm = rfm.assign(segment_id=km.labels_)
    rfm["segment"] = label_segments(rfm)
    models_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump({"kmeans": km, "scaler": scaler}, models_dir / "kmeans_segments.joblib")
    profile = rfm.groupby("segment").agg(customers=("customer_id", "count"), recency=("recency", "mean"),
                                         frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(1)
    profile["interpretation"] = [interpret(n) for n in profile.index]
    profile.to_csv(out_dir / "segment_profile.csv")
    plot_clusters(rfm, scaler, out_dir)
    rfm.to_csv(out_dir / "customer_segments.csv", index=False)
    return rfm, scores, best_k, profile


def build_lapse_dataset(orders, customers, products):
    """Leakage-free: features from orders before LAPSE_CUTOFF, label = no completed order in September."""
    hist = orders[orders["order_ts"] < LAPSE_CUTOFF]
    done = hist[hist["status"] == "completed"].merge(products[["product_id", "category"]], on="product_id")
    g = done.groupby("customer_id")
    data = g.agg(last_order=("order_ts", "max"), first_order=("order_ts", "min"), frequency=("order_id", "nunique"),
                 monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                 avg_delivery=("delivery_minutes", lambda s: s.astype(float).mean())).reset_index()
    data["recency"] = (LAPSE_CUTOFF - data["last_order"]).dt.days
    final = hist[hist["status"].isin(["completed", "cancelled"])].groupby("customer_id")["status"].apply(
        lambda s: (s == "cancelled").mean()).rename("cancellation_rate")
    fav = done.groupby(["customer_id", "category"]).size().rename("n").reset_index().sort_values(
        ["customer_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("customer_id")
    fav = fav.rename(columns={"category": "favourite_category"})[["customer_id", "favourite_category"]]
    data = data.merge(final, on="customer_id", how="left").merge(fav, on="customer_id", how="left")
    data = data.merge(customers[["customer_id", "city", "gender", "age", "signup_date"]], on="customer_id", how="left")
    data["tenure_days"] = (LAPSE_CUTOFF - pd.to_datetime(data["signup_date"])).dt.days
    sept = orders[(orders["order_ts"] >= LAPSE_CUTOFF) & (orders["status"] == "completed")]["customer_id"].unique()
    data["lapsed"] = (~data["customer_id"].isin(sept)).astype(int)
    data[NUMERIC] = data[NUMERIC].fillna(data[NUMERIC].median())
    data[CATEGORICAL] = data[CATEGORICAL].fillna("Unknown")
    return data


def churn_model(orders, customers, products, out_dir=OUTPUT_DIR, models_dir=MODELS_DIR, seed=SEED):
    """Train/test split, GridSearchCV over logistic regression and random forest, report precision/recall/F1."""
    data = build_lapse_dataset(orders, customers, products)
    X, y = data[NUMERIC + CATEGORICAL], data["lapsed"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL)])
    candidates = {
        "logistic_regression": (LogisticRegression(max_iter=2000), {"model__C": [0.01, 0.1, 1, 10, 100]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=seed),
                          {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}),
    }
    rows, fitted = [], {}
    for name, (est, grid) in candidates.items():
        search = GridSearchCV(Pipeline([("prep", prep), ("model", est)]), grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(X_tr, y_tr)
        pred = search.predict(X_te)
        rows.append({"model": name, "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred, zero_division=0),
                     "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred), "best_params": search.best_params_})
        fitted[name] = search
    metrics = pd.DataFrame(rows)
    metrics[["accuracy", "precision", "recall", "f1"]] = metrics[["accuracy", "precision", "recall", "f1"]].round(3)
    best = metrics.loc[metrics["f1"].idxmax(), "model"]
    baseline = max(y_te.mean(), 1 - y_te.mean())

    out_dir.mkdir(exist_ok=True, parents=True)
    metrics.to_csv(out_dir / "churn_metrics.csv", index=False)
    report = classification_report(y_te, fitted[best].predict(X_te), target_names=["active", "lapsed"])
    (out_dir / "churn_report.txt").write_text(
        f"Lapse = no completed order in September 2025 (features use orders before 2025-09-01)\n"
        f"Customers: {len(data)}, lapsed rate: {y.mean():.3f}, majority-class baseline accuracy: {baseline:.3f}\n"
        f"Best model: {best}\n\n{metrics.to_string(index=False)}\n\n{report}")
    models_dir.mkdir(exist_ok=True, parents=True)
    for fname in ("laps_model.joblib", "lapse_model.joblib"):   # transcript spelling + teacher's spelling
        joblib.dump(fitted[best], models_dir / fname)
    return metrics, best, fitted[best], data, baseline


def retention_list(segmented, data, model, out_dir=OUTPUT_DIR, top_n=100):
    """Rank customers by predicted lapse probability (ties -> higher spend) for the retention team."""
    scored = data.copy()
    scored["lapse_probability"] = model.predict_proba(scored[NUMERIC + CATEGORICAL])[:, 1]
    scored = scored.merge(segmented[["customer_id", "segment"]], on="customer_id", how="left")
    scored = scored.sort_values(["lapse_probability", "monetary"], ascending=[False, False]).head(top_n)
    out_dir.mkdir(exist_ok=True, parents=True)
    scored[["customer_id", "city", "segment", "recency", "frequency", "monetary", "lapse_probability"]] \
        .round(3).to_csv(out_dir / "retention_list.csv", index=False)
    return scored
