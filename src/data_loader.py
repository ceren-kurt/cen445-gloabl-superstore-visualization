from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "superstore.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE / "superstore.csv"

DISCOUNT_LABELS = ["0%", "1%-20%", "21%-40%", "41%+"]

NUM = ["discount", "sales", "quantity", "shipping_cost", "delivery_days"]
CAT = ["category", "market", "segment", "ship_mode"]

COUNTRY_FIX = {
    "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Congo (DRC)": "Democratic Republic of the Congo",
    "Congo (Brazzaville)": "Republic of the Congo",
    "Congo": "Republic of the Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Myanmar (Burma)": "Myanmar",
    "Burma": "Myanmar",
    "Czechia": "Czech Republic",
    "Swaziland": "eSwatini",
    "Eswatini": "eSwatini",
    "East Timor": "Timor-Leste",
    "Macedonia": "North Macedonia",
    "The Gambia": "Gambia",
    "Hong Kong SAR": "Hong Kong",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Russian Federation": "Russia",
    "Viet Nam": "Vietnam",
    "Lao PDR": "Laos",
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    report = {"Raw rows": len(df), "Raw columns": df.shape[1]}

    df.columns = df.columns.str.replace(".", "_", regex=False).str.lower()
    df = df.drop(columns=["记录数", "row_id", "market2", "weeknum"], errors="ignore")

    if "postal_code" in df.columns:
        report["postal_code missing (%)"] = round(df["postal_code"].isna().mean() * 100, 1)
        df = df.drop(columns=["postal_code"])

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    before = len(df)
    df = df.dropna()
    report["Rows removed (missing values)"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["Rows removed (duplicates)"] = before - len(df)

    df["delivery_days"] = (df["ship_date"] - df["order_date"]).dt.days
    before = len(df)
    df = df[df["delivery_days"] >= 0]
    report["Rows removed (negative delivery time)"] = before - len(df)

    df["country"] = df["country"].str.strip().replace(COUNTRY_FIX)
    df["year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["discount_group"] = pd.cut(
        df["discount"], bins=[-0.01, 0.0, 0.20, 0.40, 1.0], labels=DISCOUNT_LABELS
    )

    q1, q3 = df["profit"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["profit_outlier"] = (df["profit"] < q1 - 1.5 * iqr) | (df["profit"] > q3 + 1.5 * iqr)
    report["Profit outliers flagged (IQR)"] = int(df["profit_outlier"].sum())

    df["is_loss"] = (df["profit"] < 0).astype(int)

    df = df.reset_index(drop=True)
    report["Clean rows"] = len(df)
    report["Clean columns"] = df.shape[1]
    return df, report


def margin_by(data: pd.DataFrame, by) -> pd.DataFrame:
    g = data.groupby(by, as_index=False, observed=True)[["sales", "profit"]].sum()
    g["margin"] = np.where(g["sales"] > 0, g["profit"] / g["sales"] * 100, np.nan)
    return g


@st.cache_data
def segment_customers(cust: pd.DataFrame, k: int) -> pd.DataFrame:
    X = StandardScaler().fit_transform(cust)
    out = cust.copy()
    out["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X).astype(str)
    comp = PCA(n_components=2, random_state=42).fit_transform(X)
    out["PC1"], out["PC2"] = comp[:, 0], comp[:, 1]
    out["avg_discount_pct"] = out["avg_discount"] * 100
    return out


@st.cache_resource
def train_model(_data: pd.DataFrame):
    Xtr, Xte, ytr, yte = train_test_split(
        _data[NUM + CAT], _data["is_loss"], test_size=0.2, random_state=42,
        stratify=_data["is_loss"],
    )
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough"
    )
    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)),
    ])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    cm = confusion_matrix(yte, pred)
    scores = {
        "accuracy": accuracy_score(yte, pred),
        "auc": roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]),
        "baseline": 1 - yte.mean(),
        "loss_recall": cm[1, 1] / cm[1].sum(),
    }
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=names)
    return pipe, scores, cm, imp