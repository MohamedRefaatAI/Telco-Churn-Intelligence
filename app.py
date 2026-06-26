"""
================================================================================
  Telco Customer Churn — Production Streamlit Dashboard
================================================================================
  Tabs:
    1. EDA           — KPIs + Interactive EDA charts
    2. Model Analytics — Performance comparison + confusion matrices
    3. Live Predictor  — Real-time churn probability scoring

  Author  : Senior Data Scientist & Full-Stack AI Engineer
  Palette : Muted Blue #4C9BE8 (No Churn) | Coral #E8634C (Churn)
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# ── Color palette ─────────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"   # No Churn
CORAL  = "#E8634C"   # Churn
AMBER  = "#F5C542"
GREEN  = "#52C77E"
BG     = "#0F1117"
CARD   = "#1A1D27"
BORDER = "#2A2D3E"
MUTED  = "#8B8FA8"


# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
#  GLOBAL CSS — dark premium theme
# =============================================================================
st.markdown("""
<style>
/* ── Root & body ── */
html, body, [data-testid="stApp"] {
    background-color: #0F1117;
    color: #E4E6F0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 2px solid #2A2D3E;
    padding-bottom: 0;
}
[data-testid="stTabs"] [role="tab"] {
    background: #1A1D27;
    color: #8B8FA8;
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid #2A2D3E;
    border-bottom: none;
    transition: all 0.2s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #252836;
    color: #4C9BE8;
    border-color: #4C9BE8;
    border-bottom: none;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #E4E6F0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700;
    color: #E4E6F0;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    color: #8B8FA8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

/* ── Selectbox / Slider labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    color: #C4C7D8;
    font-size: 0.83rem;
    font-weight: 500;
}

/* ── Selectbox dropdowns ── */
[data-testid="stSelectbox"] > div > div {
    background: #252836 !important;
    border: 1px solid #2A2D3E !important;
    color: #E4E6F0 !important;
    border-radius: 8px !important;
}

/* ── Section header ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #8B8FA8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 28px 0 14px 0;
    border-bottom: 1px solid #2A2D3E;
    padding-bottom: 8px;
}

/* ── Insight card ── */
.insight-card {
    background: #1A1D27;
    border: 1px solid #2A2D3E;
    border-left: 4px solid #4C9BE8;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #C4C7D8;
}

/* ── Churn warning / safe banner ── */
.churn-high {
    background: rgba(232, 99, 76, 0.12);
    border: 1px solid #E8634C;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.churn-low {
    background: rgba(82, 199, 126, 0.1);
    border: 1px solid #52C77E;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.churn-mid {
    background: rgba(245, 197, 66, 0.1);
    border: 1px solid #F5C542;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.banner-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.banner-sub {
    font-size: 0.9rem;
    color: #C4C7D8;
    margin-top: 6px;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2A2D3E;
}

/* ── Page title bar ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
}
.page-title {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4C9BE8, #7B6CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.page-sub {
    font-size: 0.88rem;
    color: #8B8FA8;
    margin: 4px 0 0 0;
}
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.pill-blue { background: rgba(76,155,232,0.15); color: #4C9BE8; }
.pill-coral { background: rgba(232,99,76,0.15); color: #E8634C; }

/* ── Divider ── */
.h-line { border: none; border-top: 1px solid #2A2D3E; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  DATA LOADING & CACHING
# =============================================================================

def _find_data_path() -> str:
    """
    Locate the CSV dynamically: checks the uploads mount, the current working
    directory, and a fallback path — so the app runs identically in a local
    dev environment or on a server.
    """
    candidates = [
        "/mnt/user-data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find the Telco Churn CSV. "
        "Place it in the same folder as app.py."
    )


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load and lightly pre-process the raw dataset (no feature engineering)."""
    path = _find_data_path()
    df = pd.read_csv(path)

    # Fix TotalCharges hidden blank strings
    df["TotalCharges"] = (
        pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    return df


@st.cache_data(show_spinner=False)
def build_feature_matrix(df: pd.DataFrame):
    """
    Mirrors Phase 2 + Phase 3 of the pipeline exactly so the predictor's
    live model matches the trained pipeline.
    Returns (X_train, X_test, y_train, y_test, feature_columns, scaler).
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_map = {
        "gender":          {"Male": 1, "Female": 0},
        "Partner":         {"Yes": 1, "No": 0},
        "Dependents":      {"Yes": 1, "No": 0},
        "PhoneService":    {"Yes": 1, "No": 0},
        "PaperlessBilling":{"Yes": 1, "No": 0},
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)

    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test, list(X.columns), scaler


@st.cache_resource(show_spinner=False)
def train_all_models(_X_train, _y_train):
    """
    Train all five models and cache the fitted objects in memory.
    Using _prefix to prevent Streamlit from trying to hash DataFrames.
    """
    models = {}

    # Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(_X_train, _y_train)
    models["Logistic Regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(_X_train, _y_train)
    models["Random Forest"] = rf

    # XGBoost / Gradient Boosting
    if XGB_AVAILABLE:
        scale_pw = (_y_train == 0).sum() / (_y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            scale_pos_weight=scale_pw, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1
        )
        xgb.fit(_X_train, _y_train)
        models["XGBoost"] = xgb
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        )
        gb.fit(_X_train, _y_train)
        models["Gradient Boosting"] = gb

    # SVM
    svc = SVC(C=1.0, kernel="rbf", class_weight="balanced",
              probability=True, random_state=42)
    svc.fit(_X_train, _y_train)
    models["SVM (RBF)"] = svc

    # LightGBM
    if LGB_AVAILABLE:
        scale_pw = (_y_train == 0).sum() / (_y_train == 1).sum()
        lgb = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=scale_pw, subsample=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb.fit(_X_train, _y_train)
        models["LightGBM"] = lgb

    return models


@st.cache_data(show_spinner=False)
def compute_metrics(_models_dict, _X_test, _y_test):
    """Evaluate every trained model and return a tidy results DataFrame."""
    rows = []
    cms  = {}
    for name, model in _models_dict.items():
        y_pred = model.predict(_X_test)
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(_y_test, y_pred)  * 100, 1),
            "Precision": round(precision_score(_y_test, y_pred, zero_division=0) * 100, 1),
            "Recall ★":  round(recall_score(_y_test, y_pred, zero_division=0)    * 100, 1),
            "F1-Score":  round(f1_score(_y_test, y_pred, zero_division=0)        * 100, 1),
        })
        cms[name] = confusion_matrix(_y_test, y_pred)
    return pd.DataFrame(rows), cms


# =============================================================================
#  PLOTLY HELPERS — consistent dark theme
# =============================================================================

def _dark_layout(**kwargs) -> dict:
    """
    Base Plotly layout tokens for the dark theme.
    NOTE: 'margin' is intentionally excluded here so each chart can
    set its own without triggering a 'multiple values' TypeError.
    """
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,39,0.6)",
        font=dict(family="Inter, sans-serif", color="#C4C7D8", size=12),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED)),
    )
    base.update(kwargs)
    return base


def plot_churn_dist(df: pd.DataFrame) -> go.Figure:
    """Donut chart — overall churn split."""
    counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=["No Churn", "Churn"],
        values=[counts.get("No", 0), counts.get("Yes", 0)],
        hole=0.62,
        marker=dict(colors=[BLUE, CORAL], line=dict(color="#0F1117", width=3)),
        textinfo="percent+label",
        textfont=dict(size=13, color="#E4E6F0"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        pull=[0, 0.04],
    ))
    fig.update_layout(
        title=dict(text="Churn Distribution", font=dict(size=14, color="#E4E6F0"), x=0.02),
        showlegend=False,
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=[dict(
            text=f"<b>{counts.get('Yes',0)/len(df)*100:.1f}%</b><br><span style='font-size:11px;color:{MUTED}'>Churn Rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=CORAL),
            xanchor="center",
        )],
    )
    return fig


def plot_contract_churn(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar — churn rate by contract type."""
    grp = (
        df.groupby(["Contract", "Churn"]).size().reset_index(name="n")
    )
    totals = grp.groupby("Contract")["n"].transform("sum")
    grp["pct"] = grp["n"] / totals * 100
    data = grp[grp["Churn"] == "Yes"].sort_values("pct", ascending=True)

    fig = go.Figure(go.Bar(
        x=data["pct"], y=data["Contract"],
        orientation="h",
        marker=dict(
            color=data["pct"],
            colorscale=[[0, BLUE], [0.5, AMBER], [1, CORAL]],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in data["pct"]],
        textposition="outside",
        textfont=dict(color="#E4E6F0", size=12),
        hovertemplate="<b>%{y}</b><br>Churn rate: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Churn Rate by Contract Type", font=dict(size=14, color="#E4E6F0"), x=0.02),
        xaxis=dict(range=[0, data["pct"].max() * 1.3], ticksuffix="%", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#C4C7D8", size=12)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        margin=dict(l=130, r=60, t=50, b=30),
    )
    return fig


def plot_internet_churn(df: pd.DataFrame) -> go.Figure:
    """Bar chart — churn rate by internet service type."""
    grp = (
        df.groupby(["InternetService", "Churn"]).size().reset_index(name="n")
    )
    totals = grp.groupby("InternetService")["n"].transform("sum")
    grp["pct"] = grp["n"] / totals * 100
    data = grp[grp["Churn"] == "Yes"].sort_values("pct", ascending=False)

    colors_map = {"Fiber optic": CORAL, "DSL": AMBER, "No": BLUE}
    bar_colors = [colors_map.get(x, BLUE) for x in data["InternetService"]]

    fig = go.Figure(go.Bar(
        x=data["InternetService"], y=data["pct"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in data["pct"]],
        textposition="outside",
        textfont=dict(color="#E4E6F0", size=12),
        hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Churn Rate by Internet Service", font=dict(size=14, color="#E4E6F0"), x=0.02),
        yaxis=dict(range=[0, data["pct"].max() * 1.25], ticksuffix="%", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#C4C7D8", size=12)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_tenure_density(df: pd.DataFrame) -> go.Figure:
    """Overlapping histograms — tenure by churn status."""
    fig = go.Figure()
    for label, color in [("No", BLUE), ("Yes", CORAL)]:
        sub = df[df["Churn"] == label]["tenure"]
        fig.add_trace(go.Histogram(
            x=sub, name="No Churn" if label == "No" else "Churned",
            marker_color=color, opacity=0.65,
            nbinsx=35,
            hovertemplate="Tenure: %{x}<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Tenure Distribution by Churn", font=dict(size=14, color="#E4E6F0"), x=0.02),
        xaxis=dict(title="Tenure (months)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Count", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor="rgba(26,29,39,0.8)", bordercolor=BORDER, borderwidth=1,
                    font=dict(color="#C4C7D8")),
    )
    return fig


def plot_monthly_charges(df: pd.DataFrame) -> go.Figure:
    """Overlapping histograms — monthly charges by churn status."""
    fig = go.Figure()
    for label, color in [("No", BLUE), ("Yes", CORAL)]:
        sub = df[df["Churn"] == label]["MonthlyCharges"]
        fig.add_trace(go.Histogram(
            x=sub, name="No Churn" if label == "No" else "Churned",
            marker_color=color, opacity=0.65,
            nbinsx=30,
            hovertemplate="Charge: $%{x:.0f}<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Monthly Charges Distribution", font=dict(size=14, color="#E4E6F0"), x=0.02),
        xaxis=dict(title="Monthly Charges ($)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Count", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor="rgba(26,29,39,0.8)", bordercolor=BORDER, borderwidth=1,
                    font=dict(color="#C4C7D8")),
    )
    return fig


def plot_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter — tenure vs monthly charges coloured by churn."""
    fig = go.Figure()
    for label, color, name in [("No", BLUE, "No Churn"), ("Yes", CORAL, "Churned")]:
        sub = df[df["Churn"] == label]
        fig.add_trace(go.Scatter(
            x=sub["tenure"], y=sub["MonthlyCharges"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.35),
            hovertemplate="Tenure: %{x}m<br>Charge: $%{y:.0f}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Tenure vs Monthly Charges", font=dict(size=14, color="#E4E6F0"), x=0.02),
        xaxis=dict(title="Tenure (months)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Monthly Charges ($)", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor="rgba(26,29,39,0.8)", bordercolor=BORDER, borderwidth=1,
                    font=dict(color="#C4C7D8")),
    )
    return fig


def plot_cat_churn_rate(df: pd.DataFrame, col: str) -> go.Figure:
    """Grouped bar — churn rate breakdown for a single categorical feature."""
    tmp = df.copy()
    tmp[col] = tmp[col].astype(str)
    grp = tmp.groupby([col, "Churn"]).size().reset_index(name="n")
    grp["total"] = grp.groupby(col)["n"].transform("sum")
    grp["pct"] = grp["n"] / grp["total"] * 100
    pivot = grp.pivot(index=col, columns="Churn", values="pct").fillna(0)
    pivot = pivot.sort_values("Yes", ascending=False)

    fig = go.Figure()
    for label, color, name in [("No", BLUE, "No Churn"), ("Yes", CORAL, "Churned")]:
        if label in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index, y=pivot[label],
                name=name,
                marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}%<extra></extra>",
            ))
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Churn Rate by {col}", font=dict(size=13, color="#E4E6F0"), x=0.02),
        xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED, size=10)),
        yaxis=dict(ticksuffix="%", gridcolor=BORDER, tickfont=dict(color=MUTED), range=[0, 115]),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor="rgba(26,29,39,0.8)", bordercolor=BORDER, borderwidth=1,
                    font=dict(color="#C4C7D8", size=10)),
        margin=dict(l=40, r=10, t=45, b=60),
    )
    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar — model × metric matrix."""
    metrics = ["Accuracy", "Precision", "Recall ★", "F1-Score"]
    colors  = [BLUE, GREEN, CORAL, AMBER]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df["Model"],
            y=results_df[metric],
            marker_color=color,
            text=[f"{v:.1f}" for v in results_df[metric]],
            textposition="outside",
            textfont=dict(size=9, color="#E4E6F0"),
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Model Comparison — All Metrics", font=dict(size=14, color="#E4E6F0"), x=0.02),
        xaxis=dict(tickangle=-15, gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#C4C7D8")),
        yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor=BORDER, tickfont=dict(color=MUTED)),
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        legend=dict(bgcolor="rgba(26,29,39,0.8)", bordercolor=BORDER, borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#C4C7D8")),
        margin=dict(l=40, r=20, t=70, b=60),
    )
    # Reference line at 50%
    fig.add_hline(y=50, line_dash="dot", line_color=MUTED, opacity=0.4)
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    """Annotated confusion matrix heatmap for a single model."""
    labels    = ["No Churn", "Churn"]
    cell_text = [[str(cm[i][j]) for j in range(2)] for i in range(2)]
    ann_labels= [["TN", "FP"], ["FN ★", "TP"]]

    z = cm.tolist()
    # Custom blue-scale colorscale
    colorscale = [
        [0.0, "#0F1117"],
        [0.3, "#1A3A5C"],
        [0.7, "#2B5F9E"],
        [1.0, "#4C9BE8"],
    ]

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"<b>{cell_text[i][j]}</b>",
                showarrow=False,
                font=dict(size=18, color="#E4E6F0"),
                xref="x", yref="y",
            ))
            color = CORAL if ann_labels[i][j] == "FN ★" else MUTED
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=ann_labels[i][j],
                showarrow=False,
                font=dict(size=10, color=color),
                xref="x", yref="y",
                yshift=-18,
            ))

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=model_name, font=dict(size=13, color="#E4E6F0"), x=0.04),
        xaxis=dict(title="Predicted", tickfont=dict(color="#C4C7D8"), gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Actual",    tickfont=dict(color="#C4C7D8"), gridcolor="rgba(0,0,0,0)"),
        annotations=annotations,
        **{k: v for k, v in _dark_layout().items() if k not in ("xaxis", "yaxis")},
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def plot_gauge(probability: float) -> go.Figure:
    """Speedometer gauge for churn probability."""
    if probability < 0.35:
        bar_color, label = GREEN, "LOW RISK"
    elif probability < 0.65:
        bar_color, label = AMBER, "MODERATE RISK"
    else:
        bar_color, label = CORAL, "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=36, color="#E4E6F0")),
        title=dict(text=f"<b>{label}</b>", font=dict(size=15, color=bar_color)),
        delta=dict(reference=26.5, suffix="%", font=dict(size=12),
                   increasing=dict(color=CORAL), decreasing=dict(color=GREEN)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED,
                      tickfont=dict(color=MUTED), nticks=6),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor="rgba(26,29,39,0.8)",
            borderwidth=0,
            steps=[
                dict(range=[0,  35],  color="rgba(82,199,126,0.12)"),
                dict(range=[35, 65],  color="rgba(245,197,66,0.12)"),
                dict(range=[65, 100], color="rgba(232,99,76,0.12)"),
            ],
            threshold=dict(
                line=dict(color=MUTED, width=2),
                thickness=0.8, value=26.5,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C4C7D8"),
        margin=dict(l=30, r=30, t=30, b=20),
        height=260,
    )
    return fig


# =============================================================================
#  PAGE HEADER HELPER
# =============================================================================

def page_header(icon: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <p class="page-title">{icon} {title}</p>
                <p class="page-sub">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
#  MAIN APP
# =============================================================================

def main():

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        df_raw = load_raw_data()

    # ── Build feature matrix (used by both the model tab and predictor) ──────
    with st.spinner("Engineering features…"):
        X_train, X_test, y_train, y_test, feat_cols, scaler = build_feature_matrix(df_raw)

    # ── Top-of-page brand bar ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;
                    border-bottom:2px solid #2A2D3E;padding-bottom:16px;margin-bottom:8px;">
            <span style="font-size:2rem;">📡</span>
            <div>
                <p style="margin:0;font-size:1.55rem;font-weight:800;
                           background:linear-gradient(90deg,#4C9BE8,#7B6CF6);
                           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    Telco Churn Intelligence
                </p>
                <p style="margin:2px 0 0 0;font-size:0.82rem;color:#8B8FA8;">
                    End-to-end ML pipeline · EDA → Feature Engineering → Model Selection
                    &nbsp;|&nbsp;
                    <span style="color:#4C9BE8;">■</span> No Churn &nbsp;
                    <span style="color:#E8634C;">■</span> Churn
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_eda, tab_model, tab_pred = st.tabs([
        "📊  Exploratory Data Analysis",
        "🤖  Model Performance Analytics",
        "⚡  Live Churn Predictor",
    ])


    # =========================================================================
    #  TAB 1 — EDA
    # =========================================================================
    with tab_eda:
        page_header("📊", "Exploratory Data Analysis",
                    "Understanding the data landscape before modelling")

        # ── KPI Row ───────────────────────────────────────────────────────────
        total     = len(df_raw)
        churn_n   = (df_raw["Churn"] == "Yes").sum()
        churn_pct = churn_n / total * 100
        avg_ten   = df_raw["tenure"].mean()
        avg_mc    = df_raw["MonthlyCharges"].mean()
        avg_tc    = df_raw["TotalCharges"].mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Customers",    f"{total:,}")
        k2.metric("Churned Customers",  f"{churn_n:,}", delta=f"{churn_pct:.1f}% rate", delta_color="inverse")
        k3.metric("Avg Tenure",         f"{avg_ten:.1f} mo")
        k4.metric("Avg Monthly Charges",f"${avg_mc:.2f}")
        k5.metric("Avg Total Charges",  f"${avg_tc:,.0f}")

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Row 1 charts ──────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Core Churn Drivers</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.3, 1.3])
        with c1:
            st.plotly_chart(plot_churn_dist(df_raw), use_container_width=True)
        with c2:
            st.plotly_chart(plot_contract_churn(df_raw), use_container_width=True)
        with c3:
            st.plotly_chart(plot_internet_churn(df_raw), use_container_width=True)

        # ── Insight callouts ──────────────────────────────────────────────────
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown("""
            <div class="insight-card">
                <b style="color:#4C9BE8;">Class Imbalance</b><br>
                73.5% of customers did not churn vs 26.5% who did — a 2.8:1 ratio.
                Models must account for this via class weighting or threshold tuning
                to avoid biasing predictions toward the majority class.
            </div>""", unsafe_allow_html=True)
        with i2:
            st.markdown("""
            <div class="insight-card">
                <b style="color:#E8634C;">Month-to-Month Contracts</b><br>
                Customers on month-to-month contracts churn at <b>42.7%</b> — over
                15× the rate of two-year contract holders (2.8%). Contract type is
                the single strongest predictor of churn in this dataset.
            </div>""", unsafe_allow_html=True)
        with i3:
            st.markdown("""
            <div class="insight-card">
                <b style="color:#E8634C;">Fiber Optic Risk</b><br>
                Fiber optic internet subscribers churn at <b>41.9%</b>, more than
                twice the DSL rate. Higher monthly charges associated with fiber
                service likely drive dissatisfaction and price sensitivity.
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Row 2 — Numeric distributions ────────────────────────────────────
        st.markdown("<div class='section-header'>Numeric Feature Distributions</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.plotly_chart(plot_tenure_density(df_raw), use_container_width=True)
        with d2:
            st.plotly_chart(plot_monthly_charges(df_raw), use_container_width=True)
        with d3:
            st.plotly_chart(plot_scatter(df_raw), use_container_width=True)

        st.markdown("""
        <div class="insight-card">
            <b style="color:#4C9BE8;">Key Pattern</b> — Churned customers (coral) are heavily
            concentrated in the <b>0–10 month tenure window</b>, confirming that early-stage
            customers are most at risk. The scatter plot also reveals a clear high-charge,
            short-tenure cluster of churners — suggesting that customers who sign up for expensive
            plans but leave quickly represent a critical intervention opportunity.
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Row 3 — Categorical deep-dive ────────────────────────────────────
        st.markdown("<div class='section-header'>Categorical Feature Deep-Dive</div>", unsafe_allow_html=True)
        cat_cols = [
            "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "PaperlessBilling", "PaymentMethod",
        ]
        selected_cats = st.multiselect(
            "Select features to visualise:",
            options=cat_cols,
            default=["OnlineSecurity", "TechSupport", "PaymentMethod", "SeniorCitizen"],
        )
        if selected_cats:
            n_cols = min(2, len(selected_cats))
            cols   = st.columns(n_cols)
            for idx, col_name in enumerate(selected_cats):
                with cols[idx % n_cols]:
                    st.plotly_chart(
                        plot_cat_churn_rate(df_raw, col_name),
                        use_container_width=True,
                    )
        else:
            st.info("Select at least one categorical feature above to visualise.")


    # =========================================================================
    #  TAB 2 — MODEL ANALYTICS
    # =========================================================================
    with tab_model:
        page_header("🤖", "Model Performance Analytics",
                    "Comparing five ML models with emphasis on Recall for the Churn class")

        # ── Train + evaluate models ───────────────────────────────────────────
        with st.spinner("Training models (cached after first run)…"):
            models_dict = train_all_models(X_train, y_train)

        with st.spinner("Evaluating…"):
            results_df, cms = compute_metrics(models_dict, X_test, y_test)

        # ── Why Recall? Callout ───────────────────────────────────────────────
        st.markdown("""
        <div class="insight-card" style="border-left-color:#E8634C;margin-bottom:20px;">
            <b style="font-size:1rem;color:#E8634C;">★ Why Recall is our Target Metric</b><br><br>
            In churn prediction, a <b>False Negative</b> (predicting a customer stays when they
            actually leave) is far more costly than a False Positive. A missed churner means lost
            revenue, failed retention, and wasted CAC — whereas a false alarm costs only a small
            retention incentive spend.<br><br>
            <b>Recall</b> = TP / (TP + FN) measures how many <i>actual churners</i> we successfully
            identify. Accuracy is misleading here: a model that predicts "No Churn" for everyone
            achieves 73.5% accuracy — yet catches zero churners. We prioritise Recall on the Churn
            class while monitoring Precision and F1 to ensure we're not flooding retention teams
            with false alerts.
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics table ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Performance Summary</div>", unsafe_allow_html=True)

        # Highlight best-in-column
        def _style_table(df: pd.DataFrame):
            numeric_cols = ["Accuracy", "Precision", "Recall ★", "F1-Score"]
            styled = df.style.format({c: "{:.1f}%" for c in numeric_cols})

            for col in numeric_cols:
                max_val = df[col].max()
                styled = styled.apply(
                    lambda s, c=col, mv=max_val: [
                        f"background-color: rgba(76,155,232,0.18); color: #4C9BE8; font-weight:700"
                        if (v == mv and c != "Recall ★") else
                        f"background-color: rgba(232,99,76,0.18); color: #E8634C; font-weight:700"
                        if (v == mv and c == "Recall ★") else ""
                        for v in s
                    ],
                    subset=[col],
                )
            return styled

        st.dataframe(
            _style_table(results_df),
            use_container_width=True,
            hide_index=True,
        )

        # ── Best model callouts ────────────────────────────────────────────────
        best_recall_row = results_df.loc[results_df["Recall ★"].idxmax()]
        best_f1_row     = results_df.loc[results_df["F1-Score"].idxmax()]
        best_acc_row    = results_df.loc[results_df["Accuracy"].idxmax()]

        b1, b2, b3 = st.columns(3)
        b1.metric("🏆 Best Recall",    f"{best_recall_row['Recall ★']}%",  best_recall_row["Model"])
        b2.metric("⚖️ Best F1-Score",  f"{best_f1_row['F1-Score']}%",      best_f1_row["Model"])
        b3.metric("🎯 Best Accuracy",  f"{best_acc_row['Accuracy']}%",     best_acc_row["Model"])

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Model comparison bar chart ─────────────────────────────────────────
        st.markdown("<div class='section-header'>Visual Comparison</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_model_comparison(results_df), use_container_width=True)

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Confusion matrices ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Confusion Matrices — Test Set</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card" style="margin-bottom:16px;">
            <b style="color:#E8634C;">FN ★ (False Negatives)</b> are the critical failure mode:
            actual churners we failed to catch. Compare the FN cell across models to find the one
            that minimises missed churners. Lower FN = higher Recall.
        </div>
        """, unsafe_allow_html=True)

        # Display confusion matrices in a responsive 2-column grid
        model_names = list(cms.keys())
        for row_start in range(0, len(model_names), 2):
            cols = st.columns(2)
            for ci, name in enumerate(model_names[row_start:row_start+2]):
                with cols[ci]:
                    st.plotly_chart(
                        plot_confusion_matrix(cms[name], name),
                        use_container_width=True,
                    )


    # =========================================================================
    #  TAB 3 — LIVE PREDICTOR
    # =========================================================================
    with tab_pred:
        page_header("⚡", "Real-Time Churn Predictor",
                    "Enter customer attributes to instantly score churn probability")

        # ── Select model for prediction ───────────────────────────────────────
        available_model_names = list(models_dict.keys())

        # Re-train with probability=True for all models if needed (SVM already has it)
        @st.cache_resource(show_spinner=False)
        def get_prob_model(_X_train, _y_train, model_name):
            """Return a probability-capable model for the predictor."""
            if model_name == "Logistic Regression":
                m = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            elif model_name == "Random Forest":
                m = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1)
            elif "XGBoost" in model_name and XGB_AVAILABLE:
                scale_pw = (_y_train == 0).sum() / (_y_train == 1).sum()
                m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                  scale_pos_weight=scale_pw, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss",
                                  random_state=42, verbosity=0, n_jobs=-1)
            elif "Gradient Boosting" in model_name:
                from sklearn.ensemble import GradientBoostingClassifier
                m = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, subsample=0.8, random_state=42)
            elif "SVM" in model_name:
                m = SVC(C=1.0, kernel="rbf", class_weight="balanced",
                        probability=True, random_state=42)
            elif "LightGBM" in model_name and LGB_AVAILABLE:
                scale_pw = (_y_train == 0).sum() / (_y_train == 1).sum()
                m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   scale_pos_weight=scale_pw, subsample=0.8,
                                   random_state=42, n_jobs=-1, verbose=-1)
            else:
                m = LogisticRegression(max_iter=1000, random_state=42)
            m.fit(_X_train, _y_train)
            return m

        # ── Input form ────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Customer Attributes</div>", unsafe_allow_html=True)

        col_left, col_mid, col_right = st.columns([1.1, 1.1, 1])

        with col_left:
            st.markdown("**📋 Account & Demographics**")
            gender         = st.selectbox("Gender",          ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen",  ["No", "Yes"])
            partner        = st.selectbox("Partner",         ["Yes", "No"])
            dependents     = st.selectbox("Dependents",      ["No", "Yes"])
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            contract       = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])

        with col_mid:
            st.markdown("**🌐 Services Subscribed**")
            phone_service  = st.selectbox("Phone Service",   ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
            internet       = st.selectbox("Internet Service",["Fiber optic", "DSL", "No"])
            online_sec     = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_bkp     = st.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
            device_prot    = st.selectbox("Device Protection",["No", "Yes", "No internet service"])

        with col_right:
            st.markdown("**🛠️ Support & Billing**")
            tech_support   = st.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
            streaming_tv   = st.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
            streaming_mov  = st.selectbox("Streaming Movies",["No", "Yes", "No internet service"])
            paperless      = st.selectbox("Paperless Billing",["Yes", "No"])
            payment        = st.selectbox("Payment Method",  [
                                "Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"
                            ])
            monthly_chg    = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_chg      = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                             float(round(monthly_chg * tenure, 2)), 1.0)

        st.markdown("<hr class='h-line'>", unsafe_allow_html=True)

        # ── Model selector + predict button ───────────────────────────────────
        mc1, mc2 = st.columns([2, 1])
        with mc1:
            selected_model_name = st.selectbox(
                "Choose Model for Prediction:",
                options=available_model_names,
                index=min(1, len(available_model_names)-1),  # default to Random Forest
                help="All models were trained on an 80/20 stratified split with StandardScaler.",
            )
        with mc2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_clicked = st.button("🔍  Predict Churn Risk", type="primary", use_container_width=True)

        # ── Prediction logic ──────────────────────────────────────────────────
        if predict_clicked:
            # Build a one-row DataFrame that mirrors the OHE feature space
            raw_input = {
                "gender":          1 if gender == "Male" else 0,
                "SeniorCitizen":   1 if senior == "Yes" else 0,
                "Partner":         1 if partner == "Yes" else 0,
                "Dependents":      1 if dependents == "Yes" else 0,
                "tenure":          tenure,
                "PhoneService":    1 if phone_service == "Yes" else 0,
                "PaperlessBilling":1 if paperless == "Yes" else 0,
                "MonthlyCharges":  monthly_chg,
                "TotalCharges":    total_chg,
            }

            # One-hot encode the multi-class columns to match training feature space
            ohe_cols = {
                "MultipleLines":    ["No", "No phone service", "Yes"],
                "InternetService":  ["DSL", "Fiber optic", "No"],
                "OnlineSecurity":   ["No", "No internet service", "Yes"],
                "OnlineBackup":     ["No", "No internet service", "Yes"],
                "DeviceProtection": ["No", "No internet service", "Yes"],
                "TechSupport":      ["No", "No internet service", "Yes"],
                "StreamingTV":      ["No", "No internet service", "Yes"],
                "StreamingMovies":  ["No", "No internet service", "Yes"],
                "Contract":         ["Month-to-month", "One year", "Two year"],
                "PaymentMethod":    [
                    "Bank transfer (automatic)", "Credit card (automatic)",
                    "Electronic check", "Mailed check"
                ],
            }
            values_map = {
                "MultipleLines":    multiple_lines,
                "InternetService":  internet,
                "OnlineSecurity":   online_sec,
                "OnlineBackup":     online_bkp,
                "DeviceProtection": device_prot,
                "TechSupport":      tech_support,
                "StreamingTV":      streaming_tv,
                "StreamingMovies":  streaming_mov,
                "Contract":         contract,
                "PaymentMethod":    payment,
            }
            for col_name, categories in ohe_cols.items():
                chosen = values_map[col_name]
                for cat in categories:
                    raw_input[f"{col_name}_{cat}"] = 1 if chosen == cat else 0

            # Align to exact training feature columns
            input_df = pd.DataFrame([raw_input])
            for c in feat_cols:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_df = input_df[feat_cols]  # enforce column order

            # Scale the numeric cols with the fitted scaler
            scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])

            # Load probability-capable model
            prob_model = get_prob_model(X_train, y_train, selected_model_name)

            # Predict
            prob = prob_model.predict_proba(input_df)[0][1]
            pred = int(prob >= 0.5)

            st.markdown("<hr class='h-line'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

            # ── Result layout ───────────────────────────────────────────────
            res_left, res_right = st.columns([1, 1.2])

            with res_left:
                st.plotly_chart(plot_gauge(prob), use_container_width=True)

                # Contextual benchmark
                st.markdown(f"""
                <p style="text-align:center;font-size:0.8rem;color:{MUTED};margin-top:-10px;">
                    Dataset average churn rate: <b style="color:#E4E6F0;">26.5%</b>
                    &nbsp;|&nbsp; Model: <b style="color:#4C9BE8;">{selected_model_name}</b>
                </p>
                """, unsafe_allow_html=True)

            with res_right:
                # ── Banner ────────────────────────────────────────────────
                if prob < 0.35:
                    banner_class = "churn-low"
                    icon_txt     = "✅"
                    headline     = f"Low Churn Risk — {prob*100:.1f}%"
                    recommendation = (
                        "This customer shows strong retention signals. Continue standard engagement. "
                        "Consider a loyalty reward at next contract renewal."
                    )
                elif prob < 0.65:
                    banner_class = "churn-mid"
                    icon_txt     = "⚠️"
                    headline     = f"Moderate Churn Risk — {prob*100:.1f}%"
                    recommendation = (
                        "Elevated risk detected. Proactively reach out with a personalised offer — "
                        "a contract upgrade discount or added service bundle may help retain this customer."
                    )
                else:
                    banner_class = "churn-high"
                    icon_txt     = "🚨"
                    headline     = f"High Churn Risk — {prob*100:.1f}%"
                    recommendation = (
                        "Critical retention alert. This customer is likely to leave. Escalate to the "
                        "retention team immediately. Offer a significant discount, contract switch, or "
                        "personalised support call within 48 hours."
                    )

                st.markdown(f"""
                <div class="{banner_class}">
                    <p class="banner-title">{icon_txt} {headline}</p>
                    <p class="banner-sub">{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Key risk factors summary ───────────────────────────────
                st.markdown("**Key Risk Factors for this Customer:**")
                risk_items = []
                if contract == "Month-to-month":
                    risk_items.append(("🔴", "Month-to-month contract", "Highest-risk contract type (42.7% churn rate)"))
                if internet == "Fiber optic":
                    risk_items.append(("🔴", "Fiber optic internet", "Highest churn internet tier (41.9%)"))
                if tenure < 12:
                    risk_items.append(("🟠", f"Short tenure ({tenure}m)", "Customers most at risk in first year"))
                if monthly_chg > 80:
                    risk_items.append(("🟠", f"High monthly charge (${monthly_chg:.0f})", "Above-average pricing increases churn likelihood"))
                if tech_support == "No" and internet != "No":
                    risk_items.append(("🟡", "No Tech Support", "Lack of support increases frustration-driven churn"))
                if online_sec == "No" and internet != "No":
                    risk_items.append(("🟡", "No Online Security", "Unprotected customers churn more frequently"))
                if payment == "Electronic check":
                    risk_items.append(("🟡", "Electronic check payment", "Highest churn payment method"))
                if senior == "Yes":
                    risk_items.append(("🟡", "Senior citizen", "Elevated churn rate among senior segment"))

                if risk_items:
                    for dot, title, desc in risk_items:
                        st.markdown(f"""
                        <div style="display:flex;gap:10px;align-items:flex-start;
                                    margin-bottom:8px;background:#1A1D27;
                                    border:1px solid #2A2D3E;border-radius:8px;
                                    padding:10px 14px;font-size:0.83rem;">
                            <span style="font-size:1rem;flex-shrink:0;">{dot}</span>
                            <div>
                                <b style="color:#E4E6F0;">{title}</b>
                                <br><span style="color:{MUTED};">{desc}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="insight-card" style="border-left-color:#52C77E;">
                        ✅ No major risk factors detected for this customer profile.
                    </div>""", unsafe_allow_html=True)

                # ── Probability breakdown ──────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                prob_cols = st.columns(2)
                with prob_cols[0]:
                    st.metric("P(No Churn)", f"{(1-prob)*100:.1f}%")
                with prob_cols[1]:
                    st.metric("P(Churn)", f"{prob*100:.1f}%", delta=f"{'↑' if prob > 0.265 else '↓'} vs 26.5% avg",
                              delta_color="inverse")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
