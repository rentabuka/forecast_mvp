import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Unilever Demand Forecast Platform",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 28px; font-weight: 700; margin-bottom: 8px;}
    .subtle {color: #64748B; font-size: 13px;}
    .card {padding: 14px 16px; border: 1px solid #E2E8F0; border-radius: 10px; background: #F8FAFC;}
    .warning-card {padding: 14px 16px; border: 1px solid #FCD34D; border-radius: 10px; background: #FFFBEB;}
    .success-card {padding: 14px 16px; border: 1px solid #86EFAC; border-radius: 10px; background: #F0FDF4;}
    .info-card {padding: 14px 16px; border: 1px solid #BFDBFE; border-radius: 10px; background: #EFF6FF;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA CONTRACT
# ============================================================
# The current Unilever extract uses 52-week measure names.
# 26-week names remain supported as a fallback so older extracts
# do not break the application.
MEASURE_SETS = {
    "52-week": {
        "value": "52 Weeks CY Value",
        "price": "52 Weeks CY Ave Price Quantity",
        "promo": "52 Weeks CY Ave RSP On Promo",
    },
    "26-week": {
        "value": "26 Weeks CY Value",
        "price": "26 Weeks CY Ave Price Quantity",
        "promo": "26 Weeks CY Ave RSP On Promo",
    },
}

BASE_COLUMNS = [
    "Full Date",
    "Brand",
    "Category",
    "Product",
    "ProductsID",
    "Subcategory",
]


# ============================================================
# DATA HELPERS
# ============================================================
def clean_number(series):
    return pd.to_numeric(series, errors="coerce")


def detect_measure_set(columns):
    columns = set(columns)

    for name in ("52-week", "26-week"):
        measures = MEASURE_SETS[name]
        if all(measures[k] in columns for k in ("value", "price", "promo")):
            return name, measures

    return None, None


def validate_and_prepare(raw_df):
    """Validate the weekly Unilever extract and create clean fields."""
    errors = []
    warnings_list = []

    missing_base = [c for c in BASE_COLUMNS if c not in raw_df.columns]
    if missing_base:
        errors.append("Missing required columns: " + ", ".join(missing_base))
        return False, errors, warnings_list, None, None

    measure_window, measures = detect_measure_set(raw_df.columns)
    if measures is None:
        errors.append(
            "Could not find the required sales measure columns. Expected either the 52-week set "
            "(52 Weeks CY Value / 52 Weeks CY Ave Price Quantity / 52 Weeks CY Ave RSP On Promo) "
            "or the older 26-week set."
        )
        return False, errors, warnings_list, None, None

    df = raw_df.copy()

    # Date
    df["date_key"] = pd.to_datetime(df["Full Date"], errors="coerce")
    bad_dates = int(df["date_key"].isna().sum())
    if bad_dates:
        warnings_list.append(f"{bad_dates:,} rows have invalid dates and were removed.")
        df = df.dropna(subset=["date_key"]).copy()

    if df.empty:
        return False, ["No valid dated rows remain after date validation."], warnings_list, None, None

    # Sales value and prices
    df["Sales Value"] = clean_number(df[measures["value"]]).fillna(0).clip(lower=0)
    df["Ave RSP"] = clean_number(df[measures["price"]])
    df["Promo RSP"] = clean_number(df[measures["promo"]])

    # IMPORTANT:
    # Units are derived at row level from weekly value / weekly average price.
    # Do not round SKU units here. Round only for display.
    df["Sales Units"] = np.where(
        df["Ave RSP"] > 0,
        df["Sales Value"] / df["Ave RSP"],
        np.nan,
    )
    df["Sales Units"] = (
        pd.to_numeric(df["Sales Units"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )

    # Optional Numeric Distribution. Never invent a 85% default.
    if "Numeric Distribution" in df.columns:
        dist = clean_number(df["Numeric Distribution"])
        non_null = dist.dropna()
        if not non_null.empty and non_null.max() <= 1.0:
            dist = dist * 100
        df["Numeric Distribution"] = dist.clip(0, 100)
    else:
        df["Numeric Distribution"] = np.nan
        warnings_list.append(
            "Numeric Distribution is not in the extract. Distribution diagnostics will be unavailable."
        )

    for col in ["Category", "Subcategory", "Brand", "Product"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["ProductsID"] = df["ProductsID"].fillna("Unknown").astype(str)

    # Row-level promo depth for diagnostic use only.
    df["Promo Depth %"] = np.where(
        (df["Ave RSP"] > 0) & df["Promo RSP"].notna(),
        ((df["Ave RSP"] - df["Promo RSP"]) / df["Ave RSP"]) * 100,
        np.nan,
    )
    df["Promo Depth %"] = (
        df["Promo Depth %"].replace([np.inf, -np.inf], np.nan).clip(0, 100)
    )

    # Determine number of distinct weekly observations.
    unique_dates = pd.Series(df["date_key"].dropna().unique()).sort_values()
    n_dates = int(len(unique_dates))

    if n_dates < 12:
        warnings_list.append(
            f"Only {n_dates} unique weekly periods are available. Forecast accuracy may be unstable."
        )
    elif n_dates < 52:
        warnings_list.append(
            f"{n_dates} weekly periods are available. Actual YoY comparison requires 52 weeks."
        )
    elif n_dates == 52:
        warnings_list.append(
            "52 weeks are available. Actual YoY comparison is enabled; annual seasonality can be used as a "
            "same-period-last-year benchmark, but there is only one annual cycle, so learned seasonality is not "
            "treated as fully validated yet."
        )

    # Check weekly regularity.
    if n_dates > 1:
        gaps = (
            pd.Series(unique_dates).diff().dropna().dt.days.astype(float)
        )
        if not gaps.empty:
            abnormal = int((gaps != 7).sum())
            if abnormal:
                warnings_list.append(
                    f"{abnormal} gaps do not equal exactly 7 days. The platform uses the actual observation dates "
                    "instead of manufacturing missing weeks."
                )

    return (
        True,
        errors,
        warnings_list,
        df.sort_values("date_key").reset_index(drop=True),
        measure_window,
    )


# Cache preparation because the Unilever extract can be large.
@st.cache_data(show_spinner=False)
def prepare_csv(csv_bytes):
    raw = pd.read_csv(pd.io.common.BytesIO(csv_bytes), low_memory=False)
    return validate_and_prepare(raw)


# ============================================================
# WEEKLY AGGREGATION
# ============================================================
def aggregate_weekly(df):
    """
    The source is weekly. Group by the actual Full Date instead of
    resampling into artificial calendar buckets.
    """
    if df.empty:
        return pd.DataFrame()

    base = (
        df.groupby("date_key", as_index=False)
        .agg(
            Sales_Units=("Sales Units", "sum"),
            Sales_Value=("Sales Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    # Scope-level effective selling price.
    base["Effective_RSP"] = np.where(
        base["Sales_Units"] > 0,
        base["Sales_Value"] / base["Sales_Units"],
        np.nan,
    )

    # Volume-weighted promo RSP. This is a diagnostic only.
    promo_part = df.copy()
    promo_part["Promo_RSP"] = pd.to_numeric(promo_part["Promo RSP"], errors="coerce")
    promo_part = promo_part[(promo_part["Promo_RSP"] > 0) & (promo_part["Sales Units"] > 0)].copy()
    promo_part["Promo_Value_Proxy"] = promo_part["Promo_RSP"] * promo_part["Sales Units"]

    if promo_part.empty:
        base["Promo_RSP_Weighted"] = np.nan
    else:
        promo_week = (
            promo_part.groupby("date_key")
            .agg(
                Promo_Value_Proxy=("Promo_Value_Proxy", "sum"),
                Promo_Units=("Sales Units", "sum"),
            )
            .reset_index()
        )
        promo_week["Promo_RSP_Weighted"] = np.where(
            promo_week["Promo_Units"] > 0,
            promo_week["Promo_Value_Proxy"] / promo_week["Promo_Units"],
            np.nan,
        )
        base = base.merge(
            promo_week[["date_key", "Promo_RSP_Weighted"]],
            on="date_key",
            how="left",
        )

    base["Promo_Depth_%"] = np.where(
        (base["Effective_RSP"] > 0) & base["Promo_RSP_Weighted"].notna(),
        ((base["Effective_RSP"] - base["Promo_RSP_Weighted"]) / base["Effective_RSP"]) * 100,
        np.nan,
    )
    base["Promo_Depth_%"] = base["Promo_Depth_%"].replace([np.inf, -np.inf], np.nan).clip(0, 100)

    return base


# ============================================================
# MODEL HELPERS
# ============================================================
def is_constant_or_empty(y):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return True
    if np.allclose(y, 0):
        return True
    return np.nanstd(y) < 1e-10


def forecast_naive(y, horizon):
    y = np.asarray(y, dtype=float)
    return np.repeat(max(0.0, float(y[-1])), horizon)


def forecast_ma4(y, horizon):
    y = np.asarray(y, dtype=float)
    n = min(4, len(y))
    return np.repeat(max(0.0, float(np.mean(y[-n:]))), horizon)


def forecast_ma6(y, horizon):
    y = np.asarray(y, dtype=float)
    n = min(6, len(y))
    return np.repeat(max(0.0, float(np.mean(y[-n:]))), horizon)


def forecast_wma6(y, horizon):
    y = np.asarray(y, dtype=float)
    n = min(6, len(y))
    recent = y[-n:]
    weights = np.arange(1, n + 1, dtype=float)
    level = np.average(recent, weights=weights)
    return np.repeat(max(0.0, float(level)), horizon)


def forecast_median6(y, horizon):
    y = np.asarray(y, dtype=float)
    n = min(6, len(y))
    return np.repeat(max(0.0, float(np.median(y[-n:]))), horizon)


def forecast_ses(y, horizon):
    y = np.asarray(y, dtype=float)
    if is_constant_or_empty(y):
        return forecast_naive(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_log_ses(y, horizon):
    y = np.asarray(y, dtype=float)
    if is_constant_or_empty(y):
        return forecast_naive(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_y = np.log1p(np.maximum(y, 0))
        fit = SimpleExpSmoothing(log_y, initialization_method="estimated").fit(optimized=True)
    return np.maximum(0.0, np.expm1(np.asarray(fit.forecast(horizon), dtype=float)))


def forecast_damped_holt(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 5 or is_constant_or_empty(y):
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Holt(y, initialization_method="estimated", damped_trend=True).fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_theta(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 8 or is_constant_or_empty(y):
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThetaModel(y, period=1).fit()
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_seasonal_naive52(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 52:
        raise ValueError("Need at least 52 weekly observations.")
    # Next week corresponds to the observation 52 weeks ago.
    return np.asarray([max(0.0, y[-52 + i]) for i in range(horizon)], dtype=float)


def forecast_croston_sba(y, horizon, alpha=0.10):
    """Croston-SBA for genuinely intermittent SKU demand."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(horizon)

    non_zero = np.flatnonzero(y > 0)
    if len(non_zero) == 0:
        return np.zeros(horizon)

    first = int(non_zero[0])
    demand_est = float(y[first])
    interval_est = float(max(1, first + 1))
    last_event = first

    for t in range(first + 1, len(y)):
        if y[t] > 0:
            interval = float(max(1, t - last_event))
            demand_est += alpha * (y[t] - demand_est)
            interval_est += alpha * (interval - interval_est)
            last_event = t

    forecast = (1.0 - alpha / 2.0) * demand_est / max(interval_est, 1e-9)
    return np.repeat(max(0.0, forecast), horizon)


def get_model_registry(n_obs, intermittent=False, allow_seasonal_validation=False):
    models = {
        "Naive Last Week": forecast_naive,
        "4-Week Moving Average": forecast_ma4,
        "6-Week Moving Average": forecast_ma6,
        "Weighted 6-Week Average": forecast_wma6,
        "6-Week Median": forecast_median6,
        "Simple Exponential Smoothing": forecast_ses,
        "Log Exponential Smoothing": forecast_log_ses,
        "Damped Holt": forecast_damped_holt,
        "Theta": forecast_theta,
    }

    # With only 52 weeks, Seasonal Naive is a valid BENCHMARK but cannot
    # be fairly backtested with a 4-week horizon. It becomes eligible for
    # automatic selection once there are at least 56 weeks.
    if n_obs >= 56 and allow_seasonal_validation:
        models["Seasonal Naive (52 Weeks)"] = forecast_seasonal_naive52

    if intermittent:
        models["Croston-SBA"] = forecast_croston_sba

    return models


# ============================================================
# METRICS
# ============================================================
def wmape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = np.sum(np.abs(actual))
    if denominator <= 0:
        return np.nan
    return 100.0 * np.sum(np.abs(actual - pred)) / denominator


def smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual) + np.abs(pred)
    terms = np.where(denom > 0, 2.0 * np.abs(actual - pred) / denom, 0.0)
    return 100.0 * np.mean(terms)


# ============================================================
# BACKTESTING
# ============================================================
def backtest_setup(n_obs, horizon=4):
    """Choose enough history to make the rolling validation meaningful."""
    horizon = min(horizon, max(1, n_obs // 4))

    if n_obs >= 52:
        min_train = 26
    elif n_obs >= 36:
        min_train = 20
    elif n_obs >= 24:
        min_train = 16
    elif n_obs >= 16:
        min_train = 12
    else:
        min_train = max(6, n_obs - horizon - 1)

    n_origins = max(0, n_obs - horizon - min_train + 1)
    return horizon, min_train, n_origins


def backtest_model(y, model_name, model_func, horizon, min_train):
    y = np.asarray(y, dtype=float)
    rows = []

    for split in range(min_train, len(y) - horizon + 1):
        train = y[:split]
        actual = y[split : split + horizon]

        try:
            pred = np.asarray(model_func(train, horizon), dtype=float)
        except Exception:
            continue

        if len(pred) != horizon or not np.all(np.isfinite(pred)):
            continue

        pred = np.maximum(pred, 0.0)

        for h, (p, a) in enumerate(zip(pred, actual), start=1):
            rows.append(
                {
                    "Model": model_name,
                    "Horizon": h,
                    "Actual": float(a),
                    "Prediction": float(p),
                    "Error": float(p - a),
                    "RelError": float((p - a) / a) if a > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate_models_cached(values_tuple, horizon=4):
    y = np.asarray(values_tuple, dtype=float)
    intermittent = np.mean(y <= 0) >= 0.20
    actual_horizon, min_train, n_origins = backtest_setup(len(y), horizon)

    if n_origins <= 0:
        return pd.DataFrame(), pd.DataFrame(), None, actual_horizon, min_train, n_origins

    models = get_model_registry(
        len(y),
        intermittent=intermittent,
        allow_seasonal_validation=True,
    )

    score_rows = []
    detail_rows = []

    for name, func in models.items():
        details = backtest_model(y, name, func, actual_horizon, min_train)
        if details.empty:
            continue

        detail_rows.append(details)
        score_rows.append(
            {
                "Model": name,
                "WMAPE %": wmape(details["Actual"], details["Prediction"]),
                "sMAPE %": smape(details["Actual"], details["Prediction"]),
                "Bias %": 100.0 * details["Error"].sum() / max(details["Actual"].sum(), 1e-9),
                "Forecasts Tested": len(details),
            }
        )

    scores = pd.DataFrame(score_rows)
    details_all = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    if scores.empty:
        return scores, details_all, None, actual_horizon, min_train, n_origins

    scores["Abs Bias"] = scores["Bias %"].abs()
    scores = scores.sort_values(["WMAPE %", "Abs Bias"]).reset_index(drop=True)

    # With 52 weeks exactly, seasonal-naive is intentionally not in this table.
    champion = str(scores.iloc[0]["Model"])

    return scores, details_all, champion, actual_horizon, min_train, n_origins


# ============================================================
# FORECAST INTERVALS
# ============================================================
def build_prediction_intervals(champion_details, future_forecast):
    """Empirical forecast interval using rolling-origin relative errors."""
    f = np.asarray(future_forecast, dtype=float)
    lower = np.zeros(len(f))
    upper = np.zeros(len(f))

    global_errors = champion_details["RelError"].dropna().to_numpy()

    for i in range(len(f)):
        horizon_number = i + 1
        horizon_errors = champion_details.loc[
            champion_details["Horizon"] == horizon_number, "RelError"
        ].dropna().to_numpy()

        errors = horizon_errors if len(horizon_errors) >= 5 else global_errors

        if len(errors) >= 5:
            q10 = float(np.quantile(errors, 0.10))
            q90 = float(np.quantile(errors, 0.90))
        else:
            q10, q90 = -0.15, 0.15

        lower[i] = max(0.0, f[i] * (1.0 + q10))
        upper[i] = max(lower[i], f[i] * (1.0 + q90))

    return lower, upper


# ============================================================
# RUN ONE FORECAST
# ============================================================
def run_forecast(series, horizon=4):
    y = np.asarray(series, dtype=float)
    values_tuple = tuple(float(x) for x in y)

    scores, details, champion, bt_horizon, min_train, n_origins = evaluate_models_cached(
        values_tuple,
        horizon,
    )

    intermittent = np.mean(y <= 0) >= 0.20
    registry = get_model_registry(
        len(y),
        intermittent=intermittent,
        allow_seasonal_validation=True,
    )

    fallback_reason = None

    if champion is None or champion not in registry:
        champion = "Naive Last Week"
        model_func = forecast_naive
        fallback_reason = "Insufficient history or model fitting failures prevented model selection."
    else:
        model_func = registry[champion]

    try:
        future = np.asarray(model_func(y, horizon), dtype=float)
    except Exception:
        future = forecast_naive(y, horizon)
        champion = "Naive Last Week"
        fallback_reason = "Champion model failed during final fit; Naive Last Week was used as fallback."

    future = np.maximum(future, 0.0)

    champion_details = (
        details[details["Model"] == champion].copy()
        if not details.empty
        else pd.DataFrame()
    )

    if champion_details.empty:
        lower = future * 0.85
        upper = future * 1.15
        interval_method = "Indicative ±15% because insufficient backtest errors are available."
    else:
        lower, upper = build_prediction_intervals(champion_details, future)
        interval_method = "Empirical 10th–90th percentile of rolling backtest relative errors."

    # Seasonal-naive same-period-last-year benchmark is always shown once 52 weeks exist.
    seasonal_benchmark = None
    if len(y) >= 52:
        seasonal_benchmark = forecast_seasonal_naive52(y, horizon)

    return {
        "forecast": future,
        "lower": lower,
        "upper": upper,
        "champion": champion,
        "scores": scores,
        "details": details,
        "bt_horizon": bt_horizon,
        "min_train": min_train,
        "n_origins": n_origins,
        "interval_method": interval_method,
        "fallback_reason": fallback_reason,
        "seasonal_benchmark": seasonal_benchmark,
    }


# ============================================================
# YOY / SAME-PERIOD-LAST-YEAR
# ============================================================
def same_period_last_year(weekly, future_dates):
    """
    Match each future week to the actual observation 52 weeks earlier.
    We do not manufacture LY values.
    """
    if len(weekly) < 52:
        return None

    ly_rows = []

    for future_date in future_dates:
        target = future_date - timedelta(weeks=52)
        distance = (weekly["date_key"] - target).abs().dt.days
        idx = distance.idxmin()

        if int(distance.loc[idx]) > 7:
            return None

        ly_rows.append(
            {
                "Week": future_date,
                "LY Units": float(weekly.loc[idx, "Sales_Units"]),
                "LY Value": float(weekly.loc[idx, "Sales_Value"]),
                "LY Date": weekly.loc[idx, "date_key"],
            }
        )

    return pd.DataFrame(ly_rows)


# ============================================================
# DRIVER DIAGNOSTICS
# ============================================================
def recent_change(series, recent_n=4, prior_n=4):
    s = pd.Series(series).dropna()
    if len(s) < recent_n + prior_n:
        return np.nan

    recent = s.iloc[-recent_n:].mean()
    prior = s.iloc[-recent_n - prior_n : -recent_n].mean()

    if prior == 0:
        return np.nan

    return 100.0 * (recent / prior - 1.0)


def build_driver_summary(weekly):
    items = []

    unit_change = recent_change(weekly["Sales_Units"])
    price_change = recent_change(weekly["Effective_RSP"])
    promo_change = recent_change(weekly["Promo_Depth_%"])
    dist_change = recent_change(weekly["Distribution"])

    if np.isfinite(unit_change):
        items.append(
            (
                "Volume",
                f"Latest 4-week average volume is {unit_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    if np.isfinite(price_change):
        items.append(
            (
                "Price",
                f"Effective RSP changed {price_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    if np.isfinite(promo_change):
        items.append(
            (
                "Promotion",
                f"Observed promotional depth changed {promo_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    if np.isfinite(dist_change):
        items.append(
            (
                "Distribution",
                f"Numeric distribution changed {dist_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    return items


# ============================================================
# APP START
# ============================================================
st.markdown(
    '<div class="main-header">Unilever Demand & Forecast Platform</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Weekly demand forecasting with automatic model selection, rolling backtesting, same-period-last-year comparison, "
    "and commercial diagnostics."
)

with st.sidebar:
    st.header("📥 Data & Filters")
    uploaded_file = st.file_uploader("Upload weekly sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload the Unilever weekly sales CSV to start the forecast.")
    st.markdown(
        """
        ### Recommended history

        **52+ weekly observations** are preferred.

        With 52 weeks the platform can calculate an actual same-period-last-year benchmark. 
        With more than 52 weeks it can increasingly validate and use annual seasonal methods.

        The platform does not create artificial LY growth, random forecast noise, or assumed promotional uplift.
        """
    )
    st.stop()

try:
    csv_bytes = uploaded_file.getvalue()
    valid, errors, warning_list, df_clean, measure_window = prepare_csv(csv_bytes)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if not valid:
    st.error("Schema validation failed.")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

for warning in warning_list:
    st.warning(warning)


# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.success(f"✅ {measure_window} measure set detected")

    categories = ["All"] + sorted(df_clean["Category"].unique().tolist())
    category = st.selectbox("1. Category", categories)

    d1 = df_clean if category == "All" else df_clean[df_clean["Category"] == category]

    subcategories = ["All"] + sorted(d1["Subcategory"].unique().tolist())
    subcategory = st.selectbox("2. Subcategory", subcategories)

    d2 = d1 if subcategory == "All" else d1[d1["Subcategory"] == subcategory]

    brands = ["All"] + sorted(d2["Brand"].unique().tolist())
    brand = st.selectbox("3. Brand", brands)

    d3 = d2 if brand == "All" else d2[d2["Brand"] == brand]

    products = sorted(d3["Product"].unique().tolist())
    selected_products = st.multiselect("4. Product SKU(s)", products)

    if selected_products:
        final_df = d3[d3["Product"].isin(selected_products)].copy()
        product_label = (
            selected_products[0]
            if len(selected_products) == 1
            else f"{len(selected_products)} selected SKUs"
        )
    else:
        final_df = d3.copy()
        product_label = "All Products"

    forecast_horizon = st.slider(
        "Forecast horizon (weeks)",
        min_value=1,
        max_value=8,
        value=4,
    )

    st.markdown("---")
    st.caption(f"Source rows: {len(final_df):,}")
    st.caption(f"Date range: {df_clean['date_key'].min().date()} → {df_clean['date_key'].max().date()}")

if final_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# ============================================================
# WEEKLY DATA
# ============================================================
weekly = aggregate_weekly(final_df)

if len(weekly) < 4:
    st.error("At least four weekly observations are required for a short-term forecast.")
    st.stop()


# ============================================================
# FORECASTS
# ============================================================
unit_result = run_forecast(weekly["Sales_Units"].values, horizon=forecast_horizon)
value_result = run_forecast(weekly["Sales_Value"].values, horizon=forecast_horizon)

last_date = weekly["date_key"].max()
future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon + 1)]

forecast_table = pd.DataFrame(
    {
        "Week": future_dates,
        "Forecast Units": unit_result["forecast"],
        "Units Lower": unit_result["lower"],
        "Units Upper": unit_result["upper"],
        "Forecast Value": value_result["forecast"],
        "Value Lower": value_result["lower"],
        "Value Upper": value_result["upper"],
    }
)


# ============================================================
# SCOPE / DATA STATUS
# ============================================================
scope_label = f"{category} > {subcategory} > {brand} > {product_label}"
st.markdown(f"### Forecast Scope\n**{scope_label}**")

weeks_available = len(weekly)

if weeks_available < 52:
    st.markdown(
        f'<div class="warning-card"><strong>{weeks_available} weekly periods available.</strong> '
        "Actual same-period-last-year comparison is not yet available for this scope.</div>",
        unsafe_allow_html=True,
    )
elif weeks_available == 52:
    st.markdown(
        '<div class="info-card"><strong>52-week history available.</strong> '
        "Actual YoY comparison and a same-period-last-year seasonal benchmark are enabled. "
        "Learned annual seasonality is not automatically treated as validated with only one annual cycle.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="success-card"><strong>{weeks_available} weekly periods available.</strong> '
        "Actual YoY comparison is enabled and annual seasonal models can be validated through backtesting.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# KPI SUMMARY
# ============================================================
st.markdown("### 📊 Forecast Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Volume", f"{forecast_table['Forecast Units'].sum():,.0f}")
col2.metric("Forecast Value", f"R {forecast_table['Forecast Value'].sum():,.0f}")
col3.metric("Volume Champion", unit_result["champion"])
col4.metric("Value Champion", value_result["champion"])

col5, col6, col7, col8 = st.columns(4)
col5.metric("History Used", f"{weeks_available} weeks")

if not unit_result["scores"].empty:
    col6.metric("Volume WMAPE", f"{unit_result['scores'].iloc[0]['WMAPE %']:.1f}%")
else:
    col6.metric("Volume WMAPE", "N/A")

if not value_result["scores"].empty:
    col7.metric("Value WMAPE", f"{value_result['scores'].iloc[0]['WMAPE %']:.1f}%")
else:
    col7.metric("Value WMAPE", "N/A")

col8.metric("Latest Weekly Units", f"{weekly['Sales_Units'].iloc[-1]:,.0f}")


# ============================================================
# SEASONAL / LY BENCHMARK
# ============================================================
st.markdown("---")
st.subheader("📅 Same Period Last Year Benchmark")

ly = same_period_last_year(weekly, future_dates)

if ly is None:
    st.info(
        f"Only {weeks_available} weeks are available for this selected scope. "
        "The platform will not manufacture a Last Year benchmark."
    )
else:
    forecast_units_total = float(forecast_table["Forecast Units"].sum())
    forecast_value_total = float(forecast_table["Forecast Value"].sum())
    ly_units_total = float(ly["LY Units"].sum())
    ly_value_total = float(ly["LY Value"].sum())

    yoy1, yoy2, yoy3, yoy4 = st.columns(4)
    yoy1.metric("Forecast Units", f"{forecast_units_total:,.0f}")
    yoy2.metric("LY Comparable Units", f"{ly_units_total:,.0f}")

    if ly_units_total > 0:
        yoy3.metric("Volume YoY", f"{((forecast_units_total / ly_units_total) - 1) * 100:+.1f}%")
    else:
        yoy3.metric("Volume YoY", "N/A")

    if ly_value_total > 0:
        yoy4.metric("Value YoY", f"{((forecast_value_total / ly_value_total) - 1) * 100:+.1f}%")
    else:
        yoy4.metric("Value YoY", "N/A")

    st.caption(
        "LY values are actual observations from approximately 52 weeks earlier. "
        "They are not simulated from the forecast."
    )

    if unit_result["seasonal_benchmark"] is not None:
        benchmark_total = float(np.sum(unit_result["seasonal_benchmark"]))
        st.write(
            f"**Seasonal Naïve 52-week benchmark (volume): {benchmark_total:,.0f} units** "
            "— this is the corresponding historical demand and is shown as a benchmark, not automatically treated as the final model with only one annual cycle."
        )


# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("---")
st.subheader("🧠 Model Selection & Backtesting")

mcol1, mcol2 = st.columns(2)

with mcol1:
    st.markdown("**Volume models**")
    if unit_result["scores"].empty:
        st.info("Not enough history for meaningful model comparison.")
    else:
        table = unit_result["scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(unit_result["champion"]))
        st.dataframe(
            table[
                ["Selected", "Model", "WMAPE %", "sMAPE %", "Bias %", "Forecasts Tested"]
            ].style.format(
                {
                    "WMAPE %": "{:.1f}%",
                    "sMAPE %": "{:.1f}%",
                    "Bias %": "{:+.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with mcol2:
    st.markdown("**Value models**")
    if value_result["scores"].empty:
        st.info("Not enough history for meaningful model comparison.")
    else:
        table = value_result["scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(value_result["champion"]))
        st.dataframe(
            table[
                ["Selected", "Model", "WMAPE %", "sMAPE %", "Bias %", "Forecasts Tested"]
            ].style.format(
                {
                    "WMAPE %": "{:.1f}%",
                    "sMAPE %": "{:.1f}%",
                    "Bias %": "{:+.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    f"Backtest method: rolling-origin validation with a {unit_result['bt_horizon']}-week horizon, "
    f"minimum training history of {unit_result['min_train']} weeks and {unit_result['n_origins']} validation origins. "
    "Champion = lowest WMAPE, with absolute bias used as the tie-breaker."
)

if unit_result["fallback_reason"]:
    st.warning(unit_result["fallback_reason"])


# ============================================================
# COMMERCIAL DRIVER DIAGNOSTICS
# ============================================================
st.markdown("---")
st.subheader("🔎 Recent Commercial Signals")

driver_items = build_driver_summary(weekly)

if not driver_items:
    st.info("Not enough history to calculate recent commercial-driver comparisons.")
else:
    for driver, detail in driver_items:
        st.markdown(
            f'<div class="card"><strong>{driver}</strong><br>{detail}</div>',
            unsafe_allow_html=True,
        )

st.caption(
    "These are observed historical movements. They are not causal elasticity estimates and the platform does not claim that a price, promo or distribution change will produce a specific unit uplift."
)


# ============================================================
# FORECAST DETAIL
# ============================================================
st.markdown("---")
st.subheader("🔮 Forecast Detail")

styled_forecast = forecast_table.copy()
styled_forecast["Week"] = styled_forecast["Week"].dt.strftime("%Y-%m-%d")

st.dataframe(
    styled_forecast.style.format(
        {
            "Forecast Units": "{:,.0f}",
            "Units Lower": "{:,.0f}",
            "Units Upper": "{:,.0f}",
            "Forecast Value": "R {:,.0f}",
            "Value Lower": "R {:,.0f}",
            "Value Upper": "R {:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Volume interval method: {unit_result['interval_method']}")
st.caption(f"Value interval method: {value_result['interval_method']}")


# ============================================================
# HISTORICAL CHART
# ============================================================
st.markdown("---")
st.subheader("📈 Historical vs Forecast")

metric = st.radio(
    "Chart metric",
    ["Volume (Units)", "Value (R)"],
    horizontal=True,
)

fig = go.Figure()

if metric.startswith("Volume"):
    hist_y = weekly["Sales_Units"]
    future_y = forecast_table["Forecast Units"]
    lower_y = forecast_table["Units Lower"]
    upper_y = forecast_table["Units Upper"]
    y_title = "Units"
else:
    hist_y = weekly["Sales_Value"]
    future_y = forecast_table["Forecast Value"]
    lower_y = forecast_table["Value Lower"]
    upper_y = forecast_table["Value Upper"]
    y_title = "Sales Value (R)"

fig.add_trace(
    go.Scatter(
        x=weekly["date_key"],
        y=hist_y,
        mode="lines+markers",
        name="Historical",
        line=dict(width=2.5),
        marker=dict(size=5),
    )
)

anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
anchor_y = [float(hist_y.iloc[-1])] + list(future_y)

fig.add_trace(
    go.Scatter(
        x=anchor_x,
        y=anchor_y,
        mode="lines+markers",
        name=f"Forecast ({unit_result['champion'] if metric.startswith('Volume') else value_result['champion']})",
        line=dict(width=3, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    )
)

fig.add_trace(
    go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_y) + list(lower_y[::-1]),
        fill="toself",
        fillcolor="rgba(30, 58, 138, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Empirical Forecast Range",
    )
)

# Optional LY benchmark line for volume when 52 weeks exist.
if metric.startswith("Volume") and unit_result["seasonal_benchmark"] is not None:
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=unit_result["seasonal_benchmark"],
            mode="lines+markers",
            name="52-Week LY Benchmark",
            line=dict(width=2, dash="dot"),
            marker=dict(size=5),
        )
    )

fig.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    xaxis=dict(title="Week"),
    yaxis=dict(title=y_title),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RECENT HISTORY TABLE
# ============================================================
st.markdown("---")
st.subheader("📊 Recent Weekly Performance")

recent = weekly.tail(12).copy()
recent["Week"] = recent["date_key"].dt.strftime("%Y-%m-%d")
recent = recent[
    [
        "Week",
        "Sales_Units",
        "Sales_Value",
        "Effective_RSP",
        "Promo_Depth_%",
        "Distribution",
    ]
]
recent.columns = [
    "Week",
    "Units",
    "Sales Value",
    "Effective RSP",
    "Promo Depth %",
    "Numeric Distribution %",
]

st.dataframe(
    recent.style.format(
        {
            "Units": "{:,.0f}",
            "Sales Value": "R {:,.0f}",
            "Effective RSP": "R {:,.2f}",
            "Promo Depth %": "{:.1f}%",
            "Numeric Distribution %": "{:.1f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# MODEL INFORMATION
# ============================================================
with st.expander("ℹ️ What forecasting model is being used?"):
    st.markdown(
        f"""
        The platform does **not force one forecasting algorithm**.

        For the selected scope it backtests multiple candidate models against the historical weekly observations and selects the model with the lowest **WMAPE**.

        **Current volume champion:** `{unit_result['champion']}`  
        **Current value champion:** `{value_result['champion']}`

        Candidate models include Naive, moving averages, weighted moving average, median, Simple Exponential Smoothing, Log Exponential Smoothing, Damped Holt and Theta. Croston-SBA is added for intermittent demand. The 52-week Seasonal Naive model is used as a same-period-last-year benchmark once 52 weeks exist and becomes eligible for automatic selection after enough history exists for a fair rolling backtest.

        The forecast contains **no random noise** and no invented YoY growth assumption.
        """
    )


# ============================================================
# DATA QUALITY DETAILS
# ============================================================
with st.expander("🔍 Data Quality Details"):
    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Rows in Scope", f"{len(final_df):,}")
    dq2.metric("SKUs", f"{final_df['Product'].nunique():,}")
    dq3.metric("Weekly Periods", f"{len(weekly):,}")
    dq4.metric("Latest Date", str(weekly["date_key"].max().date()))

    st.write(f"Detected measure window: **{measure_window}**")
    st.write("Columns available in the extract:")
    st.write(list(final_df.columns))
