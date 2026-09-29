import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


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
MEASURE_SETS = {
    "52-week": {
        "value": "52 Weeks CY Value",
        "price": "52 Weeks CY Ave Price Quantity",
        "promo": "52 Weeks CY Ave RSP On Promo",
        "baseline": "52 Weeks CY Sales Baseline",
        "incremental": "52 Weeks CY Sales Incremental",
    },
    "26-week": {
        "value": "26 Weeks CY Value",
        "price": "26 Weeks CY Ave Price Quantity",
        "promo": "26 Weeks CY Ave RSP On Promo",
        "baseline": None,
        "incremental": None,
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

    m52 = MEASURE_SETS["52-week"]
    base_52 = [m52["value"], m52["price"], m52["promo"]]

    if all(c in columns for c in base_52):
        if m52["baseline"] in columns:
            return "52-week", m52, None
        return "52-week-missing-baseline", m52, m52["baseline"]

    m26 = MEASURE_SETS["26-week"]
    base_26 = [m26["value"], m26["price"], m26["promo"]]

    if all(c in columns for c in base_26):
        return "26-week", m26, None

    return None, None, None


def validate_and_prepare(raw_df):
    """Validate the weekly Unilever extract and create clean analytical fields."""
    errors = []
    warnings_list = []

    missing_base = [c for c in BASE_COLUMNS if c not in raw_df.columns]
    if missing_base:
        errors.append("Missing required columns: " + ", ".join(missing_base))
        return False, errors, warnings_list, None, None

    measure_window, measures, missing_measure = detect_measure_set(raw_df.columns)

    if measure_window == "52-week-missing-baseline":
        errors.append(
            f"The 52-week sales measures are present, but the required baseline column "
            f"'{missing_measure}' is missing. Add this column to the extract before loading it."
        )
        return False, errors, warnings_list, None, None

    if measures is None:
        errors.append(
            "Could not find the required sales measure columns. Expected either the 52-week set "
            "(52 Weeks CY Value / 52 Weeks CY Ave Price Quantity / 52 Weeks CY Ave RSP On Promo / "
            "52 Weeks CY Sales Baseline) or the older 26-week set."
        )
        return False, errors, warnings_list, None, None

    df = raw_df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------
    df["date_key"] = pd.to_datetime(df["Full Date"], errors="coerce")
    bad_dates = int(df["date_key"].isna().sum())

    if bad_dates:
        warnings_list.append(f"{bad_dates:,} rows have invalid dates and were removed.")
        df = df.dropna(subset=["date_key"]).copy()

    if df.empty:
        return False, ["No valid dated rows remain after date validation."], warnings_list, None, None

    # --------------------------------------------------------
    # SALES VALUE / PRICES
    # --------------------------------------------------------
    df["Sales Value"] = (
        clean_number(df[measures["value"]])
        .fillna(0)
        .clip(lower=0)
    )

    df["Ave RSP"] = clean_number(df[measures["price"]])
    df["Promo RSP"] = clean_number(df[measures["promo"]])

    # --------------------------------------------------------
    # ACTUAL UNITS
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # BASELINE + INCREMENTAL UNITS (WITH AUTO-SCALE FIX)
    # --------------------------------------------------------
    if measures["baseline"] is not None:
        raw_baseline = (
            clean_number(df[measures["baseline"]])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

        baseline_missing = int(clean_number(df[measures["baseline"]]).isna().sum())
        if baseline_missing:
            warnings_list.append(
                f"{baseline_missing:,} rows have missing baseline units; they are treated as 0."
            )

        # Smart Scale Detection: Detect if baseline is in Currency Value or Units
        val_mean = df["Sales Value"].mean()
        units_mean = df["Sales Units"].mean()

        if units_mean > 0 and val_mean > 0 and abs(raw_baseline.mean() - val_mean) < abs(raw_baseline.mean() - units_mean):
            # Baseline is in Currency Value (Rand); convert to Units using Ave RSP
            df["Baseline Units"] = np.where(
                df["Ave RSP"] > 0,
                raw_baseline / df["Ave RSP"],
                0
            )
            warnings_list.append(
                "Source baseline field appears to be in Currency Value (Rand). Automatically converted to Baseline Units using Ave RSP."
            )
        else:
            df["Baseline Units"] = raw_baseline

        df["Baseline Units"] = (
            pd.to_numeric(df["Baseline Units"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

        calculated_incremental = df["Sales Units"] - df["Baseline Units"]
        df["Calculated Incremental Units"] = calculated_incremental

        incremental_col = measures.get("incremental")
        if incremental_col and incremental_col in df.columns:
            raw_inc = (
                clean_number(df[incremental_col])
                .replace([np.inf, -np.inf], np.nan)
            )

            missing_inc = int(raw_inc.isna().sum())
            if missing_inc:
                warnings_list.append(
                    f"{missing_inc:,} rows have missing incremental source data; calculated fallback used for those rows."
                )

            if units_mean > 0 and val_mean > 0 and abs(raw_inc.abs().mean() - val_mean) < abs(raw_inc.abs().mean() - units_mean):
                scaled_inc = np.where(df["Ave RSP"] > 0, raw_inc / df["Ave RSP"], np.nan)
            else:
                scaled_inc = raw_inc

            df["Source Incremental Units"] = scaled_inc
            df["Incremental Units"] = df["Source Incremental Units"].fillna(calculated_incremental)
            df["Incremental Source"] = np.where(
                df["Source Incremental Units"].notna(),
                "Source measure",
                "Calculated fallback",
            )

            df["Incremental Reconciliation Gap"] = np.where(
                df["Source Incremental Units"].notna(),
                df["Source Incremental Units"] - calculated_incremental,
                np.nan,
            )
        else:
            df["Source Incremental Units"] = np.nan
            df["Incremental Units"] = calculated_incremental
            df["Incremental Source"] = "Calculated fallback"
            df["Incremental Reconciliation Gap"] = np.nan
            warnings_list.append(
                "'52 Weeks CY Sales Incremental' is not present in this extract. "
                "Using Actual Units − Baseline Units as fallback."
            )

        df["Incremental % of Baseline"] = np.where(
            df["Baseline Units"] > 0,
            (df["Incremental Units"] / df["Baseline Units"]) * 100,
            np.nan,
        )
    else:
        df["Baseline Units"] = np.nan
        df["Source Incremental Units"] = np.nan
        df["Calculated Incremental Units"] = np.nan
        df["Incremental Reconciliation Gap"] = np.nan
        df["Incremental Units"] = np.nan
        df["Incremental Source"] = "Unavailable"
        df["Incremental % of Baseline"] = np.nan

        warnings_list.append(
            "This is the older 26-week measure set. 52-week baseline/incremental columns are unavailable."
        )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------
    if "Numeric Distribution" in df.columns:
        dist = clean_number(df["Numeric Distribution"])
        non_null = dist.dropna()
        if not non_null.empty and non_null.max() <= 1.0:
            dist = dist * 100
        df["Numeric Distribution"] = dist.clip(0, 100)
    else:
        df["Numeric Distribution"] = np.nan
        warnings_list.append(
            "Numeric Distribution is not in the extract."
        )

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------
    for col in ["Category", "Subcategory", "Brand", "Product"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["ProductsID"] = df["ProductsID"].fillna("Unknown").astype(str)

    # --------------------------------------------------------
    # PROMO DEPTH
    # --------------------------------------------------------
    df["Promo Depth %"] = np.where(
        (df["Ave RSP"] > 0) & df["Promo RSP"].notna(),
        ((df["Ave RSP"] - df["Promo RSP"]) / df["Ave RSP"]) * 100,
        np.nan,
    )
    df["Promo Depth %"] = (
        df["Promo Depth %"].replace([np.inf, -np.inf], np.nan).clip(0, 100)
    )

    unique_dates = pd.Series(df["date_key"].dropna().unique()).sort_values()
    n_dates = int(len(unique_dates))

    if n_dates < 12:
        warnings_list.append(f"Only {n_dates} unique weekly periods available. Forecast may be volatile.")
    elif n_dates < 52:
        warnings_list.append(f"{n_dates} weekly periods available. Actual YoY comparison requires 52 weeks.")

    return (
        True,
        errors,
        warnings_list,
        df.sort_values("date_key").reset_index(drop=True),
        measure_window,
    )


@st.cache_data(show_spinner=False)
def prepare_csv(csv_bytes):
    raw = pd.read_csv(pd.io.common.BytesIO(csv_bytes), low_memory=False)
    return validate_and_prepare(raw)


# ============================================================
# WEEKLY AGGREGATION
# ============================================================
def aggregate_weekly(df):
    if df.empty:
        return pd.DataFrame()

    base = (
        df.groupby("date_key", as_index=False)
        .agg(
            Sales_Units=("Sales Units", "sum"),
            Baseline_Units=("Baseline Units", "sum"),
            Incremental_Units=("Incremental Units", "sum"),
            Source_Incremental_Units=("Source Incremental Units", lambda x: x.sum(min_count=1)),
            Calculated_Incremental_Units=("Calculated Incremental Units", "sum"),
            Incremental_Reconciliation_Gap=("Incremental Reconciliation Gap", lambda x: x.sum(min_count=1)),
            Sales_Value=("Sales Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    base["Incremental_Units"] = base["Sales_Units"] - base["Baseline_Units"]

    base["Effective_RSP"] = np.where(
        base["Sales_Units"] > 0,
        base["Sales_Value"] / base["Sales_Units"],
        np.nan,
    )

    promo_part = df.copy()
    promo_part["Promo_RSP"] = pd.to_numeric(promo_part["Promo RSP"], errors="coerce")
    promo_part = promo_part[
        (promo_part["Promo_RSP"] > 0) & (promo_part["Sales Units"] > 0)
    ].copy()

    if promo_part.empty:
        base["Promo_RSP_Weighted"] = np.nan
    else:
        promo_part["Promo_Value_Proxy"] = promo_part["Promo_RSP"] * promo_part["Sales Units"]
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
    base["Promo_Depth_%"] = (
        base["Promo_Depth_%"].replace([np.inf, -np.inf], np.nan).clip(0, 100)
    )

    return base


# ============================================================
# MODEL HELPERS
# ============================================================
def is_constant_or_empty(y):
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.allclose(y, 0):
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
    value = np.average(recent, weights=weights)
    return np.repeat(max(0.0, float(value)), horizon)


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
        fit = SimpleExpSmoothing(
            y,
            initialization_method="estimated",
        ).fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_log_ses(y, horizon):
    y = np.asarray(y, dtype=float)
    if is_constant_or_empty(y):
        return forecast_naive(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_y = np.log1p(np.maximum(y, 0))
        fit = SimpleExpSmoothing(
            log_y,
            initialization_method="estimated",
        ).fit(optimized=True)
    return np.maximum(0.0, np.expm1(np.asarray(fit.forecast(horizon), dtype=float)))


def forecast_damped_holt(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 5 or is_constant_or_empty(y):
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Holt(
            y,
            initialization_method="estimated",
            damped_trend=True,
        ).fit(optimized=True)
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
    forecasts = [y[-52 + i] for i in range(horizon)]
    return np.maximum(0.0, np.asarray(forecasts, dtype=float))


def forecast_croston_sba(y, horizon, alpha=0.10):
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


def base_model_registry(n_obs, intermittent=False, allow_seasonal=False):
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

    if n_obs >= 70 and allow_seasonal:
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


def bias_pct(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return 100.0 * np.sum(pred - actual) / max(np.sum(np.abs(actual)), 1e-9)


# ============================================================
# BACKTEST SETUP & EVALUATION
# ============================================================
def backtest_setup(n_obs, horizon=4):
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
            rows.append({
                "Model": model_name,
                "Horizon": h,
                "Actual": float(a),
                "Prediction": float(p),
                "Error": float(p - a),
                "RelError": float((p - a) / a) if a > 0 else np.nan,
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate_models_cached(values_tuple, horizon=4, use_seasonal=False):
    y = np.asarray(values_tuple, dtype=float)
    intermittent = np.mean(y <= 0) >= 0.20
    actual_horizon, min_train, n_origins = backtest_setup(len(y), horizon)

    if n_origins <= 0:
        return pd.DataFrame(), pd.DataFrame(), None, actual_horizon, min_train, n_origins

    models = base_model_registry(len(y), intermittent=intermittent, allow_seasonal=use_seasonal)

    score_rows = []
    detail_rows = []

    for name, func in models.items():
        details = backtest_model(y, name, func, actual_horizon, min_train)
        if details.empty:
            continue

        detail_rows.append(details)
        score_rows.append({
            "Model": name,
            "WMAPE %": wmape(details["Actual"], details["Prediction"]),
            "sMAPE %": smape(details["Actual"], details["Prediction"]),
            "Bias %": bias_pct(details["Actual"], details["Prediction"]),
            "Forecasts Tested": len(details),
        })

    scores = pd.DataFrame(score_rows)
    details_all = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    if scores.empty:
        return scores, details_all, None, actual_horizon, min_train, n_origins

    scores["Abs Bias"] = scores["Bias %"].abs()
    scores = scores.sort_values(["WMAPE %", "Abs Bias"]).reset_index(drop=True)
    champion = str(scores.iloc[0]["Model"])

    return scores, details_all, champion, actual_horizon, min_train, n_origins


# ============================================================
# BASELINE + ACTUAL DECOMPOSITION BACKTEST
# ============================================================
def residual_forecast_zero(residual, horizon):
    return np.zeros(horizon)

def residual_forecast_naive(residual, horizon):
    return np.repeat(float(residual[-1]), horizon)

def residual_forecast_wma4(residual, horizon):
    n = min(4, len(residual))
    recent = residual[-n:]
    weights = np.arange(1, n + 1, dtype=float)
    value = np.average(recent, weights=weights)
    return np.repeat(float(value), horizon)

def residual_forecast_ses(residual, horizon):
    r = np.asarray(residual, dtype=float)
    if is_constant_or_empty(r):
        return residual_forecast_zero(r, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(r, initialization_method="estimated").fit(optimized=True)
    return np.asarray(fit.forecast(horizon), dtype=float)

RESIDUAL_MODELS = {
    "No future incremental uplift": residual_forecast_zero,
    "Last incremental week": residual_forecast_naive,
    "Weighted recent incremental": residual_forecast_wma4,
    "Exponential smoothing incremental": residual_forecast_ses,
}


def backtest_decomposed_model(actual, baseline, incremental, baseline_model_name, baseline_model_func, residual_model_name, residual_model_func, horizon, min_train):
    rows = []
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    incremental = np.asarray(incremental, dtype=float)

    for split in range(min_train, len(actual) - horizon + 1):
        train_baseline = baseline[:split]
        actual_future = actual[split : split + horizon]

        try:
            baseline_pred = np.asarray(baseline_model_func(train_baseline, horizon), dtype=float)
            incremental_history = incremental[:split]
            residual_pred = np.asarray(residual_model_func(incremental_history, horizon), dtype=float)
        except Exception:
            continue

        if len(baseline_pred) != horizon or len(residual_pred) != horizon:
            continue

        prediction = np.maximum(0.0, baseline_pred + residual_pred)
        model_name = f"Baseline: {baseline_model_name} + Incremental: {residual_model_name}"

        for h, (p, a) in enumerate(zip(prediction, actual_future), start=1):
            rows.append({
                "Model": model_name,
                "Horizon": h,
                "Actual": float(a),
                "Prediction": float(p),
                "Error": float(p - a),
                "RelError": float((p - a) / a) if a > 0 else np.nan,
                "Baseline Model": baseline_model_name,
                "Incremental Model": residual_model_name,
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate_decomposed_cached(actual_tuple, baseline_tuple, incremental_tuple, horizon=4, top_baseline_models=5):
    actual = np.asarray(actual_tuple, dtype=float)
    baseline = np.asarray(baseline_tuple, dtype=float)
    incremental = np.asarray(incremental_tuple, dtype=float)

    if len(actual) != len(baseline) or len(actual) < 16:
        return pd.DataFrame(), pd.DataFrame(), None, None, None, None

    actual_horizon, min_train, n_origins = backtest_setup(len(actual), horizon)

    base_scores, _, _, _, _, _ = evaluate_models_cached(tuple(float(x) for x in baseline), horizon=horizon, use_seasonal=False)

    if base_scores.empty:
        return pd.DataFrame(), pd.DataFrame(), None, actual_horizon, min_train, n_origins

    selected_baseline_names = base_scores.head(top_baseline_models)["Model"].tolist()
    baseline_registry = base_model_registry(len(baseline), intermittent=False, allow_seasonal=False)

    score_rows = []
    detail_rows = []

    for baseline_name in selected_baseline_names:
        if baseline_name not in baseline_registry:
            continue
        baseline_func = baseline_registry[baseline_name]

        for residual_name, residual_func in RESIDUAL_MODELS.items():
            details = backtest_decomposed_model(
                actual, baseline, incremental, baseline_name, baseline_func, residual_name, residual_func, actual_horizon, min_train
            )
            if details.empty:
                continue

            detail_rows.append(details)
            score_rows.append({
                "Model": details["Model"].iloc[0],
                "WMAPE %": wmape(details["Actual"], details["Prediction"]),
                "sMAPE %": smape(details["Actual"], details["Prediction"]),
                "Bias %": bias_pct(details["Actual"], details["Prediction"]),
                "Forecasts Tested": len(details),
            })

    scores = pd.DataFrame(score_rows)
    details_all = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    if scores.empty:
        return scores, details_all, None, actual_horizon, min_train, n_origins

    scores["Abs Bias"] = scores["Bias %"].abs()
    scores = scores.sort_values(["WMAPE %", "Abs Bias"]).reset_index(drop=True)
    champion = str(scores.iloc[0]["Model"])

    return scores, details_all, champion, actual_horizon, min_train, n_origins


# ============================================================
# FORECAST INTERVALS
# ============================================================
def build_prediction_intervals(champion_details, future_forecast):
    f = np.asarray(future_forecast, dtype=float)
    lower = np.zeros(len(f))
    upper = np.zeros(len(f))

    global_errors = champion_details["RelError"].dropna().to_numpy()

    for i in range(len(f)):
        horizon_number = i + 1
        horizon_errors = champion_details.loc[champion_details["Horizon"] == horizon_number, "RelError"].dropna().to_numpy()
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
# RUN BASELINE FORECAST
# ============================================================
def run_baseline_forecast(series, horizon=4):
    y = np.asarray(series, dtype=float)
    values_tuple = tuple(float(x) for x in y)

    scores, details, champion, bt_horizon, min_train, n_origins = evaluate_models_cached(
        values_tuple, horizon, use_seasonal=True
    )

    registry = base_model_registry(len(y), intermittent=np.mean(y <= 0) >= 0.20, allow_seasonal=True)
    fallback_reason = None

    if champion is None or champion not in registry:
        champion = "Naive Last Week"
        model_func = forecast_naive
        fallback_reason = "Fallback model used due to insufficient data."
    else:
        model_func = registry[champion]

    try:
        future = np.asarray(model_func(y, horizon), dtype=float)
    except Exception:
        future = forecast_naive(y, horizon)
        champion = "Naive Last Week"
        fallback_reason = "Champion fit failed; fallback Naive model applied."

    future = np.maximum(future, 0.0)

    champion_details = details[details["Model"] == champion].copy() if not details.empty else pd.DataFrame()

    if champion_details.empty:
        lower = future * 0.85
        upper = future * 1.15
        interval_method = "Indicative ±15%"
    else:
        lower, upper = build_prediction_intervals(champion_details, future)
        interval_method = "Empirical 10th–90th percentile"

    seasonal_benchmark = forecast_seasonal_naive52(y, horizon) if len(y) >= 52 else None

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
# RUN VOLUME FORECAST
# ============================================================
def run_final_volume_forecast(actual_series, baseline_series, incremental_series, horizon=4):
    actual = np.asarray(actual_series, dtype=float)
    baseline = np.asarray(baseline_series, dtype=float)
    incremental = np.asarray(incremental_series, dtype=float)

    baseline_result = run_baseline_forecast(baseline, horizon)

    actual_values_tuple = tuple(float(x) for x in actual)
    actual_scores, actual_details, actual_champion, bt_horizon, min_train, n_origins = evaluate_models_cached(
        actual_values_tuple, horizon=horizon, use_seasonal=True
    )

    actual_registry = base_model_registry(len(actual), intermittent=np.mean(actual <= 0) >= 0.20, allow_seasonal=True)

    if actual_champion in actual_registry:
        actual_func = actual_registry[actual_champion]
        try:
            direct_actual_forecast = np.maximum(0.0, np.asarray(actual_func(actual, horizon), dtype=float))
        except Exception:
            direct_actual_forecast = forecast_naive(actual, horizon)
    else:
        direct_actual_forecast = forecast_naive(actual, horizon)
        actual_champion = "Naive Last Week"

    recent_n = min(4, len(incremental))
    recent_incremental = float(np.median(incremental[-recent_n:]))

    baseline_only = baseline_result["forecast"]
    incremental_scenario = np.maximum(0.0, baseline_only + recent_incremental)

    decomp_scores, decomp_details, decomp_champion, decomp_horizon, decomp_min_train, decomp_origins = evaluate_decomposed_cached(
        tuple(float(x) for x in actual), tuple(float(x) for x in baseline), tuple(float(x) for x in incremental), horizon=horizon, top_baseline_models=5
    )

    decomp_forecast = None
    if decomp_champion:
        parts = decomp_champion.split(" + Incremental: ")
        baseline_name = parts[0].replace("Baseline: ", "", 1)
        residual_name = parts[1] if len(parts) > 1 else "No future incremental uplift"

        baseline_registry = base_model_registry(len(baseline), intermittent=False, allow_seasonal=False)
        baseline_func = baseline_registry.get(baseline_name)
        residual_func = RESIDUAL_MODELS.get(residual_name)

        if baseline_func and residual_func:
            try:
                base_future = baseline_func(baseline, horizon)
                residual_future = residual_func(incremental, horizon)
                decomp_forecast = np.maximum(0.0, np.asarray(base_future, dtype=float) + np.asarray(residual_future, dtype=float))
            except Exception:
                decomp_forecast = None

    return {
        "baseline": baseline_result,
        "direct_actual_forecast": direct_actual_forecast,
        "actual_champion": actual_champion,
        "actual_scores": actual_scores,
        "actual_details": actual_details,
        "incremental_history": incremental,
        "recent_incremental_median": recent_incremental,
        "incremental_scenario": incremental_scenario,
        "decomp_scores": decomp_scores,
        "decomp_details": decomp_details,
        "decomp_champion": decomp_champion,
        "decomp_forecast": decomp_forecast,
        "bt_horizon": bt_horizon,
        "min_train": min_train,
        "n_origins": n_origins,
    }


def same_period_last_year(weekly, future_dates):
    if len(weekly) < 52:
        return None

    ly_rows = []
    for future_date in future_dates:
        target = future_date - timedelta(weeks=52)
        distance = (weekly["date_key"] - target).abs().dt.days
        idx = distance.idxmin()

        if int(distance.loc[idx]) > 7:
            return None

        ly_rows.append({
            "Week": future_date,
            "LY Units": float(weekly.loc[idx, "Sales_Units"]),
            "LY Baseline Units": float(weekly.loc[idx, "Baseline_Units"]),
            "LY Incremental Units": float(weekly.loc[idx, "Incremental_Units"]),
            "LY Value": float(weekly.loc[idx, "Sales_Value"]),
            "LY Date": weekly.loc[idx, "date_key"],
        })

    return pd.DataFrame(ly_rows)


def recent_change(series, recent_n=4, prior_n=4):
    s = pd.Series(series).dropna()
    if len(s) < recent_n + prior_n:
        return np.nan
    recent = s.iloc[-recent_n:].mean()
    prior = s.iloc[-recent_n - prior_n : -recent_n].mean()
    return np.nan if prior == 0 else 100.0 * (recent / prior - 1.0)


def build_driver_summary(weekly):
    items = []
    actual_change = recent_change(weekly["Sales_Units"])
    baseline_change = recent_change(weekly["Baseline_Units"])
    incremental_change = recent_change(weekly["Incremental_Units"])
    price_change = recent_change(weekly["Effective_RSP"])
    promo_change = recent_change(weekly["Promo_Depth_%"])
    dist_change = recent_change(weekly["Distribution"])

    if np.isfinite(actual_change):
        items.append(("Actual Volume", f"Latest 4-week average actual volume is {actual_change:+.1f}% vs preceding 4 weeks."))
    if np.isfinite(baseline_change):
        items.append(("Baseline Demand", f"Latest 4-week average baseline demand is {baseline_change:+.1f}% vs preceding 4 weeks."))
    if np.isfinite(incremental_change):
        items.append(("Incremental / Commercial", f"Recent incremental component changed {incremental_change:+.1f}% vs preceding 4 weeks."))
    if np.isfinite(price_change):
        items.append(("Price", f"Effective RSP changed {price_change:+.1f}% vs preceding 4 weeks."))
    if np.isfinite(promo_change):
        items.append(("Promotion", f"Promotional depth changed {promo_change:+.1f}% vs preceding 4 weeks."))
    if np.isfinite(dist_change):
        items.append(("Distribution", f"Numeric distribution changed {dist_change:+.1f}% vs preceding 4 weeks."))

    return items


# ============================================================
# APP START
# ============================================================
st.markdown('<div class="main-header">Unilever Demand & Forecast Platform</div>', unsafe_allow_html=True)
st.caption("Baseline-driven weekly demand forecasting with model backtesting, decomposition, and commercial diagnostics.")

with st.sidebar:
    st.header("📥 Data & Filters")
    uploaded_file = st.file_uploader("Upload weekly sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload the Unilever weekly sales CSV to start the forecast.")
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
        product_label = selected_products[0] if len(selected_products) == 1 else f"{len(selected_products)} selected SKUs"
    else:
        final_df = d3.copy()
        product_label = "All Products"

    forecast_horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=8, value=4)

    forecast_basis = st.radio(
        "Forecast basis",
        [
            "Model-Selected Baseline + Incremental",
            "Baseline Demand Only",
            "Baseline + Recent Incremental Scenario",
        ],
        index=0
    )

if final_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# ============================================================
# WEEKLY AGGREGATION & FORECAST
# ============================================================
weekly = aggregate_weekly(final_df)

if len(weekly) < 4:
    st.error("At least 4 weekly observations are required for a forecast.")
    st.stop()

last_date = weekly["date_key"].max()
future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon + 1)]

volume_result = run_final_volume_forecast(
    weekly["Sales_Units"].values,
    weekly["Baseline_Units"].values,
    weekly["Incremental_Units"].values,
    horizon=forecast_horizon,
)

value_result = run_baseline_forecast(
    weekly["Sales_Value"].values,
    horizon=forecast_horizon,
)

baseline_forecast = volume_result["baseline"]["forecast"]
scenario_forecast = volume_result["incremental_scenario"]

selected_volume_lower = volume_result["baseline"]["lower"]
selected_volume_upper = volume_result["baseline"]["upper"]

if forecast_basis == "Model-Selected Baseline + Incremental":
    if volume_result["decomp_forecast"] is not None:
        selected_volume_forecast = volume_result["decomp_forecast"]
        forecast_basis_label = "Model-Selected Baseline + Incremental"
        if not volume_result["decomp_details"].empty:
            decomp_rows = volume_result["decomp_details"]
            selected_decomp_rows = decomp_rows[decomp_rows["Model"] == volume_result["decomp_champion"]].copy()
            if not selected_decomp_rows.empty:
                selected_volume_lower, selected_volume_upper = build_prediction_intervals(
                    selected_decomp_rows, selected_volume_forecast
                )
    else:
        selected_volume_forecast = baseline_forecast
        forecast_basis_label = "Baseline Demand Only (decomposition unavailable)"
elif forecast_basis == "Baseline Demand Only":
    selected_volume_forecast = baseline_forecast
    forecast_basis_label = "Baseline Demand Only"
else:
    selected_volume_forecast = scenario_forecast
    forecast_basis_label = "Baseline + Recent Incremental Scenario"

selected_value_forecast = value_result["forecast"]

forecast_table = pd.DataFrame({
    "Week": future_dates,
    "Baseline Forecast Units": baseline_forecast,
    "Selected Forecast Units": selected_volume_forecast,
    "Units Lower": selected_volume_lower,
    "Units Upper": selected_volume_upper,
    "Forecast Value": selected_value_forecast,
    "Value Lower": value_result["lower"],
    "Value Upper": value_result["upper"],
    "Recent Incremental Scenario Units": scenario_forecast,
})


# ============================================================
# KPI SUMMARY
# ============================================================
scope_label = f"{category} > {subcategory} > {brand} > {product_label}"
st.markdown(f"### Forecast Scope\n**{scope_label}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Volume", f"{forecast_table['Selected Forecast Units'].sum():,.0f}")
col2.metric("Forecast Value", f"R {forecast_table['Forecast Value'].sum():,.0f}")
col3.metric("Baseline Model", volume_result["baseline"]["champion"])
col4.metric("Forecast Basis", forecast_basis_label)


# ============================================================
# HISTORICAL + FORECAST CHART (FIXED SCALE & CONNECTIVITY)
# ============================================================
st.markdown("---")
st.subheader("📈 Actual vs Baseline vs Forecast")

metric = st.radio("Chart metric", ["Volume (Units)", "Value (R)"], horizontal=True)

fig = go.Figure()

if metric.startswith("Volume"):
    # Actual history
    fig.add_trace(
        go.Scatter(
            x=weekly["date_key"],
            y=weekly["Sales_Units"],
            mode="lines+markers",
            name="Actual Units",
            line=dict(width=2.5, color="#7C3AED"),
            marker=dict(size=5),
        )
    )

    # Baseline history
    fig.add_trace(
        go.Scatter(
            x=weekly["date_key"],
            y=weekly["Baseline_Units"],
            mode="lines+markers",
            name="Baseline Units",
            line=dict(width=2, dash="dot", color="#EA580C"),
            marker=dict(size=4),
        )
    )

    # Dynamic Anchor Selection for Seamless Continuity
    if forecast_basis == "Baseline Demand Only":
        anchor_start = float(weekly["Baseline_Units"].iloc[-1])
        forecast_legend_name = f"Baseline Forecast ({volume_result['baseline']['champion']})"
    else:
        anchor_start = float(weekly["Sales_Units"].iloc[-1])
        forecast_legend_name = f"Forecast ({forecast_basis_label})"

    anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
    anchor_y = [anchor_start] + list(selected_volume_forecast)

    fig.add_trace(
        go.Scatter(
            x=anchor_x,
            y=anchor_y,
            mode="lines+markers",
            name=forecast_legend_name,
            line=dict(width=3, dash="dash", color="#059669"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    # Forecast Range Interval Band
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(selected_volume_upper) + list(selected_volume_lower[::-1]),
            fill="toself",
            fillcolor="rgba(5, 150, 105, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Forecast Range",
        )
    )

    # 52-Week Last Year Benchmark
    if volume_result["baseline"]["seasonal_benchmark"] is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=volume_result["baseline"]["seasonal_benchmark"],
                mode="lines+markers",
                name="52-Week LY Benchmark",
                line=dict(width=2, dash="dot", color="#D97706"),
                marker=dict(size=5),
            )
        )

    y_title = "Units"

else:
    hist_y = weekly["Sales_Value"]
    future_y = forecast_table["Forecast Value"]

    fig.add_trace(
        go.Scatter(
            x=weekly["date_key"],
            y=hist_y,
            mode="lines+markers",
            name="Historical Sales Value",
            line=dict(width=2.5, color="#1E3A8A"),
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
            name=f"Value Forecast ({value_result['champion']})",
            line=dict(width=3, dash="dash", color="#059669"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(value_result["upper"]) + list(value_result["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(5, 150, 105, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Value Forecast Range",
        )
    )

    y_title = "Sales Value (R)"

fig.update_layout(
    template="plotly_white",
    height=520,
    hovermode="x unified",
    xaxis=dict(title="Week"),
    yaxis=dict(title=y_title),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# FORECAST DETAIL TABLE
# ============================================================
st.markdown("---")
st.subheader("🔮 Forecast Detail")

styled_forecast = forecast_table.copy()
styled_forecast["Week"] = styled_forecast["Week"].dt.strftime("%Y-%m-%d")

st.dataframe(
    styled_forecast.style.format({
        "Baseline Forecast Units": "{:,.0f}",
        "Selected Forecast Units": "{:,.0f}",
        "Units Lower": "{:,.0f}",
        "Units Upper": "{:,.0f}",
        "Forecast Value": "R {:,.0f}",
        "Value Lower": "R {:,.0f}",
        "Value Upper": "R {:,.0f}",
        "Recent Incremental Scenario Units": "{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)
