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
# The production 52-week extract is expected to contain:
#   52 Weeks CY Value
#   52 Weeks CY Ave Price Quantity
#   52 Weeks CY Ave RSP On Promo
#   52 Weeks CY Sales Baseline
#   52 Weeks CY Sales Incremental
#
# Baseline and Incremental are assumed to be WEEKLY UNIT measures for
# the same observation/SKU. They are not treated as rolling totals.
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
    """
    Detect the current 52-week contract first.

    If the 52-week sales measures are present but the baseline column
    is missing, return a specific error state rather than silently
    falling back to a model that ignores the baseline.
    """
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
    # The supplied sales observation is weekly. Units are derived at
    # row level and are NOT rounded until presentation.
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
    # BASELINE + INCREMENTAL UNITS
    # --------------------------------------------------------
    if measures["baseline"] is not None:
        df["Baseline Units"] = (
            clean_number(df[measures["baseline"]])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

        baseline_missing = int(clean_number(df[measures["baseline"]]).isna().sum())
        if baseline_missing:
            warnings_list.append(
                f"{baseline_missing:,} rows have missing baseline units; they are treated as 0. "
                "Verify that 0 represents true zero demand in the source system."
            )

        calculated_incremental = df["Sales Units"] - df["Baseline Units"]
        df["Calculated Incremental Units"] = calculated_incremental

        incremental_col = measures.get("incremental")
        if incremental_col and incremental_col in df.columns:
            df["Source Incremental Units"] = (
                clean_number(df[incremental_col])
                .replace([np.inf, -np.inf], np.nan)
            )

            missing_inc = int(df["Source Incremental Units"].isna().sum())
            if missing_inc:
                warnings_list.append(
                    f"{missing_inc:,} rows have missing 52 Weeks CY Sales Incremental; calculated Actual − Baseline is used only for those rows."
                )

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
                "The platform is using Actual Units − Baseline Units as a temporary fallback."
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
            "This is the older 26-week measure set. 52 Weeks CY Sales Baseline and 52 Weeks CY Sales Incremental are unavailable."
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
            "Numeric Distribution is not in the extract. Distribution diagnostics will be unavailable."
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

    # --------------------------------------------------------
    # UNIQUE WEEKLY PERIODS
    # --------------------------------------------------------
    unique_dates = pd.Series(df["date_key"].dropna().unique()).sort_values()
    n_dates = int(len(unique_dates))

    if n_dates < 12:
        warnings_list.append(
            f"Only {n_dates} unique weekly periods are available. Forecast accuracy may be unstable."
        )
    elif n_dates < 52:
        warnings_list.append(
            f"{n_dates} weekly periods are available. Actual same-period-last-year comparison requires 52 weeks."
        )
    elif n_dates == 52:
        warnings_list.append(
            "52 weekly observations are available. Actual YoY comparison and the 52-week seasonal benchmark "
            "are enabled. A learned annual seasonal model is not automatically selected with only one annual cycle."
        )

    # --------------------------------------------------------
    # WEEKLY REGULARITY
    # --------------------------------------------------------
    if n_dates > 1:
        gaps = pd.Series(unique_dates).diff().dropna().dt.days.astype(float)
        if not gaps.empty:
            abnormal = int((gaps != 7).sum())
            if abnormal:
                warnings_list.append(
                    f"{abnormal} date gaps are not exactly 7 days. The platform uses the actual observation dates."
                )

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
    """
    Aggregate weekly source rows to the selected scope.

    Baseline Units are summed across SKUs. This is appropriate when
    the source baseline is a weekly unit quantity per SKU.
    """
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

    # Volume-weighted promo RSP for diagnosis only.
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
    return np.maximum(
        0.0,
        np.asarray(fit.forecast(horizon), dtype=float),
    )


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
    return np.maximum(
        0.0,
        np.expm1(np.asarray(fit.forecast(horizon), dtype=float)),
    )


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
    return np.maximum(
        0.0,
        np.asarray(fit.forecast(horizon), dtype=float),
    )


def forecast_theta(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 8 or is_constant_or_empty(y):
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThetaModel(y, period=1).fit()
    return np.maximum(
        0.0,
        np.asarray(fit.forecast(horizon), dtype=float),
    )


def forecast_seasonal_naive52(y, horizon):
    y = np.asarray(y, dtype=float)
    if len(y) < 52:
        raise ValueError("Need at least 52 weekly observations.")

    # The next forecast week is matched to the corresponding week 52 weeks earlier.
    forecasts = []
    for i in range(horizon):
        forecasts.append(y[-52 + i])

    return np.maximum(
        0.0,
        np.asarray(forecasts, dtype=float),
    )


def forecast_croston_sba(y, horizon, alpha=0.10):
    """Croston-SBA for genuinely intermittent demand."""
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

    # With one year of history the Seasonal Naive is shown as a benchmark,
    # but it is not automatically selected because there is not enough history
    # for meaningful rolling validation. It becomes eligible once enough
    # post-year observations exist to create validation origins.
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
# BACKTEST SETUP
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


# ============================================================
# DIRECT MODEL BACKTEST
# ============================================================
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
def evaluate_models_cached(values_tuple, horizon=4, use_seasonal=False):
    y = np.asarray(values_tuple, dtype=float)
    intermittent = np.mean(y <= 0) >= 0.20
    actual_horizon, min_train, n_origins = backtest_setup(len(y), horizon)

    if n_origins <= 0:
        return pd.DataFrame(), pd.DataFrame(), None, actual_horizon, min_train, n_origins

    models = base_model_registry(
        len(y),
        intermittent=intermittent,
        allow_seasonal=use_seasonal,
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
                "Bias %": bias_pct(details["Actual"], details["Prediction"]),
                "Forecasts Tested": len(details),
            }
        )

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
    """No future commercial uplift is assumed."""
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
        fit = SimpleExpSmoothing(
            r,
            initialization_method="estimated",
        ).fit(optimized=True)
    return np.asarray(fit.forecast(horizon), dtype=float)


RESIDUAL_MODELS = {
    "No future incremental uplift": residual_forecast_zero,
    "Last incremental week": residual_forecast_naive,
    "Weighted recent incremental": residual_forecast_wma4,
    "Exponential smoothing incremental": residual_forecast_ses,
}


def backtest_decomposed_model(
    actual,
    baseline,
    incremental,
    baseline_model_name,
    baseline_model_func,
    residual_model_name,
    residual_model_func,
    horizon,
    min_train,
):
    rows = []
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    incremental = np.asarray(incremental, dtype=float)

    for split in range(min_train, len(actual) - horizon + 1):
        train_actual = actual[:split]
        train_baseline = baseline[:split]
        actual_future = actual[split : split + horizon]

        try:
            baseline_pred = np.asarray(
                baseline_model_func(train_baseline, horizon),
                dtype=float,
            )
            incremental_history = incremental[:split]
            residual_pred = np.asarray(
                residual_model_func(incremental_history, horizon),
                dtype=float,
            )
        except Exception:
            continue

        if (
            len(baseline_pred) != horizon
            or len(residual_pred) != horizon
            or not np.all(np.isfinite(baseline_pred))
            or not np.all(np.isfinite(residual_pred))
        ):
            continue

        prediction = np.maximum(
            0.0,
            baseline_pred + residual_pred,
        )

        model_name = (
            f"Baseline: {baseline_model_name} + Incremental: {residual_model_name}"
        )

        for h, (p, a) in enumerate(zip(prediction, actual_future), start=1):
            rows.append(
                {
                    "Model": model_name,
                    "Horizon": h,
                    "Actual": float(a),
                    "Prediction": float(p),
                    "Error": float(p - a),
                    "RelError": float((p - a) / a) if a > 0 else np.nan,
                    "Baseline Model": baseline_model_name,
                    "Incremental Model": residual_model_name,
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate_decomposed_cached(actual_tuple, baseline_tuple, incremental_tuple, horizon=4, top_baseline_models=5):
    actual = np.asarray(actual_tuple, dtype=float)
    baseline = np.asarray(baseline_tuple, dtype=float)
    incremental = np.asarray(incremental_tuple, dtype=float)

    if len(actual) != len(baseline) or len(actual) != len(incremental) or len(actual) < 16:
        return pd.DataFrame(), pd.DataFrame(), None, None, None, None

    actual_horizon, min_train, n_origins = backtest_setup(len(actual), horizon)

    # First evaluate baseline models.
    base_scores, _, _, _, _, _ = evaluate_models_cached(
        tuple(float(x) for x in baseline),
        horizon=horizon,
        use_seasonal=False,
    )

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
                actual,
                baseline,
                incremental,
                baseline_name,
                baseline_func,
                residual_name,
                residual_func,
                actual_horizon,
                min_train,
            )

            if details.empty:
                continue

            detail_rows.append(details)
            score_rows.append(
                {
                    "Model": details["Model"].iloc[0],
                    "WMAPE %": wmape(details["Actual"], details["Prediction"]),
                    "sMAPE %": smape(details["Actual"], details["Prediction"]),
                    "Bias %": bias_pct(details["Actual"], details["Prediction"]),
                    "Forecasts Tested": len(details),
                }
            )

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
    """Empirical forecast interval from rolling-origin relative errors."""
    f = np.asarray(future_forecast, dtype=float)
    lower = np.zeros(len(f))
    upper = np.zeros(len(f))

    global_errors = champion_details["RelError"].dropna().to_numpy()

    for i in range(len(f)):
        horizon_number = i + 1
        horizon_errors = champion_details.loc[
            champion_details["Horizon"] == horizon_number,
            "RelError",
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
# RUN BASELINE FORECAST
# ============================================================
def run_baseline_forecast(series, horizon=4):
    y = np.asarray(series, dtype=float)
    result = {}
    values_tuple = tuple(float(x) for x in y)

    scores, details, champion, bt_horizon, min_train, n_origins = evaluate_models_cached(
        values_tuple,
        horizon,
        use_seasonal=True,
    )

    registry = base_model_registry(
        len(y),
        intermittent=np.mean(y <= 0) >= 0.20,
        allow_seasonal=True,
    )

    fallback_reason = None

    if champion is None or champion not in registry:
        champion = "Naive Last Week"
        model_func = forecast_naive
        fallback_reason = "Insufficient history or model-fitting failures prevented baseline model selection."
    else:
        model_func = registry[champion]

    try:
        future = np.asarray(model_func(y, horizon), dtype=float)
    except Exception:
        future = forecast_naive(y, horizon)
        champion = "Naive Last Week"
        fallback_reason = "Champion baseline model failed during final fit; Naive Last Week was used."

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

    seasonal_benchmark = None
    if len(y) >= 52:
        seasonal_benchmark = forecast_seasonal_naive52(y, horizon)

    result.update(
        {
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
    )

    return result


# ============================================================
# RUN ACTUAL / BASELINE DECOMPOSED FORECAST
# ============================================================
def run_final_volume_forecast(actual_series, baseline_series, incremental_series, horizon=4):
    """
    Final production forecast is baseline-driven.

    The supplied baseline is treated as the underlying demand signal.
    Historical actual-minus-baseline is shown as incremental/commercial
    activity, but no future uplift is assumed unless the user explicitly
    selects a scenario that carries recent incremental demand forward.
    """
    actual = np.asarray(actual_series, dtype=float)
    baseline = np.asarray(baseline_series, dtype=float)
    incremental = np.asarray(incremental_series, dtype=float)

    baseline_result = run_baseline_forecast(baseline, horizon)

    # Direct actual model is retained as a benchmark so users can compare
    # the old time-series-only view with the baseline-driven forecast.
    actual_values_tuple = tuple(float(x) for x in actual)
    actual_scores, actual_details, actual_champion, bt_horizon, min_train, n_origins = evaluate_models_cached(
        actual_values_tuple,
        horizon=horizon,
        use_seasonal=True,
    )

    actual_registry = base_model_registry(
        len(actual),
        intermittent=np.mean(actual <= 0) >= 0.20,
        allow_seasonal=True,
    )

    if actual_champion in actual_registry:
        actual_func = actual_registry[actual_champion]
        try:
            direct_actual_forecast = np.maximum(
                0.0,
                np.asarray(actual_func(actual, horizon), dtype=float),
            )
        except Exception:
            direct_actual_forecast = forecast_naive(actual, horizon)
    else:
        direct_actual_forecast = forecast_naive(actual, horizon)
        actual_champion = "Naive Last Week"

    # Source incremental history.
    recent_n = min(4, len(incremental))
    recent_incremental = float(np.median(incremental[-recent_n:]))

    # Scenario 1: baseline only.
    baseline_only = baseline_result["forecast"]
    incremental_scenario = np.maximum(
        0.0,
        baseline_only + recent_incremental,
    )

    # Decomposed model selection is used as a validation diagnostic only.
    decomp_scores, decomp_details, decomp_champion, decomp_horizon, decomp_min_train, decomp_origins = (
        evaluate_decomposed_cached(
            tuple(float(x) for x in actual),
            tuple(float(x) for x in baseline),
            tuple(float(x) for x in incremental),
            horizon=horizon,
            top_baseline_models=5,
        )
    )

    decomp_forecast = None
    if decomp_champion:
        parts = decomp_champion.split(" + Incremental: ")
        baseline_name = parts[0].replace("Baseline: ", "", 1)
        residual_name = parts[1] if len(parts) > 1 else "No future incremental uplift"

        baseline_registry = base_model_registry(
            len(baseline),
            intermittent=False,
            allow_seasonal=False,
        )
        baseline_func = baseline_registry.get(baseline_name)
        residual_func = RESIDUAL_MODELS.get(residual_name)

        if baseline_func and residual_func:
            try:
                base_future = baseline_func(baseline, horizon)
                residual_future = residual_func(incremental, horizon)
                decomp_forecast = np.maximum(
                    0.0,
                    np.asarray(base_future, dtype=float) + np.asarray(residual_future, dtype=float),
                )
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
        "decomp_horizon": decomp_horizon,
        "decomp_min_train": decomp_min_train,
        "decomp_origins": decomp_origins,
    }


# ============================================================
# SAME-PERIOD-LAST-YEAR
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
                "LY Baseline Units": float(weekly.loc[idx, "Baseline_Units"]),
                "LY Incremental Units": float(weekly.loc[idx, "Incremental_Units"]),
                "LY Value": float(weekly.loc[idx, "Sales_Value"]),
                "LY Date": weekly.loc[idx, "date_key"],
            }
        )

    return pd.DataFrame(ly_rows)


# ============================================================
# COMMERCIAL DIAGNOSTICS
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

    actual_change = recent_change(weekly["Sales_Units"])
    baseline_change = recent_change(weekly["Baseline_Units"])
    incremental_change = recent_change(weekly["Incremental_Units"])
    price_change = recent_change(weekly["Effective_RSP"])
    promo_change = recent_change(weekly["Promo_Depth_%"])
    dist_change = recent_change(weekly["Distribution"])

    if np.isfinite(actual_change):
        items.append(
            (
                "Actual Volume",
                f"Latest 4-week average actual volume is {actual_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    if np.isfinite(baseline_change):
        items.append(
            (
                "Baseline Demand",
                f"Latest 4-week average baseline demand is {baseline_change:+.1f}% versus the preceding 4 weeks.",
            )
        )

    if np.isfinite(incremental_change):
        items.append(
            (
                "Incremental / Commercial",
                f"The recent incremental component changed {incremental_change:+.1f}% versus the preceding 4 weeks.",
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
    "Baseline-driven weekly demand forecasting with model backtesting, actual-vs-baseline decomposition, "
    "same-period-last-year comparison, and commercial diagnostics."
)

with st.sidebar:
    st.header("📥 Data & Filters")
    uploaded_file = st.file_uploader("Upload weekly sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload the Unilever weekly sales CSV to start the forecast.")
    st.markdown(
        """
        ### Required 52-week measures

        The current production extract should contain:

        - **52 Weeks CY Value**
        - **52 Weeks CY Ave Price Quantity**
        - **52 Weeks CY Ave RSP On Promo**
        - **52 Weeks CY Sales Baseline**
        - **52 Weeks CY Sales Incremental**

        Baseline and Incremental are treated as **weekly unit measures** for the same
        SKU/week observation. Actual sales units are derived from weekly value ÷ average RSP.
        If the incremental source field is temporarily absent, the platform transparently falls back to Actual − Baseline and flags the fallback.
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
    if measure_window == "52-week":
        incremental_status = " + incremental" if "52 Weeks CY Sales Incremental" in df_clean.columns else " (calculated incremental fallback)"
        st.success(f"✅ 52-week measure set + baseline detected{incremental_status}")
    else:
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

    forecast_basis = st.radio(
        "Forecast basis",
        [
            "Model-Selected Baseline + Incremental",
            "Baseline Demand Only",
            "Baseline + Recent Incremental Scenario",
        ],
        index=0,
        help=(
            "Baseline Demand forecasts the underlying baseline only. "
            "The incremental scenario carries the recent median actual-minus-baseline component forward "
            "for what-if analysis; it is not a claim about future promotions."
        ),
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

if weekly["Baseline_Units"].isna().all():
    st.error(
        "52 Weeks CY Sales Baseline is required for the baseline-driven production forecast."
    )
    st.stop()


# ============================================================
# FINAL FORECAST
# ============================================================
last_date = weekly["date_key"].max()
future_dates = [
    last_date + timedelta(weeks=i)
    for i in range(1, forecast_horizon + 1)
]

volume_result = run_final_volume_forecast(
    weekly["Sales_Units"].values,
    weekly["Baseline_Units"].values,
    weekly["Incremental_Units"].values,
    horizon=forecast_horizon,
)

value_values = weekly["Sales_Value"].values
value_result = run_baseline_forecast(
    value_values,
    horizon=forecast_horizon,
)

baseline_forecast = volume_result["baseline"]["forecast"]
scenario_forecast = volume_result["incremental_scenario"]

# Default volume intervals come from the baseline champion. When the
# model-selected decomposition is used, replace them with the empirical
# error distribution of the selected baseline+incremental combination.
selected_volume_lower = volume_result["baseline"]["lower"]
selected_volume_upper = volume_result["baseline"]["upper"]

if forecast_basis == "Model-Selected Baseline + Incremental":
    if volume_result["decomp_forecast"] is not None:
        selected_volume_forecast = volume_result["decomp_forecast"]
        forecast_basis_label = "Model-Selected Baseline + Incremental"

        if not volume_result["decomp_details"].empty:
            decomp_rows = volume_result["decomp_details"]
            selected_decomp_rows = decomp_rows[
                decomp_rows["Model"] == volume_result["decomp_champion"]
            ].copy()
            if not selected_decomp_rows.empty:
                selected_volume_lower, selected_volume_upper = build_prediction_intervals(
                    selected_decomp_rows,
                    selected_volume_forecast,
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

# Value is kept as an independently modelled weekly sales-value forecast.
selected_value_forecast = value_result["forecast"]

forecast_table = pd.DataFrame(
    {
        "Week": future_dates,
        "Baseline Forecast Units": baseline_forecast,
        "Selected Forecast Units": selected_volume_forecast,
        "Units Lower": selected_volume_lower,
        "Units Upper": selected_volume_upper,
        "Forecast Value": selected_value_forecast,
        "Value Lower": value_result["lower"],
        "Value Upper": value_result["upper"],
        "Recent Incremental Scenario Units": scenario_forecast,
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
        "Baseline forecasting is active, but actual same-period-last-year comparison is not yet available.</div>",
        unsafe_allow_html=True,
    )
elif weeks_available == 52:
    st.markdown(
        '<div class="info-card"><strong>52 weekly periods available.</strong> '
        "The baseline forecast is active, actual YoY comparison is enabled, and the 52-week seasonal benchmark "
        "is shown separately. With exactly one annual cycle, the seasonal benchmark is not allowed to overrule "
        "the validated baseline forecast automatically.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="success-card"><strong>{weeks_available} weekly periods available.</strong> '
        "Actual YoY comparison is enabled and annual seasonal models can be validated when enough post-year "
        "history is present.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# KPI SUMMARY
# ============================================================
st.markdown("### 📊 Forecast Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Forecast Volume",
    f"{forecast_table['Selected Forecast Units'].sum():,.0f}",
)
col2.metric(
    "Forecast Value",
    f"R {forecast_table['Forecast Value'].sum():,.0f}",
)
col3.metric(
    "Baseline Model",
    volume_result["baseline"]["champion"],
)
col4.metric(
    "Forecast Basis",
    forecast_basis_label,
)

col5, col6, col7, col8 = st.columns(4)
col5.metric("History Used", f"{weeks_available} weeks")

if not volume_result["baseline"]["scores"].empty:
    col6.metric(
        "Baseline WMAPE",
        f"{volume_result['baseline']['scores'].iloc[0]['WMAPE %']:.1f}%",
    )
else:
    col6.metric("Baseline WMAPE", "N/A")

if not volume_result["actual_scores"].empty:
    col7.metric(
        "Actual-Only WMAPE",
        f"{volume_result['actual_scores'].iloc[0]['WMAPE %']:.1f}%",
    )
else:
    col7.metric("Actual-Only WMAPE", "N/A")

recent_increment = float(weekly["Incremental_Units"].tail(min(4, len(weekly))).median())
col8.metric("Recent Median Increment", f"{recent_increment:+,.0f}")


# ============================================================
# BASELINE / ACTUAL DECOMPOSITION
# ============================================================
st.markdown("---")
st.subheader("🧩 Actual vs Baseline Demand")

baseline_total_recent = float(weekly["Baseline_Units"].tail(4).mean())
actual_total_recent = float(weekly["Sales_Units"].tail(4).mean())
recent_increment_avg = actual_total_recent - baseline_total_recent

b1, b2, b3, b4 = st.columns(4)
b1.metric("Latest Week Actual Units", f"{weekly['Sales_Units'].iloc[-1]:,.0f}")
b2.metric("Latest Week Baseline Units", f"{weekly['Baseline_Units'].iloc[-1]:,.0f}")
b3.metric("Latest Week Incremental Units", f"{weekly['Incremental_Units'].iloc[-1]:+,.0f}")
b4.metric("Recent 4-Wk Avg Increment", f"{recent_increment_avg:+,.0f}")

st.caption(
    "Actual Units = observed sales value ÷ average RSP. Baseline Units come directly from 52 Weeks CY Sales Baseline. "
    "Incremental Units come from 52 Weeks CY Sales Incremental when present; otherwise the platform uses Actual Units − Baseline Units as a fallback. "
    "The forecasting engine backtests whether incremental demand improves the final historical forecast before carrying it forward."
)


# ============================================================
# SAME PERIOD LAST YEAR
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
    forecast_units_total = float(forecast_table["Selected Forecast Units"].sum())
    forecast_value_total = float(forecast_table["Forecast Value"].sum())
    ly_units_total = float(ly["LY Units"].sum())
    ly_baseline_total = float(ly["LY Baseline Units"].sum())
    ly_value_total = float(ly["LY Value"].sum())

    yoy1, yoy2, yoy3, yoy4 = st.columns(4)
    yoy1.metric("Selected Forecast Units", f"{forecast_units_total:,.0f}")
    yoy2.metric("LY Comparable Units", f"{ly_units_total:,.0f}")

    if ly_units_total > 0:
        yoy3.metric(
            "Volume YoY",
            f"{((forecast_units_total / ly_units_total) - 1) * 100:+.1f}%",
        )
    else:
        yoy3.metric("Volume YoY", "N/A")

    yoy4.metric("LY Baseline Units", f"{ly_baseline_total:,.0f}")

    st.caption(
        "LY values are actual observations from approximately 52 weeks earlier. They are not simulated from the forecast."
    )

    if volume_result["baseline"]["seasonal_benchmark"] is not None:
        benchmark_total = float(np.sum(volume_result["baseline"]["seasonal_benchmark"]))
        st.write(
            f"**52-week same-period baseline benchmark: {benchmark_total:,.0f} units**"
        )


# ============================================================
# MODEL SELECTION
# ============================================================
st.markdown("---")
st.subheader("🧠 Model Selection & Backtesting")

mcol1, mcol2 = st.columns(2)

with mcol1:
    st.markdown("**Baseline demand models**")
    if volume_result["baseline"]["scores"].empty:
        st.info("Not enough history for meaningful baseline model comparison.")
    else:
        table = volume_result["baseline"]["scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(volume_result["baseline"]["champion"]))
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
    st.markdown("**Actual-demand models (benchmark)**")
    if volume_result["actual_scores"].empty:
        st.info("Not enough history for meaningful actual-demand model comparison.")
    else:
        table = volume_result["actual_scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(volume_result["actual_champion"]))
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
    f"Rolling-origin backtest: {volume_result['bt_horizon']}-week horizon, "
    f"minimum training history of {volume_result['min_train']} weeks, "
    f"{volume_result['n_origins']} validation origins. Champion = lowest WMAPE, "
    "with absolute bias used as the tie-breaker."
)

if volume_result["baseline"]["fallback_reason"]:
    st.warning(volume_result["baseline"]["fallback_reason"])


# ============================================================
# DECOMPOSED MODEL DIAGNOSTIC
# ============================================================
if not volume_result["decomp_scores"].empty:
    st.markdown("---")
    st.subheader("🧪 Baseline + Incremental Validation")

    dtable = volume_result["decomp_scores"].copy()
    dtable.insert(0, "Selected", dtable["Model"].eq(volume_result["decomp_champion"]))

    st.dataframe(
        dtable[
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

    if volume_result["decomp_forecast"] is not None:
        st.caption(
            f"Best baseline + incremental historical model: {volume_result['decomp_champion']}. "
            "The model-selected option uses this combination when available; the other options are provided for transparent comparison."
        )


# ============================================================
# COMMERCIAL SIGNALS
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
    "These are observed historical movements. They are not causal elasticity estimates. The platform does not claim that a specific "
    "price, promotion or distribution change will produce a specific unit uplift."
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
            "Baseline Forecast Units": "{:,.0f}",
            "Selected Forecast Units": "{:,.0f}",
            "Units Lower": "{:,.0f}",
            "Units Upper": "{:,.0f}",
            "Forecast Value": "R {:,.0f}",
            "Value Lower": "R {:,.0f}",
            "Value Upper": "R {:,.0f}",
            "Recent Incremental Scenario Units": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Baseline volume model: {volume_result['baseline']['champion']}. "
    f"Production volume basis: {forecast_basis_label}. "
    f"Volume interval: empirical historical errors where available."
)
st.caption(
    f"Value model: {value_result['champion']}. "
    f"Value interval: {value_result['interval_method']}"
)


# ============================================================
# HISTORICAL + FORECAST CHART
# ============================================================
st.markdown("---")
st.subheader("📈 Actual vs Baseline vs Forecast")

metric = st.radio(
    "Chart metric",
    ["Volume (Units)", "Value (R)"],
    horizontal=True,
)

fig = go.Figure()

if metric.startswith("Volume"):
    # Actual history
    fig.add_trace(
        go.Scatter(
            x=weekly["date_key"],
            y=weekly["Sales_Units"],
            mode="lines+markers",
            name="Actual Units",
            line=dict(width=2.5),
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
            line=dict(width=2, dash="dot"),
            marker=dict(size=4),
        )
    )

    # Baseline forecast anchor
    anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
    anchor_y = [float(weekly["Baseline_Units"].iloc[-1])] + list(selected_volume_forecast)

    fig.add_trace(
        go.Scatter(
            x=anchor_x,
            y=anchor_y,
            mode="lines+markers",
            name=f"Forecast ({volume_result['baseline']['champion']})",
            line=dict(width=3, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    # Forecast interval
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(selected_volume_upper) + list(selected_volume_lower[::-1]),
            fill="toself",
            fillcolor="rgba(30, 58, 138, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Baseline Forecast Range",
        )
    )

    # LY benchmark
    if volume_result["baseline"]["seasonal_benchmark"] is not None:
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=volume_result["baseline"]["seasonal_benchmark"],
                mode="lines+markers",
                name="52-Week LY Benchmark",
                line=dict(width=2, dash="dot"),
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
            name=f"Value Forecast ({value_result['champion']})",
            line=dict(width=3, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(value_result["upper"]) + list(value_result["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(30, 58, 138, 0.12)",
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
# BASELINE / INCREMENTAL HISTORY TABLE
# ============================================================
st.markdown("---")
st.subheader("📊 Recent Baseline & Commercial Increment")

recent = weekly.tail(12).copy()
recent["Week"] = recent["date_key"].dt.strftime("%Y-%m-%d")

# Use the actual internal weekly column names first, then rename them
# only for presentation. This prevents a KeyError when the source
# dataframe uses underscores internally.
recent = recent[
    [
        "Week",
        "Sales_Units",
        "Baseline_Units",
        "Incremental_Units",
        "Source_Incremental_Units",
        "Incremental_Reconciliation_Gap",
        "Sales_Value",
        "Effective_RSP",
        "Promo_Depth_%",
        "Distribution",
    ]
].rename(
    columns={
        "Sales_Units": "Actual Units",
        "Baseline_Units": "Baseline Units",
        "Incremental_Units": "Incremental Units",
        "Source_Incremental_Units": "Source Incremental Units",
        "Incremental_Reconciliation_Gap": "Incremental QA Gap",
        "Sales_Value": "Sales Value",
        "Effective_RSP": "Effective RSP",
        "Promo_Depth_%": "Promo Depth %",
        "Distribution": "Numeric Distribution %",
    }
)

st.dataframe(
    recent.style.format(
        {
            "Actual Units": "{:,.0f}",
            "Baseline Units": "{:,.0f}",
            "Incremental Units": "{:+,.0f}",
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
        **Primary volume forecast**

        The platform now treats **52 Weeks CY Sales Baseline** as the underlying weekly demand signal.
        It backtests multiple forecasting methods on that baseline and selects the lowest-WMAPE model.

        **Current baseline champion:** `{volume_result['baseline']['champion']}`

        **Current production volume model:** `{volume_result['decomp_champion'] or volume_result['baseline']['champion']}`

        Candidate methods include Naive, moving averages, weighted moving average, median,
        Simple Exponential Smoothing, Log Exponential Smoothing, Damped Holt, Theta and
        Croston-SBA for intermittent demand. A 52-week Seasonal Naive benchmark is shown once
        52 weeks exist and is only made eligible for automatic model selection after sufficient
        post-year history exists for rolling validation.

        **Actual-vs-baseline decomposition**

        `Incremental Units` come from **52 Weeks CY Sales Incremental** when that source field is present.
        The platform also calculates `Actual Units - Baseline Units` as a reconciliation check.

        The forecasting engine backtests combinations of baseline and incremental forecasting methods and
        only uses an incremental component in the model-selected production forecast when that combination
        improves historical WMAPE. A transparent recent-increment scenario is available separately.

        **Important:** the platform does not invent Last Year sales, random forecast noise, or fixed
        promotional uplift percentages.
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
    st.write("Baseline field: **52 Weeks CY Sales Baseline**")
    st.write(f"Incremental field used: **{'52 Weeks CY Sales Incremental' if '52 Weeks CY Sales Incremental' in final_df.columns else 'Actual Units − Baseline fallback'}**")
    st.write("Columns available in the extract:")
    st.write(list(final_df.columns))
