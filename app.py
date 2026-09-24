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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA CONTRACT
# ============================================================
REQUIRED_COLUMNS = [
    "26 Weeks CY Value",
    "Full Date",
    "Brand",
    "Category",
    "Product",
    "ProductsID",
    "Subcategory",
    "26 Weeks CY Ave Price Quantity",
    "26 Weeks CY Ave RSP On Promo",
]


# ============================================================
# HELPERS
# ============================================================
def clean_number(series):
    """Convert numeric-like values safely."""
    return pd.to_numeric(series, errors="coerce")


def validate_and_prepare(raw_df):
    """Validate and create clean row-level analytical fields."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        return False, [f"Missing required columns: {', '.join(missing)}"], [], None

    df = raw_df.copy()
    errors = []
    warnings_list = []

    # Date
    df["date_key"] = pd.to_datetime(df["Full Date"], errors="coerce")
    bad_dates = int(df["date_key"].isna().sum())
    if bad_dates:
        warnings_list.append(f"{bad_dates:,} rows have invalid dates and were removed.")
        df = df.dropna(subset=["date_key"]).copy()

    if df.empty:
        return False, ["No valid dated rows remain after date validation."], warnings_list, None

    # Numeric fields
    df["Sales Value"] = clean_number(df["26 Weeks CY Value"]).fillna(0).clip(lower=0)
    df["Ave RSP"] = clean_number(df["26 Weeks CY Ave Price Quantity"])
    df["Promo RSP"] = clean_number(df["26 Weeks CY Ave RSP On Promo"])

    # IMPORTANT: do not round units row-by-row. Keep decimals and round only for display.
    df["Sales Units"] = np.where(
        df["Ave RSP"] > 0,
        df["Sales Value"] / df["Ave RSP"],
        np.nan,
    )
    df["Sales Units"] = (
        pd.Series(df["Sales Units"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
        .to_numpy()
    )

    # Optional Numeric Distribution. We do not invent a value if it is absent.
    if "Numeric Distribution" in df.columns:
        dist = clean_number(df["Numeric Distribution"])
        non_null = dist.dropna()
        if not non_null.empty and non_null.max() <= 1.0:
            dist = dist * 100
        df["Numeric Distribution"] = dist.clip(lower=0, upper=100)
    else:
        df["Numeric Distribution"] = np.nan
        warnings_list.append(
            "Numeric Distribution is not present in this file; distribution will be shown as unavailable."
        )

    for col in ["Category", "Subcategory", "Brand", "Product"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["ProductsID"] = df["ProductsID"].astype(str)

    # Diagnostics only; not directly fed into the baseline forecast.
    df["Promo Depth %"] = np.where(
        df["Ave RSP"] > 0,
        ((df["Ave RSP"] - df["Promo RSP"]) / df["Ave RSP"]) * 100,
        np.nan,
    )
    df["Promo Depth %"] = (
        df["Promo Depth %"]
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0, upper=100)
    )

    return True, errors, warnings_list, df.sort_values("date_key").reset_index(drop=True)


def aggregate_weekly(df):
    """
    The source is already weekly. Group by the actual Full Date instead of
    resampling into artificial week buckets.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("date_key", as_index=False)
        .agg(
            Sales_Units=("Sales Units", "sum"),
            Sales_Value=("Sales Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
            Avg_RSP=("Ave RSP", "mean"),
            Promo_RSP=("Promo RSP", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    # Scope-level effective selling price. This is much better than averaging
    # SKU prices equally when SKUs have very different volumes.
    grouped["Effective_RSP"] = np.where(
        grouped["Sales_Units"] > 0,
        grouped["Sales_Value"] / grouped["Sales_Units"],
        np.nan,
    )

    grouped["Promo_Depth_%"] = np.where(
        (grouped["Avg_RSP"] > 0) & grouped["Promo_RSP"].notna(),
        ((grouped["Avg_RSP"] - grouped["Promo_RSP"]) / grouped["Avg_RSP"]) * 100,
        np.nan,
    )
    grouped["Promo_Depth_%"] = grouped["Promo_Depth_%"].clip(0, 100)

    return grouped


def is_constant_or_empty(y):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return True
    if np.allclose(y, 0):
        return True
    return np.nanstd(y) < 1e-10


# ============================================================
# FORECAST CANDIDATES
# ============================================================
def forecast_naive(y, horizon):
    y = np.asarray(y, dtype=float)
    return np.repeat(max(0.0, y[-1]), horizon)


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
        fit = SimpleExpSmoothing(np.log1p(np.maximum(y, 0)), initialization_method="estimated").fit(
            optimized=True
        )
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
        raise ValueError("Need at least 52 observations for seasonal-naive-52.")
    out = []
    for i in range(horizon):
        out.append(max(0.0, y[-52 + i]))
    return np.asarray(out, dtype=float)


def forecast_croston_sba(y, horizon, alpha=0.10):
    """Simple Croston-SBA for intermittent SKU demand."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.zeros(horizon)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(horizon)

    first = int(nz[0])
    demand_est = float(y[first])
    interval_est = float(max(1, first + 1))
    last_event = first

    for t in range(first + 1, len(y)):
        if y[t] > 0:
            interval = float(max(1, t - last_event))
            demand_est = demand_est + alpha * (y[t] - demand_est)
            interval_est = interval_est + alpha * (interval - interval_est)
            last_event = t

    forecast = (1.0 - alpha / 2.0) * demand_est / max(interval_est, 1e-9)
    return np.repeat(max(0.0, forecast), horizon)


def get_model_registry(n_obs, intermittent=False):
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

    if n_obs >= 56:
        # A 4-week rolling backtest needs at least 52 weeks of training
        # plus the 4-week validation horizon before Seasonal Naive can
        # be evaluated fairly.
        models["Seasonal Naive (52 Weeks)"] = forecast_seasonal_naive52

    if intermittent:
        models["Croston-SBA"] = forecast_croston_sba

    return models


# ============================================================
# BACKTESTING
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
    denominator = np.abs(actual) + np.abs(pred)
    terms = np.where(denominator > 0, 2 * np.abs(actual - pred) / denominator, 0)
    return 100.0 * np.mean(terms)


def select_backtest_setup(n_obs, requested_horizon=4):
    """Use a true 4-week horizon when enough data exists."""
    horizon = min(requested_horizon, max(1, n_obs // 4))

    if n_obs >= 20:
        min_train = 12
    elif n_obs >= 16:
        min_train = 10
    elif n_obs >= 12:
        min_train = 8
    else:
        min_train = max(4, n_obs - horizon - 1)

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
                    "AbsError": float(abs(p - a)),
                    "AbsPctError": float(abs(p - a) / a) if a > 0 else np.nan,
                    "RelError": float((p - a) / a) if a > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def evaluate_models(y, horizon=4):
    """Evaluate candidate models and select the champion by WMAPE."""
    y = np.asarray(y, dtype=float)
    intermittent = np.mean(y <= 0) >= 0.20
    models = get_model_registry(len(y), intermittent=intermittent)
    actual_horizon, min_train, n_origins = select_backtest_setup(len(y), horizon)

    score_rows = []
    detail_rows = []

    if n_origins <= 0:
        return pd.DataFrame(), pd.DataFrame(), None, actual_horizon, min_train, n_origins

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

    # Primary: WMAPE. Tie-break: absolute bias.
    scores["Abs Bias"] = scores["Bias %"].abs()
    scores = scores.sort_values(["WMAPE %", "Abs Bias"]).reset_index(drop=True)
    champion = scores.iloc[0]["Model"]

    return scores, details_all, champion, actual_horizon, min_train, n_origins


def build_prediction_intervals(champion_details, future_forecast):
    """Use empirical relative errors from backtests, horizon by horizon."""
    f = np.asarray(future_forecast, dtype=float)
    lower = np.zeros(len(f))
    upper = np.zeros(len(f))

    global_errors = champion_details["RelError"].dropna().to_numpy()

    for i in range(len(f)):
        h = i + 1
        horizon_errors = champion_details.loc[
            champion_details["Horizon"] == h, "RelError"
        ].dropna().to_numpy()

        errors = horizon_errors if len(horizon_errors) >= 5 else global_errors

        if len(errors) >= 5:
            q10 = np.quantile(errors, 0.10)
            q90 = np.quantile(errors, 0.90)
        else:
            q10, q90 = -0.15, 0.15

        lower[i] = max(0.0, f[i] * (1.0 + q10))
        upper[i] = max(lower[i], f[i] * (1.0 + q90))

    return lower, upper


# ============================================================
# MODEL FORECAST
# ============================================================
def run_forecast(series, horizon=4):
    """Backtest, select champion, fit champion on all history and forecast."""
    y = np.asarray(series, dtype=float)
    scores, details, champion, bt_horizon, min_train, n_origins = evaluate_models(y, horizon=horizon)

    # If too little data for selection, use a transparent fallback.
    if champion is None:
        champion = "Naive Last Week"
        model_func = forecast_naive
        fallback_reason = "Insufficient history for meaningful model comparison."
    else:
        model_func = get_model_registry(
            len(y), intermittent=np.mean(y <= 0) >= 0.20
        )[champion]
        fallback_reason = None

    future = np.asarray(model_func(y, horizon), dtype=float)
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
    }


# ============================================================
# DRIVER DIAGNOSTICS (FACTUAL, NOT CAUSAL)
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
        items.append(("Volume", f"Latest 4-week average volume is {unit_change:+.1f}% vs the preceding 4 weeks."))
    if np.isfinite(price_change):
        items.append(("Price", f"Effective RSP changed {price_change:+.1f}% on a 4-week-vs-4-week basis."))
    if np.isfinite(promo_change):
        items.append(("Promotion", f"Observed promotional depth changed {promo_change:+.1f}% vs the preceding 4 weeks."))
    if np.isfinite(dist_change):
        items.append(("Distribution", f"Numeric distribution changed {dist_change:+.1f}% vs the preceding 4 weeks."))

    return items


# ============================================================
# UI
# ============================================================
st.markdown('<div class="main-header">Unilever Demand & Forecast Platform</div>', unsafe_allow_html=True)
st.caption(
    "Automatic model selection using rolling-origin backtesting. "
    "The platform does not manufacture YoY benchmarks or promotional uplift assumptions."
)

with st.sidebar:
    st.header("📥 Data & Filters")
    uploaded_file = st.file_uploader("Upload weekly sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload the weekly CSV to start the forecast.")
    st.markdown(
        """
        **Minimum recommended history:** 12 weeks.

        **Preferred:** 52+ weeks if you want true same-period-last-year comparison and annual seasonality.
        """
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

valid, errors, warning_list, df_clean = validate_and_prepare(raw_df)

if not valid:
    st.error("Schema validation failed.")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

for w in warning_list:
    st.warning(w)

# --------------------------
# Sidebar filters
# --------------------------
with st.sidebar:
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
        product_label = ", ".join(selected_products) if len(selected_products) <= 3 else f"{len(selected_products)} selected SKUs"
    else:
        final_df = d3.copy()
        product_label = "All Products"

    forecast_horizon = st.slider("Forecast horizon (weeks)", 1, 8, 4)

    st.markdown("---")
    st.caption(f"Source rows: {len(final_df):,}")
    st.caption(f"Date range: {df_clean['date_key'].min().date()} → {df_clean['date_key'].max().date()}")

if final_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

weekly = aggregate_weekly(final_df)

if len(weekly) < 4:
    st.error("At least 4 weekly observations are required for a short-term forecast.")
    st.stop()

# --------------------------
# Forecast units and value separately
# --------------------------
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

# --------------------------
# Header / scope
# --------------------------
scope_label = f"{category} > {subcategory} > {brand} > {product_label}"
st.markdown(f"### Forecast Scope\n**{scope_label}**")

# --------------------------
# KPIs
# --------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Volume", f"{forecast_table['Forecast Units'].sum():,.0f}")
col2.metric("Forecast Value", f"R {forecast_table['Forecast Value'].sum():,.0f}")
col3.metric("Volume Champion", unit_result["champion"])
col4.metric("Value Champion", value_result["champion"])

col5, col6, col7, col8 = st.columns(4)
col5.metric("History Used", f"{len(weekly)} weeks")
col6.metric(
    "Volume WMAPE",
    f"{unit_result['scores'].iloc[0]['WMAPE %']:.1f}%" if not unit_result["scores"].empty else "N/A",
)
col7.metric(
    "Value WMAPE",
    f"{value_result['scores'].iloc[0]['WMAPE %']:.1f}%" if not value_result["scores"].empty else "N/A",
)
col8.metric("Latest Weekly Units", f"{weekly['Sales_Units'].iloc[-1]:,.0f}")

# --------------------------
# YoY comparison
# --------------------------
st.markdown("---")
st.subheader("📅 Comparable Period vs Last Year")

if len(weekly) >= 52:
    ly_units = []
    ly_value = []

    for fd in future_dates:
        target = fd - timedelta(weeks=52)
        distance = (weekly["date_key"] - target).abs().dt.days
        idx = distance.idxmin()
        if distance.loc[idx] <= 7:
            ly_units.append(weekly.loc[idx, "Sales_Units"])
            ly_value.append(weekly.loc[idx, "Sales_Value"])

    if len(ly_units) == len(future_dates):
        ly_units_total = float(np.sum(ly_units))
        ly_value_total = float(np.sum(ly_value))
        fc_units = float(forecast_table["Forecast Units"].sum())
        fc_value = float(forecast_table["Forecast Value"].sum())

        yoy1, yoy2, yoy3, yoy4 = st.columns(4)
        yoy1.metric("Forecast Units", f"{fc_units:,.0f}")
        yoy2.metric("LY Comparable Units", f"{ly_units_total:,.0f}")
        yoy3.metric("Volume YoY", f"{((fc_units / ly_units_total) - 1) * 100:+.1f}%")
        yoy4.metric("Value YoY", f"{((fc_value / ly_value_total) - 1) * 100:+.1f}%")
    else:
        st.info("52-week comparable dates could not be matched cleanly for every forecast week.")
else:
    st.info(
        f"Actual YoY is not calculated because this scope contains only {len(weekly)} weeks. "
        "The platform will not manufacture a Last Year benchmark."
    )

# --------------------------
# Model performance
# --------------------------
st.markdown("---")
st.subheader("🧠 Model Selection & Backtesting")

mcol1, mcol2 = st.columns(2)

with mcol1:
    st.markdown("**Volume models**")
    if unit_result["scores"].empty:
        st.info("Not enough history for model comparison.")
    else:
        table = unit_result["scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(unit_result["champion"]))
        st.dataframe(
            table[["Selected", "Model", "WMAPE %", "sMAPE %", "Bias %", "Forecasts Tested"]].style.format(
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
        st.info("Not enough history for model comparison.")
    else:
        table = value_result["scores"].copy()
        table.insert(0, "Selected", table["Model"].eq(value_result["champion"]))
        st.dataframe(
            table[["Selected", "Model", "WMAPE %", "sMAPE %", "Bias %", "Forecasts Tested"]].style.format(
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
    f"minimum training history of {unit_result['min_train']} weeks, and {unit_result['n_origins']} validation origins. "
    "The champion is selected by lowest WMAPE, with absolute bias used as the tie-breaker."
)

# --------------------------
# Forecast detail
# --------------------------
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

st.caption(f"Volume interval: {unit_result['interval_method']}")
st.caption(f"Value interval: {value_result['interval_method']}")

# --------------------------
# Chart
# --------------------------
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

# Add anchor so forecast visually connects to the last actual.
anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
anchor_y = [hist_y.iloc[-1]] + list(future_y)

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

fig.add_vline(
    x=weekly["date_key"].iloc[-1],
    line_dash="dot",
    line_width=1,
)

fig.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    xaxis_title="Week",
    yaxis_title=y_title,
    legend=dict(orientation="h", y=1.03, x=1, xanchor="right"),
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Driver facts
# --------------------------
st.markdown("---")
st.subheader("🔎 Recent Commercial Signals")

signals = build_driver_summary(weekly)
if signals:
    for signal_type, text_value in signals:
        st.markdown(
            f'<div class="card"><strong>{signal_type}</strong><br>{text_value}</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Not enough observations to calculate recent driver comparisons.")

# --------------------------
# Recent history table
# --------------------------
st.markdown("---")
st.subheader("📊 Recent Weekly History")

recent = weekly.tail(8).copy()
recent = recent[
    [
        "date_key",
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
recent["Week"] = recent["Week"].dt.strftime("%Y-%m-%d")

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

# --------------------------
# Methodology
# --------------------------
with st.expander("ℹ️ Forecast Methodology"):
    st.markdown(
        f"""
        **Volume champion:** {unit_result['champion']}  
        **Value champion:** {value_result['champion']}  

        **How the platform works**

        1. The source is treated as weekly observations using `Full Date`.
        2. Sales Units are calculated as Sales Value / Ave RSP without rounding at row level.
        3. Candidate forecasting models are backtested using rolling historical cut-offs.
        4. WMAPE is the primary model-selection metric because it is more robust than ordinary MAPE when demand is low or zero.
        5. The best-performing model is refit using all available history and used for the future forecast.
        6. Forecast ranges are estimated from historical forecast errors; no arbitrary ±12% band is hard-coded.
        7. YoY is shown only when actual comparable history is present.
        8. Price, promotion and distribution are shown as commercial diagnostics. They are not treated as causal drivers unless future driver assumptions are explicitly supplied and modelled.

        **Current source history:** {len(weekly)} weekly observations.
        """
    )

    if unit_result["fallback_reason"]:
        st.warning(unit_result["fallback_reason"])
