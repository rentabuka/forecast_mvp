import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Unilever FMCG Demand & Prescriptive Engine",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 24px; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;}
    
    /* Reduced KPI Font Styling */
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    .alert-card {padding: 16px; border: 1px solid #FCA5A5; border-radius: 8px; background: #FEF2F2; margin-bottom: 16px;}
    .solution-box {background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 12px 16px; margin-top: 8px; border-radius: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DATA CONTRACT & SCHEMA DETECTOR
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
# 3. PREPROCESSING & FOOLPROOF SCALE ENFORCEMENT ENGINE
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
    """Validate raw CSV extract and create clean, auto-scaled analytical fields."""
    errors = []
    warnings_list = []

    missing_base = [c for c in BASE_COLUMNS if c not in raw_df.columns]
    if missing_base:
        errors.append("Missing required columns: " + ", ".join(missing_base))
        return False, errors, warnings_list, None, None

    measure_window, measures, missing_measure = detect_measure_set(raw_df.columns)

    if measure_window == "52-week-missing-baseline":
        errors.append(
            f"Required baseline column '{missing_measure}' is missing from the extract."
        )
        return False, errors, warnings_list, None, None

    if measures is None:
        errors.append("Could not find the required sales measure columns.")
        return False, errors, warnings_list, None, None

    df = raw_df.copy()

    # Parse Dates
    df["date_key"] = pd.to_datetime(df["Full Date"], errors="coerce")
    bad_dates = int(df["date_key"].isna().sum())
    if bad_dates:
        warnings_list.append(f"{bad_dates:,} rows have invalid dates and were removed.")
        df = df.dropna(subset=["date_key"]).copy()

    if df.empty:
        return False, ["No valid dated rows remain after date validation."], warnings_list, None, None

    # Pricing & Revenue
    df["Sales Value"] = clean_number(df[measures["value"]]).fillna(0).clip(lower=0)
    df["Ave RSP"] = clean_number(df[measures["price"]])
    df["Promo RSP"] = clean_number(df[measures["promo"]])

    # Weekly Actual Volume Units = Sales Value / Ave RSP
    df["Sales Units"] = np.where(
        df["Ave RSP"] > 0,
        df["Sales Value"] / df["Ave RSP"],
        np.nan,
    )
    df["Sales Units"] = pd.to_numeric(df["Sales Units"], errors="coerce").fillna(0).clip(lower=0)

    # Baseline & Incremental Processing
    #
    # The old logic guessed the baseline scale from broad ratio bands.
    # That can accidentally interpret a value/annual field as weekly units,
    # producing forecasts many times larger than actual demand.
    if measures["baseline"] is not None and measures["baseline"] in df.columns:
        raw_b = (
            clean_number(df[measures["baseline"]])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

        valid_mask = (
            (df["Sales Units"] > 0)
            & (df["Ave RSP"] > 0)
            & (raw_b > 0)
        )

        if valid_mask.any():
            candidates = {
                "weekly_units": raw_b,
                "annual_units_div_52": raw_b / 52.0,
                "weekly_value_to_units": raw_b / df["Ave RSP"],
                "annual_value_div_52_to_units": raw_b / (52.0 * df["Ave RSP"]),
            }

            candidate_stats = []
            actual = df.loc[valid_mask, "Sales Units"].astype(float)
            for mode, candidate in candidates.items():
                test = pd.to_numeric(candidate.loc[valid_mask], errors="coerce")
                ratio = (test / actual).replace([np.inf, -np.inf], np.nan).dropna()
                ratio = ratio[ratio > 0]
                if ratio.empty:
                    continue
                median_ratio = float(ratio.median())
                score = abs(np.log(max(median_ratio, 1e-9)))
                if median_ratio < 0.35 or median_ratio > 3.0:
                    score += 2.0
                candidate_stats.append((score, mode, median_ratio))

            if candidate_stats:
                _, chosen_mode, chosen_ratio = min(candidate_stats, key=lambda x: x[0])
                baseline_units = candidates[chosen_mode]

                if chosen_mode == "weekly_value_to_units":
                    baseline_value = raw_b
                elif chosen_mode == "annual_value_div_52_to_units":
                    baseline_value = raw_b / 52.0
                else:
                    baseline_value = baseline_units * df["Ave RSP"]

                df["Baseline Units"] = baseline_units
                df["Baseline Value"] = baseline_value

                mode_labels = {
                    "weekly_units": "source treated as weekly units",
                    "annual_units_div_52": "source treated as 52-week cumulative units and divided by 52",
                    "weekly_value_to_units": "source treated as weekly Rand value and divided by RSP",
                    "annual_value_div_52_to_units": "source treated as 52-week cumulative Rand value and divided by 52 and RSP",
                }
                warnings_list.append(
                    f"Baseline scaling: {mode_labels[chosen_mode]}. "
                    f"Median Baseline/Actual ratio = {chosen_ratio:.2f}x."
                )
            else:
                df["Baseline Units"] = df["Sales Units"]
                df["Baseline Value"] = df["Sales Value"]
                warnings_list.append("Baseline could not be reliably scaled; actual sales units are being used as the baseline.")
        else:
            df["Baseline Units"] = df["Sales Units"]
            df["Baseline Value"] = df["Sales Value"]
            warnings_list.append("Baseline had no usable overlap with actual sales; actual sales units are being used as the baseline.")

        df["Baseline Units"] = pd.to_numeric(df["Baseline Units"], errors="coerce").fillna(0).clip(lower=0)
        df["Baseline Value"] = pd.to_numeric(df["Baseline Value"], errors="coerce").fillna(0).clip(lower=0)

        df["Calculated Incremental Units"] = df["Sales Units"] - df["Baseline Units"]

        incremental_col = measures.get("incremental")
        if incremental_col and incremental_col in df.columns:
            raw_inc = clean_number(df[incremental_col]).replace([np.inf, -np.inf], np.nan)
            df["Source Incremental Units"] = raw_inc
            # Always use the coherent weekly calculation for downstream aggregation.
            df["Incremental Units"] = df["Calculated Incremental Units"]
        else:
            df["Source Incremental Units"] = np.nan
            df["Incremental Units"] = df["Calculated Incremental Units"]
    else:
        # Fallback for 26-week extracts missing explicit baseline columns
        df["Baseline Units"] = df["Sales Units"]
        df["Baseline Value"] = df["Sales Value"]
        df["Source Incremental Units"] = 0.0
        df["Calculated Incremental Units"] = 0.0
        df["Incremental Units"] = 0.0

    # Distribution & Dimensions
    if "Numeric Distribution" in df.columns:
        dist = clean_number(df["Numeric Distribution"])
        if not dist.dropna().empty and dist.dropna().max() <= 1.0:
            dist = dist * 100
        df["Numeric Distribution"] = dist.clip(0, 100)
    else:
        df["Numeric Distribution"] = 85.0

    for col in ["Category", "Subcategory", "Brand", "Product"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["ProductsID"] = df["ProductsID"].fillna("Unknown").astype(str)

    return True, errors, warnings_list, df.sort_values("date_key").reset_index(drop=True), measure_window


@st.cache_data(show_spinner=False)
def prepare_csv(csv_bytes):
    raw = pd.read_csv(pd.io.common.BytesIO(csv_bytes), low_memory=False)
    return validate_and_prepare(raw)


# ============================================================
# 4. SAFE WEEKLY AGGREGATION ENGINE
# ============================================================
def aggregate_weekly(df):
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()

    # Defensively ensure required columns exist to avoid KeyErrors
    for col, default_val in [
        ("Sales Units", 0.0),
        ("Sales Value", 0.0),
        ("Baseline Units", 0.0),
        ("Baseline Value", 0.0),
        ("Incremental Units", 0.0),
        ("Numeric Distribution", 85.0),
    ]:
        if col not in df_copy.columns:
            df_copy[col] = default_val

    base = (
        df_copy.groupby("date_key", as_index=False)
        .agg(
            Sales_Units=("Sales Units", "sum"),
            Baseline_Units=("Baseline Units", "sum"),
            Incremental_Units=("Incremental Units", "sum"),
            Sales_Value=("Sales Value", "sum"),
            Baseline_Value=("Baseline Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    base["Incremental_Units"] = base["Sales_Units"] - base["Baseline_Units"]
    base["Effective_RSP"] = np.where(base["Sales_Units"] > 0, base["Sales_Value"] / base["Sales_Units"], np.nan)

    if "Promo RSP" in df_copy.columns:
        promo_part = df_copy.copy()
        promo_part["Promo_RSP"] = pd.to_numeric(promo_part["Promo RSP"], errors="coerce")
        promo_part = promo_part[(promo_part["Promo_RSP"] > 0) & (promo_part["Sales Units"] > 0)].copy()

        if not promo_part.empty:
            promo_part["Promo_Value_Proxy"] = promo_part["Promo_RSP"] * promo_part["Sales Units"]
            promo_week = promo_part.groupby("date_key").agg(Promo_Value_Proxy=("Promo_Value_Proxy", "sum"), Promo_Units=("Sales Units", "sum")).reset_index()
            promo_week["Promo_RSP_Weighted"] = np.where(promo_week["Promo_Units"] > 0, promo_week["Promo_Value_Proxy"] / promo_week["Promo_Units"], np.nan)
            base = base.merge(promo_week[["date_key", "Promo_RSP_Weighted"]], on="date_key", how="left")
        else:
            base["Promo_RSP_Weighted"] = np.nan
    else:
        base["Promo_RSP_Weighted"] = np.nan

    base["Promo_Depth_%"] = np.where(
        (base["Effective_RSP"] > 0) & base["Promo_RSP_Weighted"].notna(),
        ((base["Effective_RSP"] - base["Promo_RSP_Weighted"]) / base["Effective_RSP"]) * 100,
        np.nan,
    )
    base["Promo_Depth_%"] = base["Promo_Depth_%"].replace([np.inf, -np.inf], np.nan).clip(0, 100)

    return base


# ============================================================
# 5. FORECASTING MODELS & BACKTESTING
# ============================================================
def forecast_naive(y, horizon):
    return np.repeat(max(0.0, float(y[-1])), horizon)

def forecast_ma4(y, horizon):
    n = min(4, len(y))
    return np.repeat(max(0.0, float(np.mean(y[-n:]))), horizon)

def forecast_ses(y, horizon):
    if len(y) == 0 or np.nanstd(y) < 1e-10:
        return forecast_naive(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))

def forecast_damped_holt(y, horizon):
    if len(y) < 5 or np.nanstd(y) < 1e-10:
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Holt(y, initialization_method="estimated", damped_trend=True).fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))

def base_model_registry():
    return {
        "Naive Last Week": forecast_naive,
        "4-Week Moving Average": forecast_ma4,
        "Simple Exponential Smoothing": forecast_ses,
        "Damped Holt": forecast_damped_holt,
    }


def run_baseline_model(series, horizon=4):
    y = np.asarray(series, dtype=float)
    registry = base_model_registry()
    best_model = "4-Week Moving Average"
    best_wmape = float("inf")
    
    min_train = 16 if len(y) >= 24 else 8
    if len(y) > min_train + horizon:
        for name, func in registry.items():
            errors, actuals = [], []
            for split in range(min_train, len(y) - horizon + 1):
                train, actual = y[:split], y[split : split + horizon]
                try:
                    pred = func(train, horizon)
                    errors.extend(np.abs(pred - actual))
                    actuals.extend(actual)
                except Exception:
                    continue
            if actuals and np.sum(actuals) > 0:
                wmape_val = (np.sum(errors) / np.sum(actuals)) * 100
                if wmape_val < best_wmape:
                    best_wmape = wmape_val
                    best_model = name

    model_func = registry.get(best_model, forecast_ma4)
    forecast_vals = np.maximum(0.0, model_func(y, horizon))
    return forecast_vals, best_model, best_wmape


def choose_forecast_series(weekly):
    """Use the baseline only when its scale is credible versus actual sales."""
    actual = pd.to_numeric(weekly["Sales_Units"], errors="coerce").fillna(0)
    baseline = pd.to_numeric(weekly["Baseline_Units"], errors="coerce").fillna(0)

    mask = (actual > 0) & (baseline > 0)
    if mask.sum() < 8:
        return actual.values, "Actual Sales Units", np.nan

    ratio = (baseline[mask] / actual[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    median_ratio = float(ratio.median()) if not ratio.empty else np.nan

    if np.isfinite(median_ratio) and 0.40 <= median_ratio <= 3.00:
        return baseline.values, "Baseline Units", median_ratio

    return actual.values, "Actual Sales Units (baseline rejected)", median_ratio


# ============================================================
# 6. 52-WEEK PROMOTIONAL OVERLAY & SA MACRO ELASTICITY
# ============================================================
def apply_52wk_promo_overlay_and_sa_macro(
    weekly,
    future_dates,
    base_volume_fcst,
    price_yoy,
    cpi_inflation,
    elasticity,
    apply_promo_overlay=True,
):
    adjusted_fcst_units = []
    promo_lift_factors = []
    ly_benchmark_units = []

    net_price_squeeze = max(0.0, price_yoy - cpi_inflation)
    macro_factor = float(np.clip(1.0 + (elasticity * (net_price_squeeze / 100.0)), 0.5, 1.5))

    for idx, f_date in enumerate(future_dates):
        target_ly_date = f_date - timedelta(weeks=52)
        dist = (weekly["date_key"] - target_ly_date).abs().dt.days
        closest_idx = dist.idxmin()

        lift_factor = 1.0
        if dist.loc[closest_idx] <= 7:
            ly_act_units = float(weekly.loc[closest_idx, "Sales_Units"])
            ly_base_units = float(weekly.loc[closest_idx, "Baseline_Units"])

            raw_lift = (ly_act_units / ly_base_units) if ly_base_units > 0 else 1.0
            # Only use a historical promo lift when the baseline is plausible.
            if apply_promo_overlay and 0.50 <= raw_lift <= 3.00:
                lift_factor = max(1.0, raw_lift)
            ly_benchmark_units.append(ly_act_units)
        else:
            ly_benchmark_units.append(np.nan)

        promo_lift_factors.append(lift_factor)

        # Apply macro price elasticity to base demand. Add promo lift only when
        # forecasting from a validated baseline; otherwise historical promotion
        # is already embedded in the actual-sales forecast and must not be doubled.
        adj_units = base_volume_fcst[idx] * macro_factor
        if apply_promo_overlay:
            effective_lift = 1.0 + ((lift_factor - 1.0) * macro_factor)
            adj_units *= effective_lift

        adjusted_fcst_units.append(max(0.0, float(adj_units)))

    return np.array(adjusted_fcst_units), promo_lift_factors, ly_benchmark_units, macro_factor


# ============================================================
# 7. PRESCRIPTIVE RECOMMENDATION ENGINE
# ============================================================
def generate_gap_closing_recommendations(scope_name, fcst_units_4wk, ly_units_4wk, fcst_val_4wk, ly_val_4wk, avg_rsp, promo_rsp, num_dist, brand_name):
    unit_deficit = ly_units_4wk - fcst_units_4wk
    unit_deficit_pct = (unit_deficit / ly_units_4wk) * 100 if ly_units_4wk > 0 else 0
    revenue_at_risk = max(0.0, ly_val_4wk - fcst_val_4wk)

    actions = []

    promo_discount_pct = ((avg_rsp - promo_rsp) / avg_rsp) * 100 if avg_rsp > 0 else 0
    target_promo_rsp = round(avg_rsp * 0.78, 2)
    est_unit_recovery_promo = int(unit_deficit * 0.60)
    
    actions.append({
        "type": "Pricing & Promotion",
        "title": f"Deepen Promotional RSP to R {target_promo_rsp:.2f} (22% Promo Depth)",
        "detail": f"Current promo depth is running at {promo_discount_pct:.1f}%. Increasing promo depth to 22% (Target Promo RSP: R {target_promo_rsp:.2f}) is projected to recover ~{est_unit_recovery_promo:,} units."
    })

    actions.append({
        "type": "Payday & SASSA Timing",
        "title": "Align Promotional Activations with Month-End Payday (25th - 1st)",
        "detail": "South African FMCG demand is hyper-cyclical around payday and SASSA grant liquidity. Ensure catalogue features and end-cap displays are scheduled during payday weeks."
    })

    if num_dist < 85.0:
        dist_gap = 85.0 - num_dist
        est_unit_recovery_dist = int(unit_deficit * 0.35)
        actions.append({
            "type": "Field Distribution Audit",
            "title": f"Field Sales Push: Audit & Recover +{dist_gap:.1f}% Numeric Distribution",
            "detail": f"Numeric distribution is lagging at {num_dist:.1f}%. Resolving out-of-stocks in Shoprite/Checkers, Pick n Pay, or Wholesale key accounts will recover ~{est_unit_recovery_dist:,} units."
        })

    if any(k in str(brand_name).lower() or k in str(scope_name).lower() for k in ["rexona", "shield"]):
        actions.append({
            "type": "Brand Rebrand Conversion",
            "title": "Deploy Co-Branded 'Shield is now Rexona' Shelf POS",
            "detail": "Shopper confusion post-rebrand is causing conversion leakage. Allocate trade spend to shelf-talkers and secondary bay displays during peak traffic weeks."
        })

    return {
        "unit_deficit": unit_deficit,
        "unit_deficit_pct": unit_deficit_pct,
        "revenue_at_risk": revenue_at_risk,
        "actions": actions
    }


# ============================================================
# 8. STREAMLIT APP LAYOUT & SIDEBAR FILTERS
# ============================================================
st.markdown('<div class="main-header">Unilever FMCG Demand & Prescriptive Forecasting Engine</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 Data Management")
    uploaded_file = st.file_uploader("Upload Weekly Sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload your Unilever sales CSV extract in the left sidebar to start.")
    st.stop()

try:
    csv_bytes = uploaded_file.getvalue()
    valid, errors, warning_list, df_clean, measure_window = prepare_csv(csv_bytes)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

if not valid:
    st.error("Schema validation failed.")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

for warning in warning_list:
    st.sidebar.warning(warning)


# Sidebar Filters & SA Macro Controls
with st.sidebar:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🇿🇦 SA Macroeconomic Controls")
    
    price_yoy = st.sidebar.number_input("Unilever YoY RSP Increase (%)", min_value=0.0, max_value=30.0, value=8.5, step=0.5)
    cpi_inflation = st.sidebar.number_input("SA Food CPI Inflation (%)", min_value=0.0, max_value=30.0, value=5.2, step=0.5)
    elasticity = st.sidebar.slider("Price Elasticity Coefficient (ε)", min_value=-2.5, max_value=-0.1, value=-1.1, step=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Hierarchy Filters")

    categories = ["All"] + sorted(df_clean["Category"].unique().tolist())
    category = st.selectbox("1. Category", categories)
    d1 = df_clean if category == "All" else df_clean[df_clean["Category"] == category]

    subcategories = ["All"] + sorted(d1["Subcategory"].unique().tolist())
    subcategory = st.selectbox("2. Subcategory", subcategories)
    d2 = d1 if subcategory == "All" else d1[d1["Subcategory"] == subcategory]

    brands = ["All"] + sorted(d2["Brand"].unique().tolist())
    brand = st.selectbox("3. Brand", brands)
    d3 = d2 if brand == "All" else d2[d2["Brand"] == brand]

    product_options = ["All"] + sorted(d3["Product"].unique().tolist())
    selected_products = st.multiselect("4. Product SKU(s)", options=product_options, default=["All"])

    if "All" in selected_products or not selected_products:
        final_df = d3.copy()
        product_label = f"All Products in {brand if brand != 'All' else 'Portfolio'}"
    else:
        final_df = d3[d3["Product"].isin(selected_products)].copy()
        product_label = selected_products[0] if len(selected_products) == 1 else f"{len(selected_products)} Selected SKUs"

    forecast_horizon = st.slider("Forecast Horizon (Weeks)", min_value=1, max_value=8, value=4)

if final_df.empty:
    st.warning("⚠️ No data matches the selected filter combination.")
    st.stop()


# ============================================================
# 9. FORECAST COMPUTATION & METRICS
# ============================================================
weekly = aggregate_weekly(final_df)

if len(weekly) < 4:
    st.error("At least 4 weekly observations are required for a short-term forecast.")
    st.stop()

last_date = weekly["date_key"].max()
future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon + 1)]

# Base demand forecast
forecast_series, forecast_source, baseline_ratio = choose_forecast_series(weekly)
if forecast_source != "Baseline Units":
    if np.isfinite(baseline_ratio):
        st.sidebar.warning(
            f"Baseline rejected for forecasting: median Baseline/Actual ratio = {baseline_ratio:.2f}x. "
            "Forecasting actual sales units instead."
        )
    else:
        st.sidebar.warning(
            "Baseline rejected for forecasting: insufficient valid overlap. "
            "Forecasting actual sales units instead."
        )

base_volume_fcst, champion_model, model_wmape = run_baseline_model(
    forecast_series, forecast_horizon
)

# 52-Week Promotional Overlay + SA Macro Elasticity Adjustment
# Do not stack a promo lift on top of an actual-sales forecast because that
# would count historical promotion twice.
apply_promo_overlay = forecast_source == "Baseline Units"
final_volume_fcst, promo_lifts, ly_benchmark_units, macro_factor = apply_52wk_promo_overlay_and_sa_macro(
    weekly,
    future_dates,
    base_volume_fcst,
    price_yoy,
    cpi_inflation,
    elasticity,
    apply_promo_overlay=apply_promo_overlay,
)

avg_rsp_latest = weekly["Effective_RSP"].iloc[-1] if weekly["Effective_RSP"].iloc[-1] > 0 else 25.0
final_value_fcst = final_volume_fcst * avg_rsp_latest

fcst_4wk_units = sum(final_volume_fcst)
fcst_4wk_val = sum(final_value_fcst)

ly_4wk_units = sum(ly_benchmark_units)
ly_4wk_val = ly_4wk_units * avg_rsp_latest

latest_promo_rsp = weekly["Promo_RSP_Weighted"].iloc[-1] if "Promo_RSP_Weighted" in weekly.columns and pd.notna(weekly["Promo_RSP_Weighted"].iloc[-1]) else avg_rsp_latest * 0.82
latest_dist = weekly["Distribution"].iloc[-1] if "Distribution" in weekly.columns else 85.0

scope_label = f"{category} > {subcategory} > {brand} > {product_label}"

diagnostics = generate_gap_closing_recommendations(
    scope_label, fcst_4wk_units, ly_4wk_units, fcst_4wk_val, ly_4wk_val,
    avg_rsp_latest, latest_promo_rsp, latest_dist, brand
)


# ============================================================
# 10. DASHBOARD METRICS & PRESCRIPTIVE PANEL
# ============================================================
st.markdown(f"### Active Scope: **{scope_label}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("4-Wk Volume Forecast", f"{fcst_4wk_units:,.0f} units")
col2.metric("4-Wk Revenue Forecast", f"R {fcst_4wk_val:,.2f}")
col3.metric("YoY Volume Variance", f"-{diagnostics['unit_deficit_pct']:.1f}%", delta=f"-{diagnostics['unit_deficit_pct']:.1f}%", delta_color="inverse")
col4.metric("Revenue at Risk", f"R {diagnostics['revenue_at_risk']:,.2f}")

st.markdown("---")


if diagnostics["unit_deficit_pct"] > 0:
    st.markdown(f"""
    <div class='alert-card'>
        <h4 style='margin:0; color:#991B1B;'>🚨 YoY Deficit Alert: 4-Week Forecast is {diagnostics['unit_deficit_pct']:.1f}% Below Last Year</h4>
        <p style='margin-top:4px; margin-bottom:0; color:#7F1D1D; font-size:14px;'>
            Projected Revenue Shortfall: <strong>R {diagnostics['revenue_at_risk']:,.2f}</strong>. 
            Commercial actions to close the gap before execution:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("💡 Actionable Commercial Recommendations")
    for action in diagnostics["actions"]:
        st.markdown(f"""
        <div class='solution-box'>
            <strong>[{action['type']}] {action['title']}</strong><br/>
            <span style='color: #374151; font-size: 14px;'>{action['detail']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)


# ============================================================
# 11. PLOTLY CHART: CONTINUOUS HISTORICAL + PROMO FORECAST
# ============================================================
st.subheader(f"📈 Historical vs. {forecast_horizon}-Week Forecast")

metric_toggle = st.radio("Select Chart View Metric:", ["Volume (Units)", "Sales Value (Revenue R)"], horizontal=True)

fig = go.Figure()

if "Units" in metric_toggle:
    # Historical Actual Units
    fig.add_trace(go.Scatter(
        x=weekly["date_key"], y=weekly["Sales_Units"],
        mode="lines+markers", name="Actual Units",
        line=dict(color="#7C3AED", width=2.5), marker=dict(size=5)
    ))

    # Historical Baseline Units - hide a rejected/invalid baseline from the main chart.
    if forecast_source == "Baseline Units":
        fig.add_trace(go.Scatter(
            x=weekly["date_key"], y=weekly["Baseline_Units"],
            mode="lines+markers", name="Baseline Units",
            line=dict(color="#EA580C", width=2, dash="dot"), marker=dict(size=4)
        ))

    # Seamless Anchor Point
    anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
    anchor_y = [float(weekly["Sales_Units"].iloc[-1])] + list(final_volume_fcst)

    # Promo Overlay Forecast Line
    fig.add_trace(go.Scatter(
        x=anchor_x, y=anchor_y,
        mode="lines+markers", name=(
            f"Forecast | {champion_model}"
            + (" | 52-Wk Promo Overlay" if apply_promo_overlay else " | Actual-based")
        ),
        line=dict(color="#059669", width=3, dash="dash"), marker=dict(size=7, symbol="diamond")
    ))

    # 52-Week Same Period Last Year Benchmark
    fig.add_trace(go.Scatter(
        x=future_dates, y=ly_benchmark_units,
        mode="lines+markers", name="52-Week LY Benchmark",
        line=dict(color="#D97706", width=2, dash="dot"), marker=dict(size=5)
    ))

    upper_bound = [u * 1.12 for u in final_volume_fcst]
    lower_bound = [u * 0.88 for u in final_volume_fcst]
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill="toself", fillcolor="rgba(5, 150, 105, 0.12)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        showlegend=True, name="Planning Range (±12%)"
    ))

    y_title = "Units"

else:
    # Historical Sales Value
    fig.add_trace(go.Scatter(
        x=weekly["date_key"], y=weekly["Sales_Value"],
        mode="lines+markers", name="Historical Sales Value",
        line=dict(color="#1E3A8A", width=2.5), marker=dict(size=5)
    ))

    anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
    anchor_y = [float(weekly["Sales_Value"].iloc[-1])] + list(final_value_fcst)

    fig.add_trace(go.Scatter(
        x=anchor_x, y=anchor_y,
        mode="lines+markers", name="Revenue Forecast",
        line=dict(color="#059669", width=3, dash="dash"), marker=dict(size=7, symbol="diamond")
    ))

    upper_bound_val = [v * 1.12 for v in final_value_fcst]
    lower_bound_val = [v * 0.88 for v in final_value_fcst]
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound_val + lower_bound_val[::-1],
        fill="toself", fillcolor="rgba(5, 150, 105, 0.12)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        showlegend=True, name="Planning Range (±12%)"
    ))

    y_title = "Sales Value (R)"

# Force every weekly observation to appear on the x-axis.
all_chart_dates = list(weekly["date_key"]) + list(future_dates)
all_chart_dates = pd.to_datetime(pd.Series(all_chart_dates)).drop_duplicates().sort_values().tolist()
tick_text = [d.strftime("%d %b") for d in all_chart_dates]

# Make the forecast boundary visually obvious.
fig.add_vline(
    x=weekly["date_key"].iloc[-1],
    line_width=1,
    line_dash="dot",
    line_color="#64748B",
)
fig.add_annotation(
    x=weekly["date_key"].iloc[-1],
    y=1.02,
    xref="x",
    yref="paper",
    text="Forecast starts →",
    showarrow=False,
    font=dict(size=11, color="#475569"),
    xanchor="left",
)

fig.update_layout(
    template="plotly_white", height=560, hovermode="x unified",
    margin=dict(l=60, r=30, t=90, b=115),
    xaxis=dict(
        title="Week Start Date",
        showgrid=True,
        tickmode="array",
        tickvals=all_chart_dates,
        ticktext=tick_text,
        tickangle=-45,
        automargin=True,
    ),
    yaxis=dict(title=y_title, showgrid=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

forecast_method_text = (
    f"The forecast is based on **{forecast_source}** using **{champion_model}**. "
    "The orange LY line is a comparison benchmark; it is not automatically added to the forecast. "
    "The green planning range is a planning band, not a statistical confidence interval."
)
st.caption(forecast_method_text)


# ============================================================
# 12. FORECAST DETAIL TABLE
# ============================================================
st.markdown("---")
st.subheader("🔮 Forecast Detail Table")

table_df = pd.DataFrame({
    "Week": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Base Forecast Units": base_volume_fcst,
    "52-Wk Promo Lift Factor": [f"{l:.2f}x" for l in promo_lifts],
    "Selected Forecast Units": final_volume_fcst,
    "Forecast Revenue": final_value_fcst,
    "Same Week LY Units": ly_benchmark_units,
})

st.dataframe(
    table_df.style.format({
        "Base Forecast Units": "{:,.0f}",
        "Selected Forecast Units": "{:,.0f}",
        "Forecast Revenue": "R {:,.2f}",
        "Same Week LY Units": "{:,.0f}",
    }),
    use_container_width=True, hide_index=True
)

st.caption(
    f"Forecast Input: **{forecast_source}** | Model: **{champion_model}** "
    f"| Backtest WMAPE: **{model_wmape:.1f}%** "
    f"| SA Macro Factor: **{macro_factor:.4f}** "
    f"(Net Price Squeeze: {max(0, price_yoy - cpi_inflation):.1f}%)"
)
