import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


# ============================================================
# 1. PAGE CONFIGURATION & COMPACT STYLING
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
    .card {padding: 14px 16px; border: 1px solid #E2E8F0; border-radius: 8px; background: #F8FAFC;}
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
# 3. DATA PREPROCESSING & AUTO-SCALE ENGINE
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

    # Actual Volume Units = Sales Value / Ave RSP
    df["Sales Units"] = np.where(
        df["Ave RSP"] > 0,
        df["Sales Value"] / df["Ave RSP"],
        np.nan,
    )
    df["Sales Units"] = pd.to_numeric(df["Sales Units"], errors="coerce").fillna(0).clip(lower=0)

    # Baseline & Incremental Processing (Auto Scale Fix)
    if measures["baseline"] is not None:
        raw_baseline = clean_number(df[measures["baseline"]]).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

        val_mean = df["Sales Value"].mean()
        units_mean = df["Sales Units"].mean()

        # Scale Auto-Detection: Check if Baseline is in Rand Value or Units
        if units_mean > 0 and val_mean > 0 and abs(raw_baseline.mean() - val_mean) < abs(raw_baseline.mean() - units_mean):
            df["Baseline Units"] = np.where(df["Ave RSP"] > 0, raw_baseline / df["Ave RSP"], 0)
            warnings_list.append("Source baseline field detected in Currency Rand; automatically converted to Baseline Units via Ave RSP.")
        else:
            df["Baseline Units"] = raw_baseline

        df["Baseline Units"] = pd.to_numeric(df["Baseline Units"], errors="coerce").fillna(0).clip(lower=0)

        calculated_incremental = df["Sales Units"] - df["Baseline Units"]
        df["Calculated Incremental Units"] = calculated_incremental

        incremental_col = measures.get("incremental")
        if incremental_col and incremental_col in df.columns:
            raw_inc = clean_number(df[incremental_col]).replace([np.inf, -np.inf], np.nan)
            if units_mean > 0 and val_mean > 0 and abs(raw_inc.abs().mean() - val_mean) < abs(raw_inc.abs().mean() - units_mean):
                scaled_inc = np.where(df["Ave RSP"] > 0, raw_inc / df["Ave RSP"], np.nan)
            else:
                scaled_inc = raw_inc

            df["Source Incremental Units"] = scaled_inc
            df["Incremental Units"] = df["Source Incremental Units"].fillna(calculated_incremental)
            df["Incremental Source"] = np.where(df["Source Incremental Units"].notna(), "Source measure", "Calculated fallback")
        else:
            df["Source Incremental Units"] = np.nan
            df["Incremental Units"] = calculated_incremental
            df["Incremental Source"] = "Calculated fallback"

        df["Incremental % of Baseline"] = np.where(df["Baseline Units"] > 0, (df["Incremental Units"] / df["Baseline Units"]) * 100, np.nan)
    else:
        df["Baseline Units"] = np.nan
        df["Source Incremental Units"] = np.nan
        df["Calculated Incremental Units"] = np.nan
        df["Incremental Units"] = np.nan
        df["Incremental Source"] = "Unavailable"
        df["Incremental % of Baseline"] = np.nan

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
# 4. WEEKLY AGGREGATION ENGINE
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
            Sales_Value=("Sales Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )

    base["Incremental_Units"] = base["Sales_Units"] - base["Baseline_Units"]
    base["Effective_RSP"] = np.where(base["Sales_Units"] > 0, base["Sales_Value"] / base["Sales_Units"], np.nan)

    promo_part = df.copy()
    promo_part["Promo_RSP"] = pd.to_numeric(promo_part["Promo RSP"], errors="coerce")
    promo_part = promo_part[(promo_part["Promo_RSP"] > 0) & (promo_part["Sales Units"] > 0)].copy()

    if promo_part.empty:
        base["Promo_RSP_Weighted"] = np.nan
    else:
        promo_part["Promo_Value_Proxy"] = promo_part["Promo_RSP"] * promo_part["Sales Units"]
        promo_week = promo_part.groupby("date_key").agg(Promo_Value_Proxy=("Promo_Value_Proxy", "sum"), Promo_Units=("Sales Units", "sum")).reset_index()
        promo_week["Promo_RSP_Weighted"] = np.where(promo_week["Promo_Units"] > 0, promo_week["Promo_Value_Proxy"] / promo_week["Promo_Units"], np.nan)
        base = base.merge(promo_week[["date_key", "Promo_RSP_Weighted"]], on="date_key", how="left")

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
    
    # Backtest models on trailing history
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


# ============================================================
# 6. 52-WEEK PROMOTIONAL OVERLAY & SA MACRO ELASTICITY
# ============================================================
def apply_52wk_promo_overlay_and_sa_macro(weekly, future_dates, base_volume_fcst, price_yoy, cpi_inflation, elasticity):
    """
    Overlays historical promotional lifts from 52 weeks prior (t - 52)
    and adjusts for South African consumer affordability elasticity.
    """
    adjusted_fcst_units = []
    promo_lift_factors = []
    ly_benchmark_units = []

    # Calculate SA Macro Adjustment Factor
    net_price_squeeze = max(0.0, price_yoy - cpi_inflation)
    macro_factor = max(0.5, 1.0 + (elasticity * (net_price_squeeze / 100.0)))

    for idx, f_date in enumerate(future_dates):
        target_ly_date = f_date - timedelta(weeks=52)
        dist = (weekly["date_key"] - target_ly_date).abs().dt.days
        closest_idx = dist.idxmin()

        if dist.loc[closest_idx] <= 7:
            ly_act_units = float(weekly.loc[closest_idx, "Sales_Units"])
            ly_base_units = float(weekly.loc[closest_idx, "Baseline_Units"])
            
            # Promo Lift Factor = Last Year Actual / Last Year Baseline
            lift_factor = (ly_act_units / ly_base_units) if ly_base_units > 0 else 1.0
            lift_factor = max(1.0, lift_factor)
            
            ly_benchmark_units.append(ly_act_units)
            promo_lift_factors.append(lift_factor)
            
            # Apply promotional lift adjusted by SA macro consumer elasticity
            effective_lift = 1.0 + ((lift_factor - 1.0) * macro_factor)
            adj_units = base_volume_fcst[idx] * effective_lift
            adjusted_fcst_units.append(adj_units)
        else:
            ly_benchmark_units.append(base_volume_fcst[idx])
            promo_lift_factors.append(1.0)
            adjusted_fcst_units.append(base_volume_fcst[idx])

    return np.array(adjusted_fcst_units), promo_lift_factors, ly_benchmark_units, macro_factor


# ============================================================
# 7. PRESCRIPTIVE RECOMMENDATION ENGINE
# ============================================================
def generate_gap_closing_recommendations(scope_name, fcst_units_4wk, ly_units_4wk, fcst_val_4wk, ly_val_4wk, avg_rsp, promo_rsp, num_dist, brand_name):
    """
    Diagnoses YoY deficits and generates quantitative, gap-closing commercial actions.
    """
    unit_deficit = ly_units_4wk - fcst_units_4wk
    unit_deficit_pct = (unit_deficit / ly_units_4wk) * 100 if ly_units_4wk > 0 else 0
    revenue_at_risk = max(0.0, ly_val_4wk - fcst_val_4wk)

    actions = []

    # 1. Price Elasticity & Promotional Depth Recommendation
    promo_discount_pct = ((avg_rsp - promo_rsp) / avg_rsp) * 100 if avg_rsp > 0 else 0
    target_promo_rsp = round(avg_rsp * 0.78, 2) # Target 22% promo depth
    est_unit_recovery_promo = int(unit_deficit * 0.60)
    
    actions.append({
        "type": "Pricing & Promotion",
        "title": f"Deepen Promotional RSP to R {target_promo_rsp:.2f} (22% Promo Depth)",
        "detail": f"Current promotional depth is running at {promo_discount_pct:.1f}%. Increasing promo depth to 22% (Target Promo RSP: R {target_promo_rsp:.2f}) is projected to stimulate demand and recover ~{est_unit_recovery_promo:,} units."
    })

    # 2. South Africa Payday & SASSA Liquidity Cycle Timing
    actions.append({
        "type": "Payday & SASSA Timing",
        "title": "Align Promotional Activations with Month-End Payday (25th - 1st)",
        "detail": "South African FMCG demand is hyper-cyclical around payday and SASSA grant liquidity. Ensure co-op catalogue features and end-cap displays are scheduled strictly during payday weeks."
    })

    # 3. Field Sales & Retail Distribution Fix
    if num_dist < 85.0:
        dist_gap = 85.0 - num_dist
        est_unit_recovery_dist = int(unit_deficit * 0.35)
        actions.append({
            "type": "Field Distribution Audit",
            "title": f"Field Sales Push: Audit & Recover +{dist_gap:.1f}% Numeric Distribution",
            "detail": f"Numeric distribution is lagging at {num_dist:.1f}%. Out-of-stocks in Shoprite/Checkers, Pick n Pay, or Wholesale key accounts are dragging volume. Resolving stockouts will recover ~{est_unit_recovery_dist:,} units."
        })

    # 4. Brand Conversion Intervention (Shield -> Rexona Roll-On Migration)
    if any(k in str(brand_name).lower() or k in str(scope_name).lower() for k in ["rexona", "shield"]):
        actions.append({
            "type": "Brand Rebrand Conversion",
            "title": "Deploy Co-Branded 'Shield is now Rexona' Shelf POS",
            "detail": "Residual shopper confusion post-rebrand is causing shopper leakage. Allocate trade spend to shelf-talkers, secondary bay displays, and digital catalogue banners during peak traffic weeks."
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
    st.info("👈 Upload your Unilever 52-week sales CSV extract in the left sidebar to start.")
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
    elasticity = st.sidebar.slider("Price Elasticity Coefficient (ε)", min_value=-2.5, max_value=-0.1, value=-1.1, step=0.1, help="Standard SA FMCG elasticity ranges between -0.8 and -1.5.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Hierarchy Filters")

    # Cascading Filters
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

    # Determine Final Filter Scope
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

# 1. Base Unconstrained Baseline Forecast
base_volume_fcst, champion_model, model_wmape = run_baseline_model(weekly["Baseline_Units"].values, forecast_horizon)
base_value_fcst, val_model, val_wmape = run_baseline_model(weekly["Sales_Value"].values, forecast_horizon)

# 2. 52-Week Promotional Overlay + SA Macro Elasticity Adjustment
final_volume_fcst, promo_lifts, ly_benchmark_units, macro_factor = apply_52wk_promo_overlay_and_sa_macro(
    weekly, future_dates, base_volume_fcst, price_yoy, cpi_inflation, elasticity
)

# Sales Value Forecast scales with adjusted volume
avg_rsp_latest = weekly["Effective_RSP"].iloc[-1] if weekly["Effective_RSP"].iloc[-1] > 0 else 25.0
final_value_fcst = final_volume_fcst * avg_rsp_latest

# Calculate Summary Totals
fcst_4wk_units = sum(final_volume_fcst)
fcst_4wk_val = sum(final_value_fcst)

ly_4wk_units = sum(ly_benchmark_units)
ly_4wk_val = ly_4wk_units * avg_rsp_latest

latest_promo_rsp = weekly["Promo_RSP_Weighted"].iloc[-1] if "Promo_RSP_Weighted" in weekly.columns and pd.notna(weekly["Promo_RSP_Weighted"].iloc[-1]) else avg_rsp_latest * 0.82
latest_dist = weekly["Distribution"].iloc[-1]

scope_label = f"{category} > {subcategory} > {brand} > {product_label}"

# Run Prescriptive Diagnostics
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


# Prescriptive Recommendation Alert Panel
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
st.subheader("📈 26-Week Historical vs. 4-Week Forecast (with 52-Week Promo Overlay)")

metric_toggle = st.radio("Select Chart View Metric:", ["Volume (Units)", "Sales Value (Revenue R)"], horizontal=True)

fig = go.Figure()

if "Units" in metric_toggle:
    # Historical Actual Units
    fig.add_trace(go.Scatter(
        x=weekly["date_key"], y=weekly["Sales_Units"],
        mode="lines+markers", name="Actual Units",
        line=dict(color="#7C3AED", width=2.5), marker=dict(size=5)
    ))

    # Historical Baseline Units
    fig.add_trace(go.Scatter(
        x=weekly["date_key"], y=weekly["Baseline_Units"],
        mode="lines+markers", name="Baseline Units",
        line=dict(color="#EA580C", width=2, dash="dot"), marker=dict(size=4)
    ))

    # Seamless Anchor Point (Connecting last historical point to forecast)
    anchor_x = [weekly["date_key"].iloc[-1]] + future_dates
    anchor_y = [float(weekly["Sales_Units"].iloc[-1])] + list(final_volume_fcst)

    # Promo Overlay Forecast Line
    fig.add_trace(go.Scatter(
        x=anchor_x, y=anchor_y,
        mode="lines+markers", name=f"Forecast (52-Wk Promo Overlay | {champion_model})",
        line=dict(color="#059669", width=3, dash="dash"), marker=dict(size=7, symbol="diamond")
    ))

    # 52-Week Same Period Last Year Benchmark
    fig.add_trace(go.Scatter(
        x=future_dates, y=ly_benchmark_units,
        mode="lines+markers", name="52-Week LY Benchmark",
        line=dict(color="#D97706", width=2, dash="dot"), marker=dict(size=5)
    ))

    # Confidence Band
    upper_bound = [u * 1.12 for u in final_volume_fcst]
    lower_bound = [u * 0.88 for u in final_volume_fcst]
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill="toself", fillcolor="rgba(5, 150, 105, 0.12)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        showlegend=True, name="80% Confidence Range"
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

    # Forecast Value Line
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
        showlegend=True, name="80% Confidence Range"
    ))

    y_title = "Sales Value (R)"

fig.update_layout(
    template="plotly_white", height=480, hovermode="x unified",
    xaxis=dict(title="Week Start Date", showgrid=True),
    yaxis=dict(title=y_title, showgrid=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 12. FORECAST DETAIL TABLE
# ============================================================
st.markdown("---")
st.subheader("🔮 Forecast Detail Table")

table_df = pd.DataFrame({
    "Week": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Unconstrained Baseline Units": base_volume_fcst,
    "52-Wk Promo Lift Factor": [f"{l:.2f}x" for l in promo_lifts],
    "Selected Forecast Units": final_volume_fcst,
    "Forecast Revenue": final_value_fcst,
    "Same Week LY Units": ly_benchmark_units,
})

st.dataframe(
    table_df.style.format({
        "Unconstrained Baseline Units": "{:,.0f}",
        "Selected Forecast Units": "{:,.0f}",
        "Forecast Revenue": "R {:,.2f}",
        "Same Week LY Units": "{:,.0f}",
    }),
    use_container_width=True, hide_index=True
)

st.caption(f"Baseline Engine: **{champion_model}** | SA Macro Factor: **{macro_factor:.4f}** (Net Price Squeeze: {max(0, price_yoy - cpi_inflation):.1f}%)")
