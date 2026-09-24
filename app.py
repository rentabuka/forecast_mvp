import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import timedelta


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Unilever Demand & Diagnostic Engine",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 20px;
    }

    .section-header {
        font-size: 20px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    [data-testid="stMetricValue"] {
        font-size: 19px !important;
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

    .alert-card {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .warning-card {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .positive-card {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .solution-box {
        background-color: #F8FAFC;
        border-left: 4px solid #1E3A8A;
        padding: 12px 16px;
        margin-top: 10px;
        border-radius: 4px;
    }

    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. REQUIRED DATA SCHEMA
# ============================================================

EXACT_REQUIRED_COLUMNS = [
    "Full Date",
    "Category",
    "Subcategory",
    "Brand",
    "ProductsID",
    "Product",
    "26 Weeks CY Value",
    "26 Weeks CY Ave Price Quantity",
    "26 Weeks CY Ave RSP On Promo"
]


# ============================================================
# 4. DATA VALIDATION AND PREPROCESSING
# ============================================================

def validate_and_preprocess(df):
    """
    Validate the uploaded dataset and create clean analytical fields.
    """

    errors = []
    warnings = []

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_cols = [
        col for col in EXACT_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_cols)
        )

        return False, errors, warnings, None

    df_clean = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df_clean["date_key"] = pd.to_datetime(
        df_clean["Full Date"],
        errors="coerce"
    )

    if df_clean["date_key"].isna().all():
        errors.append(
            "Full Date could not be converted into valid dates."
        )
        return False, errors, warnings, None

    invalid_dates = df_clean["date_key"].isna().sum()

    if invalid_dates > 0:
        warnings.append(
            f"{invalid_dates:,} rows contain invalid dates and will be excluded."
        )

        df_clean = df_clean.dropna(
            subset=["date_key"]
        ).copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df_clean["Sales Value"] = pd.to_numeric(
        df_clean["26 Weeks CY Value"],
        errors="coerce"
    )

    df_clean["Ave RSP"] = pd.to_numeric(
        df_clean["26 Weeks CY Ave Price Quantity"],
        errors="coerce"
    )

    df_clean["Promo RSP"] = pd.to_numeric(
        df_clean["26 Weeks CY Ave RSP On Promo"],
        errors="coerce"
    )

    # Missing values
    df_clean["Sales Value"] = (
        df_clean["Sales Value"]
        .fillna(0)
        .clip(lower=0)
    )

    df_clean["Ave RSP"] = (
        df_clean["Ave RSP"]
        .replace([np.inf, -np.inf], np.nan)
    )

    df_clean["Promo RSP"] = (
        df_clean["Promo RSP"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # Promo RSP fallback
    # --------------------------------------------------------

    df_clean["Promo RSP"] = df_clean["Promo RSP"].fillna(
        df_clean["Ave RSP"]
    )

    # --------------------------------------------------------
    # Sales Units
    # --------------------------------------------------------

    df_clean["Sales Units"] = np.where(
        df_clean["Ave RSP"] > 0,
        df_clean["Sales Value"] / df_clean["Ave RSP"],
        np.nan
    )

    df_clean["Sales Units"] = (
        df_clean["Sales Units"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Numeric Distribution
    # --------------------------------------------------------

    if "Numeric Distribution" in df_clean.columns:

        df_clean["Numeric Distribution"] = pd.to_numeric(
            df_clean["Numeric Distribution"],
            errors="coerce"
        )

        # Detect whether distribution is stored as 0-1
        valid_dist = df_clean["Numeric Distribution"].dropna()

        if not valid_dist.empty and valid_dist.max() <= 1.0:
            df_clean["Numeric Distribution"] *= 100

        df_clean["Numeric Distribution"] = (
            df_clean["Numeric Distribution"]
            .clip(lower=0, upper=100)
        )

    else:

        df_clean["Numeric Distribution"] = np.nan

        warnings.append(
            "Numeric Distribution was not found. "
            "Distribution diagnostics will be unavailable."
        )

    # --------------------------------------------------------
    # Promotion depth
    # --------------------------------------------------------

    df_clean["Promo Depth %"] = np.where(
        df_clean["Ave RSP"] > 0,
        (
            (df_clean["Ave RSP"] - df_clean["Promo RSP"])
            / df_clean["Ave RSP"]
        ) * 100,
        np.nan
    )

    df_clean["Promo Depth %"] = (
        df_clean["Promo Depth %"]
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0, upper=100)
    )

    # --------------------------------------------------------
    # Product names
    # --------------------------------------------------------

    for col in [
        "Category",
        "Subcategory",
        "Brand",
        "ProductsID",
        "Product"
    ]:

        df_clean[col] = (
            df_clean[col]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df_clean = df_clean.sort_values(
        "date_key"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Data quality warnings
    # --------------------------------------------------------

    number_of_dates = df_clean["date_key"].nunique()

    if number_of_dates < 12:

        warnings.append(
            f"Only {number_of_dates} unique dates were found. "
            "Forecast reliability will be limited."
        )

    elif number_of_dates < 26:

        warnings.append(
            f"Only {number_of_dates} unique dates were found. "
            "At least 26 weekly observations are recommended."
        )

    # --------------------------------------------------------
    # Important semantic warning
    # --------------------------------------------------------

    if number_of_dates > 1:

        date_diff = (
            df_clean["date_key"]
            .sort_values()
            .drop_duplicates()
            .diff()
            .dt.days
            .dropna()
        )

        if not date_diff.empty:

            median_gap = date_diff.median()

            if median_gap > 10:

                warnings.append(
                    "The dates do not appear to be weekly. "
                    f"Median date gap is approximately {median_gap:.0f} days."
                )

    return True, errors, warnings, df_clean


# ============================================================
# 5. WEEKLY AGGREGATION
# ============================================================

def aggregate_weekly(df):
    """
    Aggregate the selected portfolio/SKU scope to weekly level.
    """

    if df.empty:
        return pd.DataFrame()

    weekly = (
        df.set_index("date_key")
        .groupby(pd.Grouper(freq="W-SUN"))
        .agg(
            {
                "Sales Value": "sum",
                "Sales Units": "sum",
                "Numeric Distribution": "mean",
                "Promo Depth %": "mean",
                "Ave RSP": "mean",
                "Promo RSP": "mean"
            }
        )
        .reset_index()
    )

    weekly = weekly.sort_values(
        "date_key"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Effective selling price
    # --------------------------------------------------------

    weekly["Effective RSP"] = np.where(
        weekly["Sales Units"] > 0,
        weekly["Sales Value"] / weekly["Sales Units"],
        np.nan
    )

    return weekly


# ============================================================
# 6. TREND CALCULATION
# ============================================================

def calculate_trend(values):
    """
    Calculate a simple linear trend using historical observations.

    Returns a weekly percentage trend.
    """

    values = pd.Series(values).dropna()

    if len(values) < 6:
        return 0.0

    # Use recent history so old structural changes don't dominate.
    lookback = min(13, len(values))

    recent = values.tail(
        lookback
    ).values

    if np.all(recent <= 0):
        return 0.0

    x = np.arange(
        len(recent),
        dtype=float
    )

    try:

        slope = np.polyfit(
            x,
            recent,
            1
        )[0]

        average = np.mean(recent)

        if average <= 0:
            return 0.0

        trend = slope / average

        # Dampen extreme trends.
        trend = np.clip(
            trend,
            -0.05,
            0.05
        )

        return float(trend)

    except Exception:
        return 0.0


# ============================================================
# 7. SEASONALITY
# ============================================================

def calculate_seasonality(df, target_col):
    """
    Estimate annual weekly seasonality when at least
    approximately one year of data exists.

    With less than 52 observations, no annual seasonality
    is imposed.
    """

    if len(df) < 52:
        return {}

    temp = df[
        ["date_key", target_col]
    ].dropna().copy()

    if temp.empty:
        return {}

    temp["week_of_year"] = (
        temp["date_key"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    overall_mean = temp[target_col].mean()

    if overall_mean <= 0:
        return {}

    seasonal = (
        temp.groupby("week_of_year")[target_col]
        .mean()
        / overall_mean
    )

    # Limit extreme seasonality.
    seasonal = seasonal.clip(
        lower=0.70,
        upper=1.30
    )

    return seasonal.to_dict()


# ============================================================
# 8. FORECAST BASELINE
# ============================================================

def forecast_series(
    history,
    periods=4,
    seasonal_factors=None
):
    """
    Forecast future values using:

    - Recent weighted run-rate
    - Recent trend
    - Annual seasonality when available

    No random noise is used.
    """

    values = (
        pd.Series(history)
        .astype(float)
        .fillna(0)
        .clip(lower=0)
    )

    if len(values) == 0:
        return np.zeros(periods)

    # --------------------------------------------------------
    # Recent weighted run-rate
    # --------------------------------------------------------

    lookback = min(6, len(values))

    recent = values.tail(
        lookback
    ).values

    weights = np.arange(
        1,
        len(recent) + 1
    )

    weighted_average = np.average(
        recent,
        weights=weights
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = calculate_trend(values)

    forecasts = []

    last_value = weighted_average

    for i in range(1, periods + 1):

        # Dampen trend as we move further into the future.
        damped_trend = trend * (0.75 ** (i - 1))

        forecast = (
            weighted_average
            * (1 + damped_trend * i)
        )

        # ----------------------------------------------------
        # Seasonality
        # ----------------------------------------------------

        if seasonal_factors:

            future_position = len(values) + i

            # Approximate week position
            seasonal_index = (
                future_position - 1
            ) % 52 + 1

            seasonal_factor = seasonal_factors.get(
                seasonal_index,
                1.0
            )

            forecast *= seasonal_factor

        # Prevent negative forecast.
        forecast = max(
            0,
            forecast
        )

        forecasts.append(
            forecast
        )

        last_value = forecast

    return np.array(
        forecasts
    )


# ============================================================
# 9. FORECAST ERROR / PREDICTION RANGE
# ============================================================

def calculate_backtest_errors(
    history,
    minimum_training=12,
    horizon=4
):
    """
    Rolling-origin backtest.

    This gives us an empirical distribution of historical
    forecast errors which can be used to create a forecast
    range.

    This is much more defensible than simply using +/- 12%.
    """

    values = (
        pd.Series(history)
        .astype(float)
        .fillna(0)
        .clip(lower=0)
        .reset_index(drop=True)
    )

    if len(values) < minimum_training + horizon:
        return np.array([])

    errors = []

    for split in range(
        minimum_training,
        len(values) - horizon + 1
    ):

        train = values.iloc[
            :split
        ]

        actual = values.iloc[
            split:split + horizon
        ]

        forecast = forecast_series(
            train,
            periods=horizon
        )

        for pred, obs in zip(
            forecast,
            actual
        ):

            if obs > 0:

                error = (
                    pred - obs
                ) / obs

                errors.append(
                    error
                )

    return np.array(
        errors
    )


# ============================================================
# 10. FORECAST INTERVAL
# ============================================================

def create_prediction_interval(
    forecast,
    historical_errors
):
    """
    Create an empirical forecast range based on historical
    forecast errors.
    """

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    if (
        historical_errors is None
        or len(historical_errors) < 10
    ):

        # Not enough evidence for a statistical interval.
        # Return a conservative illustrative range.
        lower_factor = 0.85
        upper_factor = 1.15

        return (
            forecast * lower_factor,
            forecast * upper_factor,
            False
        )

    lower_error = np.nanpercentile(
        historical_errors,
        10
    )

    upper_error = np.nanpercentile(
        historical_errors,
        90
    )

    lower = forecast * (
        1 + lower_error
    )

    upper = forecast * (
        1 + upper_error
    )

    lower = np.maximum(
        0,
        lower
    )

    return (
        lower,
        upper,
        True
    )


# ============================================================
# 11. FULL FORECAST ENGINE
# ============================================================

def compute_forecast(
    df_aggregated,
    periods=4
):
    """
    Generate the full historical + forecast dataset.
    """

    if df_aggregated.empty:
        return pd.DataFrame(), {}

    df = (
        df_aggregated
        .sort_values("date_key")
        .copy()
    )

    # Keep all available history for modelling.
    unit_history = (
        df["Sales Units"]
        .fillna(0)
        .values
    )

    value_history = (
        df["Sales Value"]
        .fillna(0)
        .values
    )

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    unit_seasonality = calculate_seasonality(
        df,
        "Sales Units"
    )

    value_seasonality = calculate_seasonality(
        df,
        "Sales Value"
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    unit_forecast = forecast_series(
        unit_history,
        periods=periods,
        seasonal_factors=unit_seasonality
    )

    value_forecast = forecast_series(
        value_history,
        periods=periods,
        seasonal_factors=value_seasonality
    )

    # --------------------------------------------------------
    # Backtest errors
    # --------------------------------------------------------

    unit_errors = calculate_backtest_errors(
        unit_history,
        horizon=periods
    )

    value_errors = calculate_backtest_errors(
        value_history,
        horizon=periods
    )

    unit_lower, unit_upper, unit_empirical = (
        create_prediction_interval(
            unit_forecast,
            unit_errors
        )
    )

    value_lower, value_upper, value_empirical = (
        create_prediction_interval(
            value_forecast,
            value_errors
        )
    )

    # --------------------------------------------------------
    # Historical
    # --------------------------------------------------------

    historical = df[
        [
            "date_key",
            "Sales Units",
            "Sales Value",
            "Numeric Distribution",
            "Promo Depth %",
            "Effective RSP"
        ]
    ].copy()

    historical["Type"] = "Historical"

    historical["units_p10"] = np.nan
    historical["units_p90"] = np.nan
    historical["value_p10"] = np.nan
    historical["value_p90"] = np.nan

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    last_date = df["date_key"].max()

    future_rows = []

    for i in range(
        periods
    ):

        future_date = (
            last_date
            + timedelta(weeks=i + 1)
        )

        forecast_units = unit_forecast[i]
        forecast_value = value_forecast[i]

        future_rows.append(
            {
                "date_key": future_date,
                "Sales Units": forecast_units,
                "Sales Value": forecast_value,
                "Numeric Distribution": np.nan,
                "Promo Depth %": np.nan,
                "Effective RSP": (
                    forecast_value / forecast_units
                    if forecast_units > 0
                    else np.nan
                ),
                "Type": "Forecast",
                "units_p10": unit_lower[i],
                "units_p90": unit_upper[i],
                "value_p10": value_lower[i],
                "value_p90": value_upper[i]
            }
        )

    forecast_df = pd.DataFrame(
        future_rows
    )

    combined = pd.concat(
        [
            historical,
            forecast_df
        ],
        ignore_index=True
    )

    metadata = {
        "unit_empirical_interval": unit_empirical,
        "value_empirical_interval": value_empirical,
        "unit_errors": unit_errors,
        "value_errors": value_errors,
        "unit_seasonality": unit_seasonality,
        "value_seasonality": value_seasonality
    }

    return combined, metadata


# ============================================================
# 12. YEAR-ON-YEAR COMPARISON
# ============================================================

def calculate_yoy_comparison(
    weekly,
    forecast_df
):
    """
    Compare forecast against actual same-period-last-year
    data where 52+ weeks are available.

    No synthetic LY data is generated.
    """

    result = {
        "available": False,
        "forecast_units": forecast_df["Sales Units"].sum(),
        "forecast_value": forecast_df["Sales Value"].sum(),
        "ly_units": np.nan,
        "ly_value": np.nan,
        "unit_growth": np.nan,
        "value_growth": np.nan,
        "unit_change": np.nan,
        "value_change": np.nan
    }

    if weekly.empty:
        return result

    # Need at least approximately one year.
    if len(weekly) < 53:
        return result

    weekly = weekly.sort_values(
        "date_key"
    ).copy()

    last_date = weekly["date_key"].max()

    forecast_dates = forecast_df[
        "date_key"
    ]

    if forecast_dates.empty:
        return result

    # --------------------------------------------------------
    # Compare each forecast week to approximately 52 weeks
    # earlier.
    # --------------------------------------------------------

    ly_units = []
    ly_value = []

    for future_date in forecast_dates:

        target_date = (
            future_date
            - timedelta(weeks=52)
        )

        if weekly.empty:
            continue

        distances = (
            weekly["date_key"] - target_date
        ).abs().dt.days

        nearest_idx = distances.idxmin()

        nearest_distance = (
            distances.loc[nearest_idx]
        )

        # Accept only reasonably close weekly observations.
        if nearest_distance <= 7:

            ly_units.append(
                weekly.loc[
                    nearest_idx,
                    "Sales Units"
                ]
            )

            ly_value.append(
                weekly.loc[
                    nearest_idx,
                    "Sales Value"
                ]
            )

    if len(ly_units) < len(forecast_dates):
        return result

    ly_units_total = np.sum(
        ly_units
    )

    ly_value_total = np.sum(
        ly_value
    )

    forecast_units = result[
        "forecast_units"
    ]

    forecast_value = result[
        "forecast_value"
    ]

    unit_growth = (
        (
            forecast_units
            / ly_units_total
        ) - 1
    ) * 100 if ly_units_total > 0 else np.nan

    value_growth = (
        (
            forecast_value
            / ly_value_total
        ) - 1
    ) * 100 if ly_value_total > 0 else np.nan

    result.update(
        {
            "available": True,
            "ly_units": ly_units_total,
            "ly_value": ly_value_total,
            "unit_growth": unit_growth,
            "value_growth": value_growth,
            "unit_change": (
                forecast_units
                - ly_units_total
            ),
            "value_change": (
                forecast_value
                - ly_value_total
            )
        }
    )

    return result


# ============================================================
# 13. COMMERCIAL DIAGNOSTICS
# ============================================================

def generate_diagnostics(
    weekly,
    yoy,
    product_label
):
    """
    Generate diagnostics from observed historical data.

    Important:
    This function does NOT invent elasticity or uplift.
    """

    diagnostics = []

    if weekly.empty:
        return diagnostics

    latest = weekly.iloc[-1]

    # --------------------------------------------------------
    # Distribution diagnostic
    # --------------------------------------------------------

    if (
        weekly["Numeric Distribution"]
        .notna()
        .sum() >= 4
    ):

        recent_dist = (
            weekly["Numeric Distribution"]
            .tail(4)
            .mean()
        )

        prior_dist = (
            weekly["Numeric Distribution"]
            .tail(13)
            .head(9)
            .mean()
        )

        if (
            not np.isnan(recent_dist)
            and not np.isnan(prior_dist)
        ):

            dist_change = (
                recent_dist
                - prior_dist
            )

            if dist_change < -2:

                diagnostics.append(
                    {
                        "type": "Distribution",
                        "title": "Distribution is weakening",
                        "detail": (
                            f"Average numeric distribution over "
                            f"the latest 4 weeks is "
                            f"{recent_dist:.1f}%, compared with "
                            f"{prior_dist:.1f}% over the earlier "
                            f"comparison period. The change is "
                            f"{dist_change:.1f} percentage points."
                        )
                    }
                )

            elif dist_change > 2:

                diagnostics.append(
                    {
                        "type": "Distribution",
                        "title": "Distribution is improving",
                        "detail": (
                            f"Average numeric distribution over "
                            f"the latest 4 weeks is "
                            f"{recent_dist:.1f}%, up "
                            f"{dist_change:.1f} percentage points "
                            f"versus the earlier period."
                        )
                    }
                )

    # --------------------------------------------------------
    # Promotion diagnostic
    # --------------------------------------------------------

    if (
        weekly["Promo Depth %"]
        .notna()
        .sum() >= 4
    ):

        recent_promo = (
            weekly["Promo Depth %"]
            .tail(4)
            .mean()
        )

        previous_promo = (
            weekly["Promo Depth %"]
            .tail(13)
            .head(9)
            .mean()
        )

        if (
            not np.isnan(recent_promo)
            and not np.isnan(previous_promo)
        ):

            promo_change = (
                recent_promo
                - previous_promo
            )

            if promo_change > 2:

                diagnostics.append(
                    {
                        "type": "Promotion",
                        "title": "Promotion depth has increased",
                        "detail": (
                            f"Average promotional depth is "
                            f"{recent_promo:.1f}% in the latest "
                            f"4 weeks versus "
                            f"{previous_promo:.1f}% in the "
                            f"earlier comparison period."
                        )
                    }
                )

            elif promo_change < -2:

                diagnostics.append(
                    {
                        "type": "Promotion",
                        "title": "Promotion depth has decreased",
                        "detail": (
                            f"Average promotional depth is "
                            f"{recent_promo:.1f}% in the latest "
                            f"4 weeks versus "
                            f"{previous_promo:.1f}% previously."
                        )
                    }
                )

    # --------------------------------------------------------
    # Price diagnostic
    # --------------------------------------------------------

    if (
        weekly["Effective RSP"]
        .notna()
        .sum() >= 6
    ):

        recent_price = (
            weekly["Effective RSP"]
            .tail(4)
            .mean()
        )

        prior_price = (
            weekly["Effective RSP"]
            .tail(13)
            .head(9)
            .mean()
        )

        if (
            prior_price > 0
            and not np.isnan(recent_price)
        ):

            price_change = (
                (
                    recent_price
                    / prior_price
                ) - 1
            ) * 100

            if abs(price_change) >= 3:

                diagnostics.append(
                    {
                        "type": "Price",
                        "title": "Effective selling price has changed",
                        "detail": (
                            f"Average effective selling price "
                            f"is R {recent_price:,.2f} in the "
                            f"latest 4 weeks versus "
                            f"R {prior_price:,.2f} previously, "
                            f"a change of {price_change:+.1f}%."
                        )
                    }
                )

    # --------------------------------------------------------
    # YoY diagnostic
    # --------------------------------------------------------

    if yoy["available"]:

        if yoy["unit_growth"] < 0:

            diagnostics.append(
                {
                    "type": "Forecast",
                    "title": "Forecast volume is below last year",
                    "detail": (
                        f"The 4-week forecast is "
                        f"{abs(yoy['unit_growth']):.1f}% below "
                        f"the comparable period last year."
                    )
                }
            )

        else:

            diagnostics.append(
                {
                    "type": "Forecast",
                    "title": "Forecast volume is above last year",
                    "detail": (
                        f"The 4-week forecast is "
                        f"{yoy['unit_growth']:.1f}% above "
                        f"the comparable period last year."
                    )
                }
            )

    return diagnostics


# ============================================================
# 14. FORECAST ACCURACY DISPLAY
# ============================================================

def calculate_mape(errors):

    if errors is None or len(errors) == 0:
        return np.nan

    return (
        np.mean(
            np.abs(errors)
        ) * 100
    )


# ============================================================
# 15. MAIN APP
# ============================================================

st.markdown(
    """
    <div class="main-header">
        Unilever Demand & Diagnostic Forecasting System
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "📥 Data & Filters"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Weekly Sales CSV",
    type=["csv"]
)


# ============================================================
# FILE PROCESSING
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload your Unilever sales CSV dataset "
        "in the left sidebar to generate forecasts "
        "and commercial diagnostics."
    )

    st.markdown(
        """
        ### Recommended data

        The platform works best when the dataset contains:

        - At least 26 weeks of history
        - Ideally 52+ weeks for YoY and seasonality
        - Category
        - Subcategory
        - Brand
        - Product / SKU
        - Date
        - Sales Value
        - Price
        - Promotional Price
        - Numeric Distribution

        **Important:** The platform will not manufacture a
        Last Year benchmark when actual comparable history
        is unavailable.
        """
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    raw_df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE
# ============================================================

is_valid, errors, warnings, df_clean = (
    validate_and_preprocess(
        raw_df
    )
)


if not is_valid:

    st.error(
        "❌ Schema Validation Errors"
    )

    for error in errors:

        st.write(
            f"- {error}"
        )

    st.stop()


# ============================================================
# WARNINGS
# ============================================================

for warning in warnings:

    st.warning(
        warning
    )


# ============================================================
# DATA SUMMARY
# ============================================================

number_of_weeks = (
    df_clean["date_key"]
    .nunique()
)

min_date = (
    df_clean["date_key"]
    .min()
)

max_date = (
    df_clean["date_key"]
    .max()
)

st.sidebar.success(
    f"✅ Dataset loaded: {number_of_weeks} periods"
)

st.sidebar.caption(
    f"{min_date.date()} → {max_date.date()}"
)

st.sidebar.markdown("---")


# ============================================================
# CATEGORY FILTER
# ============================================================

categories = (
    ["All"]
    + sorted(
        df_clean["Category"]
        .unique()
        .tolist()
    )
)

selected_category = st.sidebar.selectbox(
    "1. Category",
    categories
)


df_category = df_clean.copy()

if selected_category != "All":

    df_category = df_category[
        df_category["Category"]
        == selected_category
    ].copy()


# ============================================================
# SUBCATEGORY
# ============================================================

subcategories = (
    ["All"]
    + sorted(
        df_category["Subcategory"]
        .unique()
        .tolist()
    )
)

selected_subcategory = st.sidebar.selectbox(
    "2. Subcategory",
    subcategories
)


df_subcategory = df_category.copy()

if selected_subcategory != "All":

    df_subcategory = df_subcategory[
        df_subcategory["Subcategory"]
        == selected_subcategory
    ].copy()


# ============================================================
# BRAND
# ============================================================

brands = (
    ["All"]
    + sorted(
        df_subcategory["Brand"]
        .unique()
        .tolist()
    )
)

selected_brand = st.sidebar.selectbox(
    "3. Brand",
    brands
)


df_brand = df_subcategory.copy()

if selected_brand != "All":

    df_brand = df_brand[
        df_brand["Brand"]
        == selected_brand
    ].copy()


# ============================================================
# PRODUCT
# ============================================================

product_options = (
    sorted(
        df_brand["Product"]
        .unique()
        .tolist()
    )
)

selected_products = st.sidebar.multiselect(
    "4. Product SKU(s)",
    options=product_options,
    default=[]
)


if not selected_products:

    final_df = df_brand.copy()

    product_label = (
        f"All Products in "
        f"{selected_brand if selected_brand != 'All' else 'Portfolio'}"
    )

else:

    final_df = df_brand[
        df_brand["Product"]
        .isin(selected_products)
    ].copy()

    if len(selected_products) == 1:

        product_label = selected_products[0]

    else:

        product_label = (
            f"{len(selected_products)} Selected SKUs"
        )


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if final_df.empty:

    st.warning(
        "⚠️ No data matches the selected filter combination."
    )

    st.stop()


# ============================================================
# SCOPE LABEL
# ============================================================

scope_label = (
    f"{selected_category} > "
    f"{selected_subcategory} > "
    f"{selected_brand} > "
    f"{product_label}"
)


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

df_weekly = aggregate_weekly(
    final_df
)


if df_weekly.empty:

    st.error(
        "Unable to create weekly aggregated data."
    )

    st.stop()


# ============================================================
# FORECAST
# ============================================================

df_forecast, forecast_metadata = (
    compute_forecast(
        df_weekly,
        periods=4
    )
)


forecast_only = df_forecast[
    df_forecast["Type"] == "Forecast"
].copy()


# ============================================================
# FORECAST TOTALS
# ============================================================

forecast_4wk_units = (
    forecast_only["Sales Units"]
    .sum()
)

forecast_4wk_value = (
    forecast_only["Sales Value"]
    .sum()
)

forecast_p10_units = (
    forecast_only["units_p10"]
    .sum()
)

forecast_p90_units = (
    forecast_only["units_p90"]
    .sum()
)

forecast_p10_value = (
    forecast_only["value_p10"]
    .sum()
)

forecast_p90_value = (
    forecast_only["value_p90"]
    .sum()
)


# ============================================================
# YOY
# ============================================================

yoy = calculate_yoy_comparison(
    df_weekly,
    forecast_only
)


# ============================================================
# DIAGNOSTICS
# ============================================================

diagnostics = generate_diagnostics(
    df_weekly,
    yoy,
    product_label
)


# ============================================================
# CURRENT METRICS
# ============================================================

latest_week = df_weekly.iloc[-1]

latest_distribution = (
    latest_week["Numeric Distribution"]
)

latest_promo_depth = (
    latest_week["Promo Depth %"]
)

latest_rsp = (
    latest_week["Effective RSP"]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    ### Forecast Scope

    **{scope_label}**
    """
)


# ============================================================
# DATA QUALITY STATUS
# ============================================================

if len(df_weekly) < 26:

    st.markdown(
        f"""
        <div class="warning-card">
        <strong>⚠️ Limited history</strong><br>
        Only {len(df_weekly)} weekly observations are available.
        Forecasts should be interpreted cautiously.
        </div>
        """,
        unsafe_allow_html=True
    )

elif len(df_weekly) < 52:

    st.markdown(
        """
        <div class="info-box">
        <strong>ℹ️ 26–51 weeks of history</strong><br>
        The model can produce a short-term forecast, but
        annual seasonality and actual same-period-last-year
        comparisons are not yet available.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="positive-card">
        <strong>✓ Sufficient historical depth</strong><br>
        The dataset contains at least 52 weeks, allowing
        annual seasonality and comparable-period analysis.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# KPI DASHBOARD
# ============================================================

st.markdown(
    "### 📊 Forecast Summary"
)

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "4-Wk Volume Forecast",
    f"{forecast_4wk_units:,.0f}"
)


col2.metric(
    "4-Wk Revenue Forecast",
    f"R {forecast_4wk_value:,.0f}"
)


if yoy["available"]:

    col3.metric(
        "Forecast YoY Volume",
        f"{yoy['unit_growth']:+.1f}%"
    )

    col4.metric(
        "Forecast YoY Value",
        f"{yoy['value_growth']:+.1f}%"
    )

else:

    col3.metric(
        "Forecast YoY Volume",
        "N/A"
    )

    col4.metric(
        "Forecast YoY Value",
        "N/A"
    )


# ============================================================
# SECOND KPI ROW
# ============================================================

col5, col6, col7, col8 = st.columns(4)


if not np.isnan(latest_distribution):

    col5.metric(
        "Latest Distribution",
        f"{latest_distribution:.1f}%"
    )

else:

    col5.metric(
        "Latest Distribution",
        "N/A"
    )


if not np.isnan(latest_promo_depth):

    col6.metric(
        "Promo Depth",
        f"{latest_promo_depth:.1f}%"
    )

else:

    col6.metric(
        "Promo Depth",
        "N/A"
    )


if not np.isnan(latest_rsp):

    col7.metric(
        "Effective RSP",
        f"R {latest_rsp:,.2f}"
    )

else:

    col7.metric(
        "Effective RSP",
        "N/A"
    )


col8.metric(
    "Historical Weeks",
    f"{len(df_weekly):,}"
)


# ============================================================
# YOY SECTION
# ============================================================

st.markdown("---")

st.markdown(
    "### 📅 Comparable Period Analysis"
)


if yoy["available"]:

    yoy_col1, yoy_col2, yoy_col3, yoy_col4 = (
        st.columns(4)
    )

    yoy_col1.metric(
        "Forecast Units",
        f"{yoy['forecast_units']:,.0f}"
    )

    yoy_col2.metric(
        "LY Comparable Units",
        f"{yoy['ly_units']:,.0f}"
    )

    yoy_col3.metric(
        "Volume Change",
        f"{yoy['unit_change']:+,.0f}"
    )

    yoy_col4.metric(
        "Volume Growth",
        f"{yoy['unit_growth']:+.1f}%"
    )

else:

    st.info(
        "Actual same-period-last-year comparison is not "
        "available because the selected scope does not "
        "contain at least 52 weeks of usable history. "
        "No artificial LY benchmark has been created."
    )


# ============================================================
# FORECAST RANGE
# ============================================================

st.markdown("---")

st.markdown(
    "### 📐 Forecast Range"
)


range_col1, range_col2 = st.columns(2)


range_col1.metric(
    "4-Wk Volume Lower Range",
    f"{forecast_p10_units:,.0f}"
)

range_col1.metric(
    "4-Wk Volume Upper Range",
    f"{forecast_p90_units:,.0f}"
)


range_col2.metric(
    "4-Wk Revenue Lower Range",
    f"R {forecast_p10_value:,.0f}"
)

range_col2.metric(
    "4-Wk Revenue Upper Range",
    f"R {forecast_p90_value:,.0f}"
)


if (
    forecast_metadata["unit_empirical_interval"]
    and forecast_metadata["value_empirical_interval"]
):

    st.caption(
        "Forecast ranges are based on empirical historical "
        "back-test errors from the selected scope."
    )

else:

    st.caption(
        "Forecast ranges are indicative because there are "
        "not enough historical observations to estimate "
        "empirical forecast errors reliably."
    )


# ============================================================
# COMMERCIAL DIAGNOSTICS
# ============================================================

st.markdown("---")

st.markdown(
    "### 🔎 Commercial Diagnostics"
)


if not diagnostics:

    st.success(
        "No major historical diagnostic signal was detected "
        "from the available data."
    )

else:

    for diagnostic in diagnostics:

        st.markdown(
            f"""
            <div class="solution-box">
                <strong>
                    [{diagnostic['type']}] 
                    {diagnostic['title']}
                </strong>
                <br>
                <span style="color:#374151;font-size:14px;">
                    {diagnostic['detail']}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PRESCRIPTIVE GUIDANCE
# ============================================================

st.markdown("---")

st.markdown(
    "### 💡 Commercial Action Framework"
)


st.info(
    """
    The system deliberately does not claim that a specific
    discount, distribution increase or promotion will recover
    a fixed number of units unless that relationship has been
    statistically estimated from the underlying data.

    Recommended commercial workflow:

    **Identify → Quantify → Test → Measure**

    1. Identify the driver.
    2. Quantify the historical relationship.
    3. Test the intervention.
    4. Measure the resulting incremental volume/value.
    """
)


# ============================================================
# CHART
# ============================================================

st.markdown("---")

st.markdown(
    "### 📈 Historical vs 4-Week Forecast"
)

st.caption(
    f"Active Scope: {scope_label}"
)


metric_toggle = st.radio(
    "Chart Metric",
    [
        "Sales Units (Volume)",
        "Sales Value (Revenue R)"
    ],
    horizontal=True
)


if "Units" in metric_toggle:

    selected_col = "Sales Units"
    lower_bound_col = "units_p10"
    upper_bound_col = "units_p90"
    y_title = "Sales Units"

else:

    selected_col = "Sales Value"
    lower_bound_col = "value_p10"
    upper_bound_col = "value_p90"
    y_title = "Sales Value (R)"


historical_plot = df_forecast[
    df_forecast["Type"] == "Historical"
].copy()


forecast_plot = df_forecast[
    df_forecast["Type"] == "Forecast"
].copy()


# ============================================================
# PLOT
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# Historical
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=historical_plot["date_key"],
        y=historical_plot[selected_col],
        mode="lines+markers",
        name="Historical",
        line=dict(
            width=2.5
        ),
        marker=dict(
            size=5
        )
    )
)


# ------------------------------------------------------------
# Forecast
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=forecast_plot["date_key"],
        y=forecast_plot[selected_col],
        mode="lines+markers",
        name="4-Week Forecast",
        line=dict(
            width=3,
            dash="dash"
        ),
        marker=dict(
            size=7,
            symbol="diamond"
        )
    )
)


# ------------------------------------------------------------
# Prediction range
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=pd.concat(
            [
                forecast_plot["date_key"],
                forecast_plot["date_key"].iloc[::-1]
            ]
        ),
        y=pd.concat(
            [
                forecast_plot[upper_bound_col],
                forecast_plot[lower_bound_col].iloc[::-1]
            ]
        ),
        fill="toself",
        fillcolor="rgba(30, 58, 138, 0.12)",
        line=dict(
            color="rgba(255,255,255,0)"
        ),
        hoverinfo="skip",
        showlegend=True,
        name="Forecast Range"
    )
)


# ------------------------------------------------------------
# Last historical date
# ------------------------------------------------------------

if not historical_plot.empty:

    last_hist_date = (
        historical_plot["date_key"].iloc[-1]
    )

    fig.add_vline(
        x=last_hist_date,
        line_dash="dot",
        line_width=1
    )


fig.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    xaxis=dict(
        title="Week"
    ),
    yaxis=dict(
        title=y_title
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HISTORICAL DRIVER TABLE
# ============================================================

st.markdown("---")

st.markdown(
    "### 📊 Recent Commercial Drivers"
)


driver_table = df_weekly.tail(
    8
).copy()


driver_table = driver_table[
    [
        "date_key",
        "Sales Units",
        "Sales Value",
        "Effective RSP",
        "Promo Depth %",
        "Numeric Distribution"
    ]
].copy()


driver_table.columns = [
    "Week",
    "Units",
    "Sales Value",
    "Effective RSP",
    "Promo Depth %",
    "Numeric Distribution %"
]


driver_table["Week"] = (
    driver_table["Week"]
    .dt.strftime("%Y-%m-%d")
)


st.dataframe(
    driver_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FORECAST TABLE
# ============================================================

st.markdown("---")

st.markdown(
    "### 🔮 4-Week Forecast Detail"
)


forecast_table = forecast_only[
    [
        "date_key",
        "Sales Units",
        "Sales Value",
        "units_p10",
        "units_p90",
        "value_p10",
        "value_p90"
    ]
].copy()


forecast_table.columns = [
    "Week",
    "Forecast Units",
    "Forecast Value",
    "Units Lower",
    "Units Upper",
    "Value Lower",
    "Value Upper"
]


forecast_table["Week"] = (
    forecast_table["Week"]
    .dt.strftime("%Y-%m-%d")
)


st.dataframe(
    forecast_table.style.format(
        {
            "Forecast Units": "{:,.0f}",
            "Forecast Value": "R {:,.0f}",
            "Units Lower": "{:,.0f}",
            "Units Upper": "{:,.0f}",
            "Value Lower": "R {:,.0f}",
            "Value Upper": "R {:,.0f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FORECAST MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    "### 🧠 Forecast Model Information"
)


model_col1, model_col2, model_col3 = st.columns(3)


model_col1.metric(
    "History Used",
    f"{len(df_weekly)} weeks"
)


unit_mape = calculate_mape(
    forecast_metadata["unit_errors"]
)

value_mape = calculate_mape(
    forecast_metadata["value_errors"]
)


if np.isnan(unit_mape):

    model_col2.metric(
        "Volume Backtest Error",
        "N/A"
    )

else:

    model_col2.metric(
        "Volume Backtest Error",
        f"{unit_mape:.1f}%"
    )


if np.isnan(value_mape):

    model_col3.metric(
        "Value Backtest Error",
        "N/A"
    )

else:

    model_col3.metric(
        "Value Backtest Error",
        f"{value_mape:.1f}%"
    )


st.caption(
    """
    The forecasting engine combines a recent weighted
    run-rate with a damped historical trend. Annual
    seasonality is introduced only when sufficient
    historical data is available. Forecast uncertainty is
    estimated from rolling historical back-tests when enough
    observations exist.
    """
)


# ============================================================
# DATA QUALITY DETAILS
# ============================================================

with st.expander(
    "🔍 Data Quality Details"
):

    dq1, dq2, dq3, dq4 = st.columns(4)

    dq1.metric(
        "Rows",
        f"{len(final_df):,}"
    )

    dq2.metric(
        "SKUs",
        f"{final_df['Product'].nunique():,}"
    )

    dq3.metric(
        "Weeks",
        f"{len(df_weekly):,}"
    )

    dq4.metric(
        "Latest Date",
        str(
            df_weekly["date_key"].max().date()
        )
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(final_df.columns)
    )
