"""
Unilever FMCG Demand & Recovery Planner
=======================================

What the tool does, in plain terms:
  1. Forecasts weekly demand for 4, 8 or 12 weeks.
  2. Tests every available forecasting model on your own history (rolling
     backtest) and uses ONLY the model with the lowest error (WMAPE).
  3. Compares the forecast with the same weeks last year.
  4. If the forecast is below last year, it sizes three recovery levers
     - Price, Promotion and Distribution - from your own data and recommends
     the one that closes the most of the decline for the best revenue result.
"""

import math
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


# ============================================================
# 1. SETTINGS
# ============================================================
HORIZON_OPTIONS = [4, 8, 12]
LY_MATCH_TOLERANCE_DAYS = 3          # weekly data: LY week must be within 3 days of t-52w
SEASONAL_CLIP = (0.50, 1.50)         # cap the influence of an abnormal LY week
MAX_BACKTEST_ORIGINS = 12            # rolling-origin folds per model (speed vs robustness)
SEASONAL_SUFFIX = " + 52-Wk Seasonality"
TIE_BAND_PCT = 10.0                  # levers within 10 pts of gap closure compete on revenue


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

BASE_COLUMNS = ["Full Date", "Brand", "Category", "Product", "ProductsID", "Subcategory"]


# ============================================================
# 3. PREPROCESSING & SCALE ENFORCEMENT
# ============================================================
def clean_number(series):
    return pd.to_numeric(series, errors="coerce")


def detect_measure_set(columns):
    columns = set(columns)
    m52 = MEASURE_SETS["52-week"]
    if all(c in columns for c in [m52["value"], m52["price"], m52["promo"]]):
        if m52["baseline"] in columns:
            return "52-week", m52, None
        return "52-week-missing-baseline", m52, m52["baseline"]

    m26 = MEASURE_SETS["26-week"]
    if all(c in columns for c in [m26["value"], m26["price"], m26["promo"]]):
        return "26-week", m26, None

    return None, None, None


def validate_and_prepare(raw_df):
    """Validate the raw CSV extract and create clean, auto-scaled analytical fields."""
    errors, warnings_list = [], []

    missing_base = [c for c in BASE_COLUMNS if c not in raw_df.columns]
    if missing_base:
        errors.append("Missing required columns: " + ", ".join(missing_base))
        return False, errors, warnings_list, None, None

    measure_window, measures, missing_measure = detect_measure_set(raw_df.columns)
    if measure_window == "52-week-missing-baseline":
        errors.append(f"Required baseline column '{missing_measure}' is missing from the extract.")
        return False, errors, warnings_list, None, None
    if measures is None:
        errors.append("Could not find the required sales measure columns.")
        return False, errors, warnings_list, None, None

    df = raw_df.copy()

    df["date_key"] = pd.to_datetime(df["Full Date"], errors="coerce")
    bad_dates = int(df["date_key"].isna().sum())
    if bad_dates:
        warnings_list.append(f"{bad_dates:,} rows have invalid dates and were removed.")
        df = df.dropna(subset=["date_key"]).copy()
    if df.empty:
        return False, ["No valid dated rows remain after date validation."], warnings_list, None, None

    df["Sales Value"] = clean_number(df[measures["value"]]).fillna(0).clip(lower=0)
    df["Ave RSP"] = clean_number(df[measures["price"]])
    df["Promo RSP"] = clean_number(df[measures["promo"]])

    df["Sales Units"] = np.where(df["Ave RSP"] > 0, df["Sales Value"] / df["Ave RSP"], np.nan)
    df["Sales Units"] = pd.to_numeric(df["Sales Units"], errors="coerce").fillna(0).clip(lower=0)

    # ---- Baseline scale detection (choose the interpretation closest to actual units)
    if measures["baseline"] is not None and measures["baseline"] in df.columns:
        raw_b = (
            clean_number(df[measures["baseline"]])
            .replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        )
        valid_mask = (df["Sales Units"] > 0) & (df["Ave RSP"] > 0) & (raw_b > 0)

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
                    f"Median baseline/actual ratio = {chosen_ratio:.2f}x."
                )
            else:
                df["Baseline Units"] = df["Sales Units"]
                df["Baseline Value"] = df["Sales Value"]
                warnings_list.append("Baseline could not be reliably scaled; actual sales units are used as the baseline.")
        else:
            df["Baseline Units"] = df["Sales Units"]
            df["Baseline Value"] = df["Sales Value"]
            warnings_list.append("Baseline had no usable overlap with actual sales; actual sales units are used as the baseline.")

        df["Baseline Units"] = pd.to_numeric(df["Baseline Units"], errors="coerce").fillna(0).clip(lower=0)
        df["Baseline Value"] = pd.to_numeric(df["Baseline Value"], errors="coerce").fillna(0).clip(lower=0)
        df["Incremental Units"] = df["Sales Units"] - df["Baseline Units"]
    else:
        df["Baseline Units"] = df["Sales Units"]
        df["Baseline Value"] = df["Sales Value"]
        df["Incremental Units"] = 0.0

    # ---- Distribution: never invent a value. Missing data disables the distribution lever.
    if "Numeric Distribution" in df.columns:
        dist = clean_number(df["Numeric Distribution"])
        if not dist.dropna().empty and dist.dropna().max() <= 1.0:
            dist = dist * 100
        df["Numeric Distribution"] = dist.clip(0, 100)
        if dist.dropna().empty:
            warnings_list.append("'Numeric Distribution' is empty; distribution recommendations are disabled.")
    else:
        df["Numeric Distribution"] = np.nan
        warnings_list.append("No 'Numeric Distribution' column found; distribution recommendations are disabled.")

    for col in ["Category", "Subcategory", "Brand", "Product"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["ProductsID"] = df["ProductsID"].fillna("Unknown").astype(str)

    return True, errors, warnings_list, df.sort_values("date_key").reset_index(drop=True), measure_window


@st.cache_data(show_spinner=False)
def prepare_csv(csv_bytes):
    raw = pd.read_csv(pd.io.common.BytesIO(csv_bytes), low_memory=False)
    return validate_and_prepare(raw)


# ============================================================
# 4. WEEKLY AGGREGATION & PROMO-WEEK DETECTION
# ============================================================
def baseline_is_distinct(weekly):
    """True when the baseline is a real measure rather than a copy of actual sales."""
    act = weekly["Sales_Units"]
    base = weekly["Baseline_Units"]
    differs = (base - act).abs() > 1e-6 * act.abs().clip(lower=1)
    return bool(differs.mean() > 0.5)


def flag_promo_weeks(weekly):
    """A week counts as promoted if it shows a real promo discount or a clear lift over baseline."""
    flags = pd.Series(False, index=weekly.index)

    depth = weekly["Promo_Depth_%"].where(weekly["Promo_RSP_Weighted"].notna())
    depth = depth.fillna(weekly["Promo_Depth_%"])
    depth_flags = (depth >= 5).fillna(False)
    # If almost every week looks "on promo", the promo RSP field is not informative.
    if depth.notna().sum() >= 4 and depth_flags.mean() <= 0.6:
        flags |= depth_flags

    if baseline_is_distinct(weekly):
        lift = weekly["Sales_Units"] / weekly["Baseline_Units"].where(weekly["Baseline_Units"] > 0)
        flags |= (lift >= 1.10).fillna(False)

    return flags


def aggregate_weekly(df):
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    for col, default_val in [
        ("Sales Units", 0.0), ("Sales Value", 0.0), ("Baseline Units", 0.0),
        ("Baseline Value", 0.0), ("Incremental Units", 0.0), ("Numeric Distribution", np.nan),
    ]:
        if col not in df_copy.columns:
            df_copy[col] = default_val

    base = (
        df_copy.groupby("date_key", as_index=False)
        .agg(
            Sales_Units=("Sales Units", "sum"),
            Baseline_Units=("Baseline Units", "sum"),
            Sales_Value=("Sales Value", "sum"),
            Baseline_Value=("Baseline Value", "sum"),
            Distribution=("Numeric Distribution", "mean"),
        )
        .sort_values("date_key")
        .reset_index(drop=True)
    )
    base["Incremental_Units"] = base["Sales_Units"] - base["Baseline_Units"]
    base["Effective_RSP"] = np.where(base["Sales_Units"] > 0, base["Sales_Value"] / base["Sales_Units"], np.nan)

    base["Promo_RSP_Weighted"] = np.nan
    if "Promo RSP" in df_copy.columns:
        promo_part = df_copy.copy()
        promo_part["Promo_RSP"] = pd.to_numeric(promo_part["Promo RSP"], errors="coerce")
        promo_part = promo_part[(promo_part["Promo_RSP"] > 0) & (promo_part["Sales Units"] > 0)]
        if not promo_part.empty:
            promo_part = promo_part.assign(Promo_Value_Proxy=promo_part["Promo_RSP"] * promo_part["Sales Units"])
            promo_week = promo_part.groupby("date_key").agg(
                Promo_Value_Proxy=("Promo_Value_Proxy", "sum"), Promo_Units=("Sales Units", "sum")
            ).reset_index()
            promo_week["Promo_RSP_Weighted"] = np.where(
                promo_week["Promo_Units"] > 0, promo_week["Promo_Value_Proxy"] / promo_week["Promo_Units"], np.nan
            )
            base = base.drop(columns="Promo_RSP_Weighted").merge(
                promo_week[["date_key", "Promo_RSP_Weighted"]], on="date_key", how="left"
            )

    # Regular shelf price = rolling median of effective price (promo weeks are the minority,
    # so the median ignores them). Promo depth is measured against this regular price,
    # because the effective price in a promo week is already discounted.
    base["Regular_RSP"] = base["Effective_RSP"].rolling(9, center=True, min_periods=3).median()
    base["Regular_RSP"] = base["Regular_RSP"].fillna(base["Effective_RSP"])
    promo_price = base["Promo_RSP_Weighted"].fillna(base["Effective_RSP"])
    base["Promo_Depth_%"] = np.where(
        base["Regular_RSP"] > 0,
        (base["Regular_RSP"] - promo_price) / base["Regular_RSP"] * 100,
        np.nan,
    )
    base["Promo_Depth_%"] = base["Promo_Depth_%"].replace([np.inf, -np.inf], np.nan).clip(0, 100)
    base["Promo_Week"] = flag_promo_weeks(base)
    return base


def choose_forecast_series(weekly):
    """Use the baseline only when its scale is credible versus actual sales."""
    actual = pd.to_numeric(weekly["Sales_Units"], errors="coerce").fillna(0)
    baseline = pd.to_numeric(weekly["Baseline_Units"], errors="coerce").fillna(0)

    if not baseline_is_distinct(weekly):
        return actual.values, "Actual Sales Units", np.nan

    mask = (actual > 0) & (baseline > 0)
    if mask.sum() < 8:
        return actual.values, "Actual Sales Units (baseline rejected)", np.nan

    ratio = (baseline[mask] / actual[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    median_ratio = float(ratio.median()) if not ratio.empty else np.nan
    if np.isfinite(median_ratio) and 0.40 <= median_ratio <= 3.00:
        return baseline.values, "Baseline Units", median_ratio
    return actual.values, "Actual Sales Units (baseline rejected)", median_ratio


# ============================================================
# 5. FORECASTING MODELS, SEASONALITY & CHAMPION SELECTION
# ============================================================
def _flat(value, horizon):
    return np.repeat(max(0.0, float(value)), horizon)


def forecast_naive(y, horizon):
    return _flat(y[-1], horizon)


def forecast_ma4(y, horizon):
    return _flat(np.mean(y[-min(4, len(y)):]), horizon)


def forecast_ses(y, horizon):
    if len(y) < 3 or np.nanstd(y) < 1e-10:
        return forecast_naive(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_damped_holt(y, horizon):
    if len(y) < 6 or np.nanstd(y) < 1e-10:
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Holt(y, initialization_method="estimated", damped_trend=True).fit(optimized=True)
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


def forecast_theta(y, horizon):
    if len(y) < 8 or np.nanstd(y) < 1e-10:
        return forecast_ses(y, horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThetaModel(np.asarray(y, dtype=float), period=1, deseasonalize=False).fit()
    return np.maximum(0.0, np.asarray(fit.forecast(horizon), dtype=float))


BASE_MODELS = {
    "Naive Last Week": forecast_naive,
    "4-Week Moving Average": forecast_ma4,
    "Simple Exponential Smoothing": forecast_ses,
    "Damped Holt": forecast_damped_holt,
    "Theta": forecast_theta,
}


def candidate_names():
    """Every base model is tested with and without a 52-week seasonal index."""
    names = []
    for base_name in BASE_MODELS:
        names += [base_name, base_name + SEASONAL_SUFFIX]
    return names


def _value_near(series_by_date, target, tol_days=LY_MATCH_TOLERANCE_DAYS):
    """Value of a date-indexed series at the date closest to target (within tolerance)."""
    if series_by_date is None or series_by_date.empty:
        return np.nan
    target = pd.Timestamp(target)
    pos = series_by_date.index.get_indexer([target], method="nearest")[0]
    if pos < 0 or abs((series_by_date.index[pos] - target).days) > tol_days:
        return np.nan
    value = series_by_date.iloc[pos]
    return float(value) if pd.notna(value) else np.nan


def seasonal_factors(y, dates, future_dates, anchor_weeks=4):
    """
    52-week seasonal index relative to *now*:
        LY value of the target week / LY average of the last `anchor_weeks` history weeks.
    Level models (naive, MA, SES...) forecast today's level; this index moves that level
    along last year's seasonal shape. Returns None when there is not enough history.
    """
    hist = pd.Series(np.asarray(y, dtype=float), index=pd.DatetimeIndex(dates))
    hist = hist.rolling(3, center=True, min_periods=1).median()  # damp one-off spikes

    anchors = [_value_near(hist, d - timedelta(weeks=52)) for d in list(hist.index[-anchor_weeks:])]
    anchors = [a for a in anchors if np.isfinite(a) and a > 0]
    if len(anchors) < 2:
        return None
    reference = float(np.mean(anchors))

    factors = []
    for d in future_dates:
        ly_val = _value_near(hist, d - timedelta(weeks=52))
        if np.isfinite(ly_val) and ly_val > 0:
            factors.append(float(np.clip(ly_val / reference, *SEASONAL_CLIP)))
        else:
            factors.append(1.0)
    return np.array(factors)


def predict_candidate(name, y, dates, horizon):
    """Forecast with one candidate. Returns (forecast, seasonal_factors) or None if not possible."""
    y = np.asarray(y, dtype=float)
    dates = pd.DatetimeIndex(dates)
    seasonal = name.endswith(SEASONAL_SUFFIX)
    base_fcst = BASE_MODELS[name.replace(SEASONAL_SUFFIX, "")](y, horizon)

    if not seasonal:
        return np.maximum(0.0, base_fcst), np.ones(horizon)

    future = [dates[-1] + timedelta(weeks=i) for i in range(1, horizon + 1)]
    factors = seasonal_factors(y, dates, future)
    if factors is None:
        return None
    return np.maximum(0.0, base_fcst * factors), factors


@st.cache_data(show_spinner=False)
def select_best_model(y_tuple, dates_tuple, horizon):
    """
    Rolling-origin backtest at the chosen horizon. Every eligible model is scored on the
    SAME folds; the one with the lowest WMAPE wins (ties broken by smaller bias).
    """
    y = np.asarray(y_tuple, dtype=float)
    dates = pd.DatetimeIndex(dates_tuple)
    n = len(y)
    min_train = max(8, min(26, n // 2))
    origins = list(range(min_train, n - horizon + 1))[-MAX_BACKTEST_ORIGINS:]

    fallback = {
        "champion": "4-Week Moving Average",
        "wmape": np.nan,
        "leaderboard": pd.DataFrame(),
        "ratio_q": None,
        "n_folds": 0,
        "weeks_needed": min_train + horizon + 2,
    }
    if len(origins) < 3:
        return fallback

    folds = {}
    for name in candidate_names():
        folds[name] = {}
        for o in origins:
            try:
                out = predict_candidate(name, y[:o], dates[:o], horizon)
            except Exception:
                out = None
            if out is not None:
                folds[name][o] = (out[0], y[o:o + horizon])

    # Keep a common fold set so every model is judged on identical weeks.
    min_folds = min(6, len(origins))
    eligible = [nm for nm in folds if len(folds[nm]) >= min_folds]

    def common_of(names):
        common = set(origins)
        for nm in names:
            common &= set(folds[nm])
        return common

    common = common_of(eligible)
    while eligible and len(common) < min_folds:
        eligible.remove(min(eligible, key=lambda nm: len(folds[nm])))
        common = common_of(eligible)

    rows, pooled = [], {}
    for nm in eligible:
        preds = np.concatenate([folds[nm][o][0] for o in sorted(common)])
        acts = np.concatenate([folds[nm][o][1] for o in sorted(common)])
        if acts.sum() <= 0:
            continue
        wmape = np.abs(preds - acts).sum() / acts.sum() * 100
        bias = (preds.sum() - acts.sum()) / acts.sum() * 100
        rows.append({"Model": nm, "WMAPE %": wmape, "Bias %": bias, "Backtest folds": len(common)})
        pooled[nm] = (preds, acts)

    if not rows:
        return fallback

    board = pd.DataFrame(rows)
    board["_abs_bias"] = board["Bias %"].abs()
    board = board.sort_values(["WMAPE %", "_abs_bias"]).drop(columns="_abs_bias").reset_index(drop=True)
    champion = board.loc[0, "Model"]

    # Empirical 80% range from the champion's own backtest misses.
    preds, acts = pooled[champion]
    ok = preds > 0
    ratios = acts[ok] / preds[ok]
    ratio_q = (float(np.quantile(ratios, 0.10)), float(np.quantile(ratios, 0.90))) if len(ratios) >= 8 else None

    return {
        "champion": champion,
        "wmape": float(board.loc[0, "WMAPE %"]),
        "leaderboard": board,
        "ratio_q": ratio_q,
        "n_folds": len(common),
        "weeks_needed": min_train + horizon + 2,
    }


# ============================================================
# 6. LAST-YEAR BENCHMARK, PROMO REPEAT & SA MACRO ADJUSTMENT
# ============================================================
def ly_benchmarks(weekly, future_dates):
    """Same week last year (t-52w) for each forecast week."""
    indexed = weekly.set_index("date_key").sort_index()
    cols = ["Sales_Units", "Sales_Value", "Baseline_Units", "Distribution", "Promo_Week"]
    out = {c: [] for c in cols}
    out["LY_Date"] = []
    for d in future_dates:
        target = d - timedelta(weeks=52)
        for c in cols:
            out[c].append(_value_near(indexed[c].astype(float), target))
        out["LY_Date"].append(target)
    return {k: (np.array(v, dtype=float) if k != "LY_Date" else v) for k, v in out.items()}


def apply_overlays(model_fcst, ly, apply_promo_repeat, use_macro, price_yoy, cpi_inflation, elasticity):
    """
    Final forecast = champion model forecast (already seasonal if the champion uses seasonality)
                     x last year's promo lift (only when forecasting baseline)
                     x SA macro factor (price increase above CPI).
    """
    lifts = np.ones(len(model_fcst))
    if apply_promo_repeat:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = ly["Sales_Units"] / np.where(ly["Baseline_Units"] > 0, ly["Baseline_Units"], np.nan)
        lifts = np.where(np.isfinite(raw) & (raw >= 1.0) & (raw <= 3.0), raw, 1.0)

    squeeze = max(0.0, price_yoy - cpi_inflation)
    macro = float(np.clip(1.0 + elasticity * squeeze / 100.0, 0.5, 1.5)) if use_macro else 1.0
    return np.maximum(0.0, model_fcst * lifts * macro), lifts, macro


def current_rsp(weekly):
    """Regular (non-promo) shelf price from recent weeks, so a promo week doesn't distort revenue."""
    recent = weekly.tail(8)
    regular = recent.loc[~recent["Promo_Week"] & (recent["Effective_RSP"] > 0), "Effective_RSP"]
    if len(regular) >= 2:
        return float(regular.tail(4).median())
    if weekly["Regular_RSP"].notna().any():
        return float(weekly["Regular_RSP"].dropna().iloc[-1])
    valid = weekly.loc[weekly["Effective_RSP"] > 0, "Effective_RSP"]
    return float(valid.tail(4).median()) if not valid.empty else np.nan


# ============================================================
# 7. RECOVERY ENGINE: PRICE vs PROMOTION vs DISTRIBUTION
# ============================================================
def estimate_elasticity(weekly, fallback):
    """
    Shelf-price elasticity from NON-promo weeks (promo weeks mix price with display and
    feature effects). The regression controls for trend and annual seasonality so that a
    price rise that simply coincides with a declining trend is not mistaken for price
    response. Falls back to the user's setting when the estimate isn't trustworthy.
    """
    w = weekly[(weekly["Sales_Units"] > 0) & (weekly["Effective_RSP"] > 0) & ~weekly["Promo_Week"]]
    if len(w) < 20:
        return fallback, "your setting (not enough regular-price weeks to estimate)"

    t = (w["date_key"] - w["date_key"].min()).dt.days.values / 7.0
    controls = np.column_stack([np.ones(len(w)), t, np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)])
    log_p, log_q = np.log(w["Effective_RSP"].values), np.log(w["Sales_Units"].values)

    # Price must move independently of trend/season for the effect to be identifiable.
    p_resid = log_p - controls @ np.linalg.lstsq(controls, log_p, rcond=None)[0]
    if np.std(p_resid) < 0.01:
        return fallback, "your setting (price only moved with the trend, so its effect can't be isolated)"

    X = np.column_stack([controls, log_p])
    coef, *_ = np.linalg.lstsq(X, log_q, rcond=None)
    resid = log_q - X @ coef
    dof = max(1, len(w) - X.shape[1])
    sigma2 = resid @ resid / dof
    try:
        se = math.sqrt(sigma2 * np.linalg.inv(X.T @ X)[-1, -1])
    except np.linalg.LinAlgError:
        return fallback, "your setting (the data estimate was not reliable)"
    slope = float(coef[-1])
    if -3.0 <= slope <= -0.2 and se > 0 and abs(slope / se) >= 2:
        return slope, f"estimated from your regular-price weeks (±{1.96 * se:.2f})"
    return fallback, "your setting (the data estimate was not reliable)"


def historical_promo_response(weekly):
    """Typical promo lift and discount depth measured from your own promoted weeks."""
    flags = weekly["Promo_Week"]
    promo = weekly[flags & (weekly["Sales_Units"] > 0)]
    regular = weekly[~flags & (weekly["Sales_Units"] > 0)]
    if len(promo) < 2 or len(regular) < 4:
        return None

    if baseline_is_distinct(weekly):
        ratio = promo["Sales_Units"] / promo["Baseline_Units"].where(promo["Baseline_Units"] > 0)
        lift = float(ratio.replace([np.inf, -np.inf], np.nan).dropna().median())
        method = "promo-week sales vs baseline"
    else:
        lift = float(promo["Sales_Units"].median() / regular["Sales_Units"].median())
        method = "promo-week sales vs regular weeks"

    depth_obs = promo["Promo_Depth_%"].dropna()
    depth = float(depth_obs.median()) if len(depth_obs) >= 2 else np.nan

    if not np.isfinite(lift) or not (1.03 <= lift <= 4.0) or not np.isfinite(depth) or depth < 1:
        return None
    return {"lift": lift, "depth": float(np.clip(depth, 0, 60)), "n_weeks": int(len(promo)), "method": method}


def sku_distribution_gaps(df, last_date, top_n=5):
    """SKUs whose numeric distribution today is below the same week last year."""
    d = df.dropna(subset=["Numeric Distribution"])
    if d.empty:
        return pd.DataFrame()
    ly_date = last_date - timedelta(weeks=52)
    rows = []
    for prod, grp in d.groupby("Product"):
        s = grp.groupby("date_key")["Numeric Distribution"].mean().sort_index()
        now, ly = _value_near(s, last_date), _value_near(s, ly_date)
        if np.isfinite(now) and np.isfinite(ly) and ly - now >= 1:
            rows.append({"SKU": prod, "Distribution LY %": ly, "Distribution now %": now, "Points lost": ly - now})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Points lost", ascending=False).head(top_n)


def _unavailable(lever, reason):
    return {"lever": lever, "available": False, "units": 0.0, "closure": 0.0,
            "net_rev": 0.0, "action": "-", "why": reason}


def evaluate_price_lever(gap, fcst_units, rsp, elasticity, elasticity_source, max_cut_pct, n_weeks, squeeze):
    eps = abs(elasticity)
    if fcst_units <= 0 or rsp <= 0 or eps < 0.05:
        return _unavailable("Price", "Price response could not be measured.")

    needed = gap / (fcst_units * eps)               # %ΔQ ≈ ε × %ΔP
    cut = min(needed, max_cut_pct / 100.0)
    recovered = min(gap, fcst_units * eps * cut)
    new_rsp = rsp * (1 - cut)
    net_rev = (fcst_units + recovered) * new_rsp - fcst_units * rsp

    why = (f"Each 1% off the shelf price adds about {eps:.1f}% volume "
           f"(elasticity {elasticity:.2f}, {elasticity_source}).")
    if squeeze > 0:
        why += f" Your price increase is running {squeeze:.1f} pts ahead of food CPI, which is part of the decline."
    if needed > cut:
        why += f" Closing the full gap would need a {needed * 100:.1f}% cut, above your {max_cut_pct:.0f}% limit."

    return {
        "lever": "Price", "available": True, "units": recovered,
        "closure": recovered / gap * 100, "net_rev": net_rev,
        "action": (f"Reduce the shelf price by {cut * 100:.1f}% "
                   f"(R {rsp:,.2f} → R {new_rsp:,.2f}) for the {n_weeks} forecast weeks."),
        "why": why,
    }


def evaluate_promo_lever(gap, fcst_weekly, shortfall_weekly, matched, overlay_lifts,
                         future_dates, rsp, promo_resp, max_promo_weeks):
    if promo_resp is None:
        return _unavailable("Promotion", "Not enough promoted weeks in the history to measure a promo response.")
    if max_promo_weeks < 1:
        return _unavailable("Promotion", "Promotion weeks are capped at zero in the settings.")

    lift, depth = promo_resp["lift"], promo_resp["depth"]
    # Only weeks not already carrying last year's promo, biggest shortfall first.
    candidates = [i for i in range(len(fcst_weekly))
                  if matched[i] and overlay_lifts[i] <= 1.0001 and fcst_weekly[i] > 0]
    candidates.sort(key=lambda i: -shortfall_weekly[i])
    if not candidates:
        return _unavailable("Promotion", "Every forecast week already carries last year's promotion.")

    chosen, recovered, net_rev = [], 0.0, 0.0
    for i in candidates:
        if recovered >= gap or len(chosen) >= max_promo_weeks:
            break
        chosen.append(i)
        recovered += fcst_weekly[i] * (lift - 1)
        net_rev += fcst_weekly[i] * lift * rsp * (1 - depth / 100) - fcst_weekly[i] * rsp
    recovered = min(recovered, gap)

    weeks_txt = ", ".join(future_dates[i].strftime("%d %b") for i in sorted(chosen))
    promo_price = rsp * (1 - depth / 100)
    return {
        "lever": "Promotion", "available": True, "units": recovered,
        "closure": recovered / gap * 100, "net_rev": net_rev,
        "action": (f"Run {len(chosen)} promo week(s) at about {depth:.0f}% off "
                   f"(R {promo_price:,.2f}) in the weeks starting {weeks_txt}."),
        "why": (f"Your past promotions lifted volume about {lift:.2f}x at a {depth:.0f}% discount "
                f"({promo_resp['n_weeks']} promoted weeks, {promo_resp['method']}). "
                "Weeks with the biggest shortfall against last year are targeted first."),
    }


def evaluate_distribution_lever(gap, fcst_units, weekly, ly, matched, rsp, sku_gaps):
    dist = weekly["Distribution"].dropna()
    if dist.empty:
        return _unavailable("Distribution", "No numeric distribution data in the extract.")

    current = float(dist.iloc[-1])
    ly_dist = float(np.nanmean(ly["Distribution"][matched])) if np.isfinite(ly["Distribution"][matched]).any() else np.nan
    recent_best = float(dist.tail(13).max())
    target = float(np.nanmax([ly_dist, recent_best]))
    points = target - current
    if current <= 0 or points < 1:
        return _unavailable(
            "Distribution",
            f"Distribution ({current:.1f}%) is already at or above last year and its recent best; no gap to recover.",
        )

    recovered = min(gap, fcst_units / current * points)  # volume scales with stores stocking
    why = (f"Numeric distribution is {current:.1f}% vs {ly_dist:.1f}% in the same weeks last year "
           f"(recent best {recent_best:.1f}%). Volume is assumed to scale with the share of stores stocking.")
    if not sku_gaps.empty:
        why += " Biggest losses: " + "; ".join(
            f"{r.SKU} ({r['Points lost']:.1f} pts)" for _, r in sku_gaps.head(3).iterrows()
        ) + "."
    return {
        "lever": "Distribution", "available": True, "units": recovered,
        "closure": recovered / gap * 100, "net_rev": recovered * rsp,
        "action": (f"Restore numeric distribution from {current:.1f}% to {target:.1f}% "
                   f"(+{points:.1f} pts): fix out-of-stocks and regain lost listings."),
        "why": why,
    }


def rank_levers(levers):
    """
    Best lever = the one that closes the most of the decline. When two levers close a
    similar share (within TIE_BAND_PCT points), the one with the better revenue result wins.
    """
    usable = [lv for lv in levers if lv["available"] and lv["units"] > 0]
    if not usable:
        return None, None
    usable.sort(key=lambda lv: (-lv["closure"], -lv["net_rev"]))
    best = usable[0]
    for lv in usable[1:]:
        if lv["closure"] >= best["closure"] - TIE_BAND_PCT and lv["net_rev"] > best["net_rev"]:
            best = lv

    support = None
    if best["closure"] < 95:
        remaining = (100 - best["closure"])
        others = [lv for lv in usable if lv is not best]
        if others:
            support = max(others, key=lambda lv: (min(lv["closure"], remaining), lv["net_rev"]))
    return best, support


# ============================================================
# 8. PAGE SETUP & STYLING
# ============================================================
st.set_page_config(page_title="Unilever Demand & Recovery Planner", page_icon="📦", layout="wide")

st.markdown(
    """
    <style>
    .main-header {font-size: 26px; font-weight: 700; color: #1F36C7; margin-bottom: 2px;}
    .sub-header {font-size: 14px; color: #475569; margin-bottom: 14px;}
    [data-testid="stMetricValue"] {font-size: 20px !important; font-weight: 600 !important; color: #0F172A !important;}
    [data-testid="stMetricLabel"] {font-size: 13px !important; color: #475569 !important;}
    .verdict {padding: 16px 20px; border-radius: 8px; margin: 6px 0 16px 0;}
    .verdict h4 {margin: 0 0 4px 0;}
    .verdict p {margin: 0; font-size: 14px;}
    .verdict-bad {background: #FEF2F2; border: 1px solid #FCA5A5;}
    .verdict-bad h4 {color: #991B1B;} .verdict-bad p {color: #7F1D1D;}
    .verdict-good {background: #F0FDF4; border: 1px solid #86EFAC;}
    .verdict-good h4 {color: #166534;} .verdict-good p {color: #14532D;}
    .rec-primary {background: #EEF2FF; border-left: 5px solid #1F36C7; padding: 14px 18px; border-radius: 4px; margin-bottom: 10px;}
    .rec-support {background: #F8FAFC; border-left: 5px solid #94A3B8; padding: 12px 18px; border-radius: 4px; margin-bottom: 10px;}
    .rec-lever {font-size: 13px; color: #475569; margin-bottom: 2px;}
    .rec-action {font-size: 16px; font-weight: 600; color: #0F172A; margin-bottom: 6px;}
    .rec-why {font-size: 14px; color: #334155;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">Unilever Demand & Recovery Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Forecast the next 4, 8 or 12 weeks, compare with the same weeks last year, '
    "and get the best price, promotion or distribution action to close any decline.</div>",
    unsafe_allow_html=True,
)


# ============================================================
# 9. SIDEBAR: UPLOAD, FILTERS, HORIZON, SETTINGS
# ============================================================
with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader("Weekly sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload your weekly sales CSV in the sidebar to start. "
            "At least 52 weeks of history is needed for last-year comparisons; 60+ weeks gives the best results.")
    st.stop()

try:
    valid, errors, warning_list, df_clean, measure_window = prepare_csv(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

if not valid:
    st.error("The file doesn't match the expected format.")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

with st.sidebar:
    st.header("2. Choose what to forecast")
    categories = ["All"] + sorted(df_clean["Category"].unique().tolist())
    category = st.selectbox("Category", categories)
    d1 = df_clean if category == "All" else df_clean[df_clean["Category"] == category]

    subcategories = ["All"] + sorted(d1["Subcategory"].unique().tolist())
    subcategory = st.selectbox("Subcategory", subcategories)
    d2 = d1 if subcategory == "All" else d1[d1["Subcategory"] == subcategory]

    brands = ["All"] + sorted(d2["Brand"].unique().tolist())
    brand = st.selectbox("Brand", brands)
    d3 = d2 if brand == "All" else d2[d2["Brand"] == brand]

    product_options = ["All"] + sorted(d3["Product"].unique().tolist())
    selected_products = st.multiselect("Product SKU(s)", options=product_options, default=["All"])

    if "All" in selected_products or not selected_products:
        final_df = d3.copy()
        product_label = f"All products in {brand if brand != 'All' else 'portfolio'}"
    else:
        final_df = d3[d3["Product"].isin(selected_products)].copy()
        product_label = selected_products[0] if len(selected_products) == 1 else f"{len(selected_products)} selected SKUs"

    st.header("3. Forecast horizon")
    forecast_horizon = st.radio(
        "Weeks to forecast", HORIZON_OPTIONS, index=0, horizontal=True,
        format_func=lambda w: f"{w} weeks", label_visibility="collapsed",
    )

    with st.expander("⚙️ Advanced settings"):
        st.markdown("**SA macro adjustment**")
        use_macro = st.checkbox("Adjust for price increase above CPI", value=True,
                                help="Reduces the forecast when your price increase outpaces food inflation.")
        price_yoy = st.number_input("Unilever YoY price increase (%)", 0.0, 30.0, 8.5, 0.5, disabled=not use_macro)
        cpi_inflation = st.number_input("SA food CPI inflation (%)", 0.0, 30.0, 5.2, 0.5, disabled=not use_macro)
        elasticity_input = st.slider("Price elasticity (fallback)", -2.5, -0.1, -1.1, 0.1,
                                     help="Used when elasticity can't be estimated reliably from your data.")
        st.markdown("**Promotions**")
        include_ly_promos = st.checkbox("Assume last year's promotions repeat", value=True,
                                        help="Only applies when the forecast is built on baseline sales.")
        st.markdown("**Recommendation limits**")
        max_price_cut = st.slider("Largest price cut to consider (%)", 2, 25, 10)
        max_promo_share = st.slider("Max share of weeks on promotion (%)", 25, 100, 50, step=25)

if final_df.empty:
    st.warning("No data matches this filter combination. Widen one of the filters in the sidebar.")
    st.stop()


# ============================================================
# 10. FORECAST COMPUTATION
# ============================================================
weekly = aggregate_weekly(final_df)
if len(weekly) < 4:
    st.error("At least 4 weeks of history are needed for this selection. Widen the filters or upload more weeks.")
    st.stop()

last_date = weekly["date_key"].max()
future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon + 1)]

forecast_series, forecast_source, baseline_ratio = choose_forecast_series(weekly)
dates_tuple = tuple(weekly["date_key"])

with st.spinner(f"Testing {len(candidate_names())} models on your history..."):
    selection = select_best_model(tuple(float(v) for v in forecast_series), dates_tuple, forecast_horizon)

champion = selection["champion"]
prediction = predict_candidate(champion, forecast_series, dates_tuple, forecast_horizon)
if prediction is None:  # safety net; the champion always ran in the backtest
    champion = "4-Week Moving Average"
    prediction = predict_candidate(champion, forecast_series, dates_tuple, forecast_horizon)
model_fcst, season_idx = prediction

ly = ly_benchmarks(weekly, future_dates)
apply_promo_repeat = forecast_source == "Baseline Units" and include_ly_promos
final_units, promo_lifts, macro_factor = apply_overlays(
    model_fcst, ly, apply_promo_repeat, use_macro, price_yoy, cpi_inflation, elasticity_input
)

rsp_now = current_rsp(weekly)
if not np.isfinite(rsp_now) or rsp_now <= 0:
    st.error("No valid price data for this selection, so revenue can't be forecast.")
    st.stop()
final_value = final_units * rsp_now

if selection["ratio_q"] is not None:
    q_lo, q_hi = selection["ratio_q"]
    range_label = "80% range (from backtest errors)"
elif np.isfinite(selection["wmape"]):
    q_lo, q_hi = max(0.0, 1 - selection["wmape"] / 100), 1 + selection["wmape"] / 100
    range_label = "Range (± backtest WMAPE)"
else:
    q_lo, q_hi = 0.8, 1.2
    range_label = "Indicative range (±20%, no backtest)"
lower_units, upper_units = final_units * min(q_lo, 1.0), final_units * max(q_hi, 1.0)

# ---- Comparison with the same weeks last year (matched weeks only)
matched = np.isfinite(ly["Sales_Units"]) & np.isfinite(ly["Sales_Value"])
n_matched = int(matched.sum())
shortfall_weekly = np.where(matched, np.nan_to_num(ly["Sales_Units"]) - final_units, 0.0)

if n_matched:
    fc_units_m = float(final_units[matched].sum())
    fc_value_m = float(final_value[matched].sum())
    ly_units_m = float(ly["Sales_Units"][matched].sum())
    ly_value_m = float(ly["Sales_Value"][matched].sum())
    gap_units = ly_units_m - fc_units_m
    unit_var_pct = (fc_units_m - ly_units_m) / ly_units_m * 100 if ly_units_m > 0 else np.nan
    value_var = fc_value_m - ly_value_m
    value_var_pct = value_var / ly_value_m * 100 if ly_value_m > 0 else np.nan
else:
    fc_units_m = fc_value_m = ly_units_m = ly_value_m = gap_units = np.nan
    unit_var_pct = value_var = value_var_pct = np.nan

is_decline = n_matched > 0 and gap_units > 0


# ============================================================
# 11. HEADLINE NUMBERS
# ============================================================
scope_label = " › ".join([category, subcategory, brand, product_label])
st.markdown(f"#### {scope_label}")

if warning_list:
    with st.expander(f"ℹ️ Data notes ({len(warning_list)})"):
        for w in warning_list:
            st.write(f"- {w}")
if forecast_source != "Baseline Units" and np.isfinite(baseline_ratio):
    st.caption(f"Baseline not used for forecasting (median baseline/actual = {baseline_ratio:.2f}x); "
               "forecasting actual sales instead.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{forecast_horizon}-week volume forecast", f"{final_units.sum():,.0f} units")
c2.metric(f"{forecast_horizon}-week revenue forecast", f"R {final_value.sum():,.0f}")
c3.metric("Volume vs same weeks LY",
          f"{unit_var_pct:+.1f}%" if pd.notna(unit_var_pct) else "N/A",
          delta=f"{fc_units_m - ly_units_m:+,.0f} units" if n_matched else None)
c4.metric("Revenue vs same weeks LY",
          f"R {value_var:+,.0f}" if pd.notna(value_var) else "N/A",
          delta=f"{value_var_pct:+.1f}%" if pd.notna(value_var_pct) else None)
c5.metric("Forecast error (WMAPE)",
          f"{selection['wmape']:.1f}%" if np.isfinite(selection["wmape"]) else "N/A",
          help="Average backtest error of the chosen model at this horizon. Lower is better.")

if n_matched == 0:
    st.warning("None of the forecast weeks have a matching week last year in the uploaded data, so the "
               "year-on-year comparison and recommendations are unavailable. Upload at least 52 weeks of history.")
elif n_matched < forecast_horizon:
    st.warning(f"Only {n_matched} of {forecast_horizon} forecast weeks have a matching week last year. "
               "The comparison and recommendations cover those weeks only.")


# ============================================================
# 12. VERDICT & RECOMMENDATION
# ============================================================
if is_decline:
    st.markdown(
        f"""<div class='verdict verdict-bad'>
        <h4>Forecast is {abs(unit_var_pct):.1f}% below the same {n_matched} week(s) last year</h4>
        <p>That is {gap_units:,.0f} fewer units and R {max(0.0, -value_var):,.0f} less revenue than last year.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    elasticity_used, elasticity_source = estimate_elasticity(weekly, elasticity_input)
    promo_resp = historical_promo_response(weekly)
    sku_gaps = sku_distribution_gaps(final_df, last_date)
    max_promo_weeks = int(math.floor(forecast_horizon * max_promo_share / 100))

    levers = [
        evaluate_price_lever(gap_units, fc_units_m, rsp_now, elasticity_used, elasticity_source,
                             max_price_cut, n_matched, max(0.0, price_yoy - cpi_inflation) if use_macro else 0.0),
        evaluate_promo_lever(gap_units, final_units, shortfall_weekly, matched, promo_lifts,
                             future_dates, rsp_now, promo_resp, max_promo_weeks),
        evaluate_distribution_lever(gap_units, fc_units_m, weekly, ly, matched, rsp_now, sku_gaps),
    ]
    best, support = rank_levers(levers)

    st.subheader("Recommended action")
    if best is None:
        st.warning("None of the three levers could be sized from this data. Check that the extract includes "
                   "price history, promoted weeks and numeric distribution.")
    else:
        st.markdown(
            f"""<div class='rec-primary'>
            <div class='rec-lever'>Best lever: {best['lever']} · closes {min(best['closure'], 100):.0f}% of the decline
            (≈{best['units']:,.0f} units) · net revenue impact R {best['net_rev']:+,.0f}</div>
            <div class='rec-action'>{best['action']}</div>
            <div class='rec-why'>{best['why']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if support is not None:
            remaining_units = gap_units - best["units"]
            st.markdown(
                f"""<div class='rec-support'>
                <div class='rec-lever'>Add-on to close the remaining {remaining_units:,.0f} units: {support['lever']}</div>
                <div class='rec-action'>{support['action']}</div>
                <div class='rec-why'>{support['why']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("How the three levers compare", expanded=best is None):
        compare = pd.DataFrame([{
            "Lever": lv["lever"] + (" ✅" if best is not None and lv is best else ""),
            "Action": lv["action"] if lv["available"] else "Not available",
            "Units recovered": lv["units"],
            "% of decline closed": min(lv["closure"], 100.0),
            "Net revenue impact (R)": lv["net_rev"],
            "Reasoning": lv["why"],
        } for lv in levers])
        st.dataframe(
            compare.style.format({"Units recovered": "{:,.0f}", "% of decline closed": "{:.0f}%",
                                  "Net revenue impact (R)": "R {:+,.0f}"}),
            width="stretch", hide_index=True,
        )
        st.caption("The best lever closes the most of the decline; when two levers close a similar share "
                   f"(within {TIE_BAND_PCT:.0f} points), the one with the better revenue result wins. "
                   "Estimates come from your own history and should be validated with the customer before committing spend.")
elif n_matched:
    st.markdown(
        f"""<div class='verdict verdict-good'>
        <h4>On track: forecast is {unit_var_pct:+.1f}% vs the same {n_matched} week(s) last year</h4>
        <p>No recovery action is needed. Keep price, promotion and distribution execution in line with the plan.</p>
        </div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# 13. CHART
# ============================================================
st.subheader(f"History and {forecast_horizon}-week forecast")
view = st.radio("Show", ["Volume (units)", "Revenue (R)"], horizontal=True, label_visibility="collapsed")
is_units = view.startswith("Volume")

hist_y = weekly["Sales_Units"] if is_units else weekly["Sales_Value"]
fc_y = final_units if is_units else final_value
lo_y = lower_units if is_units else lower_units * rsp_now
hi_y = upper_units if is_units else upper_units * rsp_now
ly_y = ly["Sales_Units"] if is_units else ly["Sales_Value"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=future_dates + future_dates[::-1], y=list(hi_y) + list(lo_y)[::-1],
                         fill="toself", fillcolor="rgba(31, 54, 199, 0.12)", line=dict(color="rgba(0,0,0,0)"),
                         hoverinfo="skip", name=range_label))
fig.add_trace(go.Scatter(x=weekly["date_key"], y=hist_y, mode="lines+markers", name="Actual",
                         line=dict(color="#0F172A", width=2), marker=dict(size=4)))
if is_units and forecast_source == "Baseline Units":
    fig.add_trace(go.Scatter(x=weekly["date_key"], y=weekly["Baseline_Units"], mode="lines", name="Baseline",
                             line=dict(color="#94A3B8", width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=[weekly["date_key"].iloc[-1]] + future_dates,
                         y=[float(hist_y.iloc[-1])] + list(fc_y), mode="lines+markers",
                         name=f"Forecast ({champion})", line=dict(color="#1F36C7", width=3, dash="dash"),
                         marker=dict(size=7, symbol="diamond")))
fig.add_trace(go.Scatter(x=future_dates, y=ly_y, mode="lines+markers", name="Same week last year",
                         line=dict(color="#D97706", width=2, dash="dot"), marker=dict(size=6)))
fig.add_vline(x=weekly["date_key"].iloc[-1], line_width=1, line_dash="dot", line_color="#64748B")
fig.add_annotation(x=weekly["date_key"].iloc[-1], y=1.02, xref="x", yref="paper", text="Forecast starts →",
                   showarrow=False, font=dict(size=11, color="#475569"), xanchor="left")
fig.update_layout(
    template="plotly_white", height=520, hovermode="x unified", margin=dict(l=60, r=30, t=70, b=60),
    xaxis=dict(title="Week starting", showgrid=True),
    yaxis=dict(title="Units" if is_units else "Revenue (R)", showgrid=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)
st.plotly_chart(fig, width="stretch")


# ============================================================
# 14. WEEK-BY-WEEK TABLE
# ============================================================
st.subheader("Week-by-week forecast")
table_df = pd.DataFrame({
    "Week starting": [d.strftime("%Y-%m-%d") for d in future_dates],
    "Forecast units": final_units,
    "Range low": lower_units,
    "Range high": upper_units,
    "Forecast revenue": final_value,
    "Same week LY units": ly["Sales_Units"],
    "Same week LY revenue": ly["Sales_Value"],
    "Units vs LY": np.where(matched, final_units - np.nan_to_num(ly["Sales_Units"]), np.nan),
    "Seasonal index": season_idx,
    "LY promo lift": promo_lifts,
})


def _colour_gap(v):
    if pd.isna(v):
        return ""
    return "color: #B91C1C; font-weight: 600" if v < 0 else "color: #15803D"


st.dataframe(
    table_df.style.format({
        "Forecast units": "{:,.0f}", "Range low": "{:,.0f}", "Range high": "{:,.0f}",
        "Forecast revenue": "R {:,.0f}", "Same week LY units": "{:,.0f}",
        "Same week LY revenue": "R {:,.0f}", "Units vs LY": "{:+,.0f}",
        "Seasonal index": "{:.2f}", "LY promo lift": "{:.2f}x",
    }, na_rep="-").map(_colour_gap, subset=["Units vs LY"]),
    width="stretch", hide_index=True,
)
st.download_button("Download forecast (CSV)", table_df.to_csv(index=False).encode("utf-8"),
                   file_name=f"forecast_{forecast_horizon}wk_{last_date:%Y%m%d}.csv", mime="text/csv")


# ============================================================
# 15. MODEL SELECTION & METHOD
# ============================================================
with st.expander("🏆 How the model was chosen"):
    board = selection["leaderboard"]
    if board.empty:
        st.info(f"Not enough history to compare models at a {forecast_horizon}-week horizon "
                f"(about {selection['weeks_needed']} weeks needed). Using the 4-week moving average.")
    else:
        st.write(f"Each model forecast {forecast_horizon} weeks ahead from {selection['n_folds']} past points in "
                 "your history, and its error was measured against what actually sold. "
                 f"**{champion}** had the lowest error, so it is the only model used.")
        shown = board.copy()
        shown.insert(0, "", ["🏆"] + [""] * (len(shown) - 1))
        st.dataframe(shown.style.format({"WMAPE %": "{:.1f}", "Bias %": "{:+.1f}"}),
                     width="stretch", hide_index=True)
        st.caption("WMAPE = total absolute error ÷ total actual sales. Bias above 0 means the model "
                   "over-forecasts. Models with '52-Wk Seasonality' follow last year's weekly shape.")

with st.expander("📘 How the forecast is built"):
    st.markdown(
        f"""
- **Forecast input:** {forecast_source}.
- **Model:** {champion}, picked automatically for the lowest backtest error.
- **Last year's promotions:** {"repeated in the forecast (baseline × last year's promo lift)" if apply_promo_repeat else "not added on top (they are already in actual sales, or switched off)"}.
- **SA macro factor:** {macro_factor:.3f} (price increase {price_yoy:.1f}% vs food CPI {cpi_inflation:.1f}%{"" if use_macro else ", switched off"}).
- **Revenue:** forecast units × current regular shelf price (R {rsp_now:,.2f}).
- **Range:** {range_label.lower()}.
- **Comparison:** each forecast week is matched to the same week 52 weeks earlier.
        """
    )
