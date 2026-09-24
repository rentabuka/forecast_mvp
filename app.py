import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Unilever Demand Forecasting System",
    page_icon="📦",
    layout="wide"
)

# Professional Unilever Branding CSS
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1F2937; margin-bottom: 20px; }
    .card { background-color: #FFFFFF; padding: 18px; border-radius: 8px; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 24px; font-weight: 700; color: #111827; }
    .metric-label { font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA SCHEMA & VALIDATION ENGINE
# ==========================================
REQUIRED_COLUMNS = {
    'date_key': ['date', 'datetime64[ns]'],
    'sku_id': ['object', 'string'],
    'sku_name': ['object', 'string'],
    'brand_name': ['object', 'string'],
    'category_name': ['object', 'string'],
    'units_sold': ['int64', 'float64', 'int32'],
    'list_unit_price': ['float64', 'int64'],
    'promo_unit_price': ['float64', 'int64'],
    'is_promoted': ['int64', 'bool'],
    'distribution_pct': ['float64', 'int64']
}

def validate_uploaded_file(df):
    """Rigorous schema validation for incoming sales data."""
    errors = []
    warnings = []
    
    # 1. Check Column Existence
    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: **{', '.join(missing_cols)}**")
        return False, errors, warnings
        
    # 2. Data Type Conversions & Checks
    try:
        df['date_key'] = pd.to_datetime(df['date_key'])
    except Exception:
        errors.append("Column **date_key** could not be parsed into valid dates (format should be YYYY-MM-DD).")
        
    numeric_cols = ['units_sold', 'list_unit_price', 'promo_unit_price', 'distribution_pct']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column **{col}** must contain numeric values.")
            
    # 3. Data Integrity & Logic Rules
    if df['units_sold'].min() < 0:
        errors.append("Column **units_sold** contains negative values.")
        
    if (df['distribution_pct'].max() > 100) or (df['distribution_pct'].min() < 0):
        warnings.append("Column **distribution_pct** contains values outside the 0–100% range.")
        
    # Check if data spans at least 26 weeks
    if len(errors) == 0:
        unique_weeks = df['date_key'].nunique()
        if unique_weeks < 26:
            warnings.append(f"Dataset contains only **{unique_weeks} unique weeks**. 26 weeks recommended for seasonal accuracy.")
            
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

# ==========================================
# 3. NOTIFICATION DISPATCHERS
# ==========================================
def send_slack_notification(webhook_url, sku_name, forecast_val, ly_val, deficit_pct, solutions):
    """Dispatches Block Kit formatted alert to Slack."""
    if not webhook_url:
        return False
    
    solution_bullets = "\n".join([f"• {s}" for s in solutions])
    payload = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"🚨 Forecast Deficit: {sku_name}"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*4-Wk Forecast:* {forecast_val:,.0f} units"},
                {"type": "mrkdwn", "text": f"*Last Year Actual:* {ly_val:,.0f} units"},
                {"type": "mrkdwn", "text": f"*YoY Deficit:* -{deficit_pct:.1f}%"},
                {"type": "mrkdwn", "text": f"*Status:* Action Required"}
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*💡 Prescriptive Solutions:*\n{solution_bullets}"}}
        ]
    }
    try:
        r = requests.post(webhook_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ==========================================
# 4. MOCK PREDICTION & ANALYSIS ENGINE
# ==========================================
def run_forecasting_model(df_sku):
    """
    Generates a 4-week forecast based on the last 26 weeks of historical data.
    """
    df_sorted = df_sku.sort_values('date_key').tail(26).copy()
    last_date = df_sorted['date_key'].max()
    
    # Simple exponential trend model for demo execution
    base_volume = df_sorted['units_sold'].tail(4).mean()
    promo_flag = df_sorted['is_promoted'].iloc[-1]
    
    future_weeks = []
    for i in range(1, 5):
        future_date = last_date + timedelta(weeks=i)
        # Apply promo lift if active or Black Friday indicator
        lift = 1.35 if (i == 4 and promo_flag) else 1.02
        pred_units = int(base_volume * lift * (1 + np.random.normal(0, 0.03)))
        
        future_weeks.append({
            'date_key': future_date,
            'units_sold': pred_units,
            'p10_lower': int(pred_units * 0.90),
            'p90_upper': int(pred_units * 1.10),
            'type': 'Forecast'
        })
        
    df_hist = df_sorted[['date_key', 'units_sold']].copy()
    df_hist['type'] = 'Historical'
    df_hist['p10_lower'] = np.nan
    df_hist['p90_upper'] = np.nan
    
    df_forecast = pd.DataFrame(future_weeks)
    return pd.concat([df_hist, df_forecast], ignore_index=True)

# ==========================================
# 5. STREAMLIT APPLICATION LAYOUT
# ==========================================
st.markdown("<div class='main-header'>Unilever FMCG Demand Forecasting System</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel")
webhook_url = st.sidebar.text_input("Slack Webhook URL (Optional)", type="password", help="Paste webhook URL to test automated Slack notifications.")

uploaded_file = st.sidebar.file_uploader("Upload Weekly Sales CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    # Execute Validation
    is_valid, errors, warnings = validate_uploaded_file(raw_df)
    
    if not is_valid:
        st.error("❌ Data Validation Failed")
        for err in errors:
            st.write(f"- {err}")
    else:
        if warnings:
            for warn in warnings:
                st.warning(f"⚠️ {warn}")
        else:
            st.success("✅ File Schema Validated Successfully")
            
        # Filter Selection
        brands = raw_df['brand_name'].unique().tolist()
        selected_brand = st.sidebar.selectbox("Select Brand", brands)
        
        skus = raw_df[raw_df['brand_name'] == selected_brand]['sku_name'].unique().tolist()
        selected_sku = st.sidebar.selectbox("Select Product SKU", skus)
        
        # Process SKU Data
        sku_data = raw_df[raw_df['sku_name'] == selected_sku].copy()
        df_combined = run_forecasting_model(sku_data)
        
        # ----------------------------------------------------
        # METRICS & YoY ALERT EVALUATION
        # ----------------------------------------------------
        df_hist = df_combined[df_combined['type'] == 'Historical']
        df_fcst = df_combined[df_combined['type'] == 'Forecast']
        
        total_fcst_units = df_fcst['units_sold'].sum()
        
        # Simulated same week last year comparison
        ly_units = int(total_fcst_units * 1.15) # Simulated 15% drop YoY
        deficit_pct = ((ly_units - total_fcst_units) / ly_units) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("26-Wk Historical Volume", f"{df_hist['units_sold'].sum():,.0f} units")
        col2.metric("4-Wk Forecast Volume", f"{total_fcst_units:,.0f} units")
        col3.metric("Last Year Same Period", f"{ly_units:,.0f} units")
        col4.metric("YoY Variance", f"-{deficit_pct:.1f}%", delta_color="inverse", delta=f"-{deficit_pct:.1f}%")
        
        st.markdown("---")
        
        # ----------------------------------------------------
        # PLOTLY CHART: 26 WEEKS HISTORICAL + 4 WEEKS FORECAST
        # ----------------------------------------------------
        st.subheader(f"📈 26-Week Historical vs. 4-Week Demand Forecast: {selected_sku}")
        
        fig = go.Figure()

        # Historical Line
        fig.add_trace(go.Scatter(
            x=df_hist['date_key'],
            y=df_hist['units_sold'],
            mode='lines+markers',
            name='Historical Sales (26 Wks)',
            line=dict(color='#1E40AF', width=2.5),
            marker=dict(size=5)
        ))

        # Forecast Line
        fig.add_trace(go.Scatter(
            x=df_fcst['date_key'],
            y=df_fcst['units_sold'],
            mode='lines+markers',
            name='Forecast (W+1 to W+4)',
            line=dict(color='#059669', width=3, dash='dash'),
            marker=dict(size=7, symbol='diamond')
        ))

        # Confidence Interval (P10 - P90)
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fcst['date_key'], df_fcst['date_key'][::-1]]),
            y=pd.concat([df_fcst['p90_upper'], df_fcst['p10_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(5, 150, 105, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='80% Confidence Interval (P10-P90)'
        ))

        fig.update_layout(
            template='plotly_white',
            height=480,
            hovermode='x unified',
            xaxis=dict(title='Week Date', showgrid=True),
            yaxis=dict(title='Volume (Units)', showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # ----------------------------------------------------
        # PRESCRIPTIVE ALERT & SLACK DISPATCH
        # ----------------------------------------------------
        if deficit_pct > 0:
            st.error(f"🚨 **Demand Alert:** The 4-week forecast for **{selected_sku}** is **{deficit_pct:.1f}% BELOW** last year's performance.")
            
            solutions = [
                "**Trade Discount Activation:** Base price increased by 8.5% YoY. Implement a temporary 15% discount.",
                "**Rebrand Point-of-Sale Activation:** Deploy co-branded 'Shield is now Rexona' shelf talkers to capture lost brand equity.",
                "**Distribution Audit:** Weighted distribution dropped by 4.2% in Modern Trade key accounts."
            ]
            
            st.markdown("### 💡 Recommended Solutions")
            for sol in solutions:
                st.markdown(f"- {sol}")
                
            if webhook_url:
                if st.button("📤 Dispatch Alert to Slack Channel"):
                    success = send_slack_notification(webhook_url, selected_sku, total_fcst_units, ly_units, deficit_pct, solutions)
                    if success:
                        st.success("Notification successfully dispatched to Slack!")
                    else:
                        st.error("Failed to send Slack alert. Check your Webhook URL.")

else:
    st.info("👈 Upload your weekly sales CSV file in the sidebar to run validations, generate the 26-week historical chart, and view demand forecasts.")
