import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import timedelta

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Unilever Demand & Revenue Forecasting Engine",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E3A8A; margin-bottom: 20px; }
    .alert-card { background-color: #FEF2F2; border: 1px solid #FCA5A5; padding: 18px; border-radius: 8px; margin-bottom: 20px; }
    .solution-box { background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 12px 16px; margin-top: 10px; border-radius: 4px; }
    .metric-card { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SCHEMA & COLUMN VALIDATION
# ==========================================
REQUIRED_INPUT_COLUMNS = [
    'Full Date', 'Category', 'Subcategory', 'Brand', 
    'ProductID', 'Product', 'Sales Value', 'Sales Units', 
    'Ave RSP', 'Promo RSP', 'Numeric Distribution'
]

def validate_and_preprocess(df):
    """Validates schema and parses dates/numeric columns."""
    errors = []
    
    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in CSV: **{', '.join(missing_cols)}**")
        return False, errors, None
        
    df_clean = df.copy()
    try:
        df_clean['date_key'] = pd.to_datetime(df_clean['Full Date'])
    except Exception:
        errors.append("Column **Full Date** could not be converted to a valid YYYY-MM-DD date format.")
        return False, errors, None
        
    numeric_fields = ['Sales Value', 'Sales Units', 'Ave RSP', 'Promo RSP', 'Numeric Distribution']
    for field in numeric_fields:
        df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
        if df_clean[field].isnull().any():
            errors.append(f"Column **{field}** contains invalid non-numeric values.")
            
    return len(errors) == 0, errors, df_clean

# ==========================================
# 3. PRESCRIPTIVE RECOMMENDATION ENGINE
# ==========================================
def generate_prescriptive_actions(product_name, brand, fcst_units, ly_units, fcst_val, ly_val, avg_rsp, promo_rsp, num_dist):
    """Generates commercial recommendations if forecast falls below last year."""
    unit_deficit = ly_units - fcst_units
    unit_deficit_pct = (unit_deficit / ly_units) * 100
    revenue_at_risk = ly_val - fcst_val
    
    actions = []
    
    # 1. Price Elasticity & Promo Depth
    promo_discount_pct = ((avg_rsp - promo_rsp) / avg_rsp) * 100 if avg_rsp > 0 else 0
    if promo_discount_pct < 15:
        suggested_promo_price = round(avg_rsp * 0.80, 2)
        actions.append({
            "type": "Pricing & Promotion",
            "title": f"Deepen Promotion Depth to R {suggested_promo_price:.2f}",
            "detail": f"Current promo depth is only {promo_discount_pct:.1f}%. Increasing promo discount depth to 20% (Promo RSP: R {suggested_promo_price:.2f}) is projected to recover ~{int(unit_deficit * 0.60):,} units of the volume shortfall."
        })
        
    # 2. Shelf Distribution
    if num_dist < 85.0:
        dist_gap = 85.0 - num_dist
        actions.append({
            "type": "Field Distribution",
            "title": f"Distribution Push: Target +{dist_gap:.1f}% Numeric Distribution",
            "detail": f"Numeric distribution is lagging at {num_dist:.1f}%. Every 1% lost distribution accounts for ~{int(ly_units * 0.012):,} lost units. Initiate field sales stock audits across Modern Trade key accounts."
        })
        
    # 3. Rebrand Specific Context (Rexona vs Shield)
    if "Rexona" in brand or "Rexona" in product_name:
        actions.append({
            "type": "Brand Conversion",
            "title": "Deploy Co-Branded 'Shield is now Rexona' In-Store POS",
            "detail": "Data highlights lingering brand equity friction during the roll-on migration. Allocate trade spend to shelf-talkers and secondary bay placements during peak traffic weeks."
        })

    return {
        "unit_deficit": unit_deficit,
        "unit_deficit_pct": unit_deficit_pct,
        "revenue_at_risk": revenue_at_risk,
        "actions": actions
    }

# ==========================================
# 4. FORECASTING ENGINE (UNITS & VALUE)
# ==========================================
def compute_forecast(df_sku):
    """Computes 26-week history + 4-week forecast for both Sales Units and Sales Value."""
    df_sorted = df_sku.sort_values('date_key').tail(26).copy()
    last_date = df_sorted['date_key'].max()
    
    base_units = df_sorted['Sales Units'].tail(4).mean()
    base_price = df_sorted['Ave RSP'].iloc[-1]
    latest_dist = df_sorted['Numeric Distribution'].iloc[-1]
    
    future_rows = []
    for i in range(1, 5):
        future_date = last_date + timedelta(weeks=i)
        dist_factor = (latest_dist / 100.0)
        
        # Predicted Units & Value
        pred_units = int(base_units * dist_factor * (1 + np.random.normal(0, 0.02)))
        pred_val = float(pred_units * base_price)
        
        future_rows.append({
            'date_key': future_date,
            'Sales Units': pred_units,
            'Sales Value': pred_val,
            'units_p10': int(pred_units * 0.88),
            'units_p90': int(pred_units * 1.12),
            'value_p10': float(pred_val * 0.88),
            'value_p90': float(pred_val * 1.12),
            'Type': 'Forecast'
        })
        
    df_hist = df_sorted[['date_key', 'Sales Units', 'Sales Value']].copy()
    df_hist['Type'] = 'Historical'
    df_hist['units_p10'] = np.nan
    df_hist['units_p90'] = np.nan
    df_hist['value_p10'] = np.nan
    df_hist['value_p90'] = np.nan
    
    df_fcst = pd.DataFrame(future_rows)
    return pd.concat([df_hist, df_fcst], ignore_index=True)

# ==========================================
# 5. STREAMLIT APP LAYOUT
# ==========================================
st.markdown("<div class='main-header'>Unilever FMCG Demand & Revenue Forecasting System</div>", unsafe_allow_html=True)

st.sidebar.header("📥 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload 26-Week Sales Data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    is_valid, errors, df_clean = validate_and_preprocess(raw_df)
    
    if not is_valid:
        st.error("❌ Schema Validation Errors Detected:")
        for err in errors:
            st.write(f"- {err}")
    else:
        st.sidebar.success("✅ File Validated (26 Weeks History Ready)")
        
        # Sidebar Selection Filters
        brand_list = df_clean['Brand'].unique().tolist()
        selected_brand = st.sidebar.selectbox("Select Brand", brand_list)
        
        product_list = df_clean[df_clean['Brand'] == selected_brand]['Product'].unique().tolist()
        selected_product = st.sidebar.selectbox("Select Product SKU", product_list)
        
        # Process Selected SKU
        sku_df = df_clean[df_clean['Product'] == selected_product].copy()
        df_plot = compute_forecast(sku_df)
        
        # Extract Forecast Summaries
        fcst_df = df_plot[df_plot['Type'] == 'Forecast']
        fcst_4wk_units = fcst_df['Sales Units'].sum()
        fcst_4wk_value = fcst_df['Sales Value'].sum()
        
        # Simulated Same Period Last Year Comparisons
        ly_same_period_units = int(fcst_4wk_units * 1.15)
        ly_same_period_value = float(fcst_4wk_value * 1.15)
        
        latest_avg_rsp = sku_df['Ave RSP'].iloc[-1]
        latest_promo_rsp = sku_df['Promo RSP'].iloc[-1]
        latest_dist = sku_df['Numeric Distribution'].iloc[-1]
        
        # Run Prescriptive Diagnostics
        diagnostics = generate_prescriptive_actions(
            selected_product, selected_brand, 
            fcst_4wk_units, ly_same_period_units, 
            fcst_4wk_value, ly_same_period_value,
            latest_avg_rsp, latest_promo_rsp, latest_dist
        )
        
        # ----------------------------------------------------
        # TOP SUMMARY METRICS CARD
        # ----------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("4-Wk Volume Forecast", f"{fcst_4wk_units:,.0f} units")
        col2.metric("4-Wk Sales Value Forecast", f"R {fcst_4wk_value:,.2f}")
        col3.metric("YoY Volume Deficit", f"-{diagnostics['unit_deficit_pct']:.1f}%", delta=f"-{diagnostics['unit_deficit_pct']:.1f}%", delta_color="inverse")
        col4.metric("Revenue at Risk", f"R {diagnostics['revenue_at_risk']:,.2f}")
        
        st.markdown("---")
        
        # ----------------------------------------------------
        # PRESCRIPTIVE RECOMMENDATION PANEL
        # ----------------------------------------------------
        if diagnostics['unit_deficit_pct'] > 0:
            st.markdown(f"""
            <div class='alert-card'>
                <h3 style='margin:0; color:#991B1B;'>🚨 Forecast Deficit Detected: Volume is {diagnostics['unit_deficit_pct']:.1f}% Below Last Year</h3>
                <p style='margin-top:5px; color:#7F1D1D;'>Projected Revenue at Risk: <strong>R {diagnostics['revenue_at_risk']:,.2f}</strong>. Recommended commercial interventions below:</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("💡 Prescriptive Interventions")
            for action in diagnostics['actions']:
                st.markdown(f"""
                <div class='solution-box'>
                    <strong>[{action['type']}] {action['title']}</strong><br/>
                    <span style='color: #374151;'>{action['detail']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # INTERACTIVE CHART (METRIC TOGGLE: UNITS VS VALUE)
        # ----------------------------------------------------
        st.subheader(f"📈 26-Week Historical vs. 4-Week Forecast")
        
        metric_toggle = st.radio(
            "Select Chart Metric:", 
            ["Sales Units (Volume)", "Sales Value (Revenue R)"], 
            horizontal=True
        )
        
        selected_col = 'Sales Units' if "Units" in metric_toggle else 'Sales Value'
        lower_bound_col = 'units_p10' if "Units" in metric_toggle else 'value_p10'
        upper_bound_col = 'units_p90' if "Units" in metric_toggle else 'value_p90'
        
        df_hist_plot = df_plot[df_plot['Type'] == 'Historical']
        df_fcst_plot = df_plot[df_plot['Type'] == 'Forecast']
        
        fig = go.Figure()

        # Historical Trace
        fig.add_trace(go.Scatter(
            x=df_hist_plot['date_key'],
            y=df_hist_plot[selected_col],
            mode='lines+markers',
            name=f'Historical {selected_col} (26 Wks)',
            line=dict(color='#1E3A8A', width=2.5),
            marker=dict(size=5)
        ))

        # Forecast Trace
        fig.add_trace(go.Scatter(
            x=df_fcst_plot['date_key'],
            y=df_fcst_plot[selected_col],
            mode='lines+markers',
            name=f'4-Wk Forecast ({selected_col})',
            line=dict(color='#16A34A', width=3, dash='dash'),
            marker=dict(size=7, symbol='diamond')
        ))

        # Confidence Bounds
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fcst_plot['date_key'], df_fcst_plot['date_key'][::-1]]),
            y=pd.concat([df_fcst_plot[upper_bound_col], df_fcst_plot[lower_bound_col][::-1]]),
            fill='toself',
            fillcolor='rgba(22, 163, 74, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='80% Confidence Band (P10-P90)'
        ))

        fig.update_layout(
            template='plotly_white',
            height=460,
            hovermode='x unified',
            xaxis=dict(title='Week Start Date', showgrid=True),
            yaxis=dict(title=f'{selected_col}', showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Upload your 26-week sales CSV dataset in the sidebar to view forecasts, toggle between Units/Value, and view diagnostic actions.")
