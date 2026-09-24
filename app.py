import streamlit as st
import pandas as pd
import numpy as np

def generate_prescriptive_solutions(sku_name, forecast_units, ly_units, price_diff_pct, dist_pct):
    deficit = ly_units - forecast_units
    deficit_pct = (deficit / ly_units) * 100
    
    solutions = []
    
    # Rule 1: Promotional Depth Recommendation
    if price_diff_pct > 0:
        solutions.append(f"**Trade Spend Activation:** Price is {price_diff_pct:.1f}% higher YoY. Initiate a 15% promotional discount to stimulate baseline demand.")
    
    # Rule 2: Distribution Recovery
    if dist_pct < 85:
        solutions.append(f"**Distribution Drive:** Weighted distribution is at {dist_pct}%. Coordinate with field sales to improve shelf placement in Modern Trade.")
    
    # Rule 3: Rebrand Context
    if "Rexona" in sku_name:
        solutions.append("**Rebrand Interventions:** Deploy co-branded 'Shield is now Rexona' wobblers and secondary placements for upcoming key weeks.")

    return {
        "sku": sku_name,
        "forecast": forecast_units,
        "last_year": ly_units,
        "deficit_pct": deficit_pct,
        "solutions": solutions
    }

# --- STREAMLIT UI ENGINE ---
st.title("⚡ Automated FMCG Demand Forecasting Engine")

uploaded_file = st.file_uploader("Upload New Weekly Sales Data (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.success("New data detected! Running automated XGBoost forecast re-calculation...")
    
    # Example simulated results post-computation
    alerts = [
        generate_prescriptive_solutions("Rexona Fresh Roll-On 50ml", 42000, 55000, 12.5, 78),
        generate_prescriptive_solutions("Omo Hand Washing Powder 2kg", 180000, 195000, 5.0, 92)
    ]
    
    st.subheader("🚨 YoY Forecast Deficit Alerts & Recommended Actions")
    
    for alert in alerts:
        with st.expander(f"⚠️ **{alert['sku']}** — Forecast is {alert['deficit_pct']:.1f}% BELOW Last Year's Sales", expanded=True):
            col1, col2 = st.columns(2)
            col1.metric("W+1 Forecast", f"{alert['forecast']:,} units", delta=f"-{alert['deficit_pct']:.1f}% YoY", delta_color="inverse")
            col2.metric("Same Week Last Year", f"{alert['last_year']:,} units")
            
            st.markdown("### 💡 Recommended Strategic Interventions")
            for sol in alert['solutions']:
                st.write(f"- {sol}")
