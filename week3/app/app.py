import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px # For interactive plots

# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load model ===
# Load model from parent directory
try:
    model = joblib.load('./forecasting_ev_model.pkl')
except FileNotFoundError:
    st.error("Error: 'forecasting_ev_model.pkl' not found. Please ensure the model file is in the parent directory.")
    st.stop()

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: aqua;
            color: #000000;
        }
        .stApp {
            background: linear-gradient(to right, #c2d3f2, #7f848a);
        }
        .stMarkdown div {
            text-align: center; /* Center-align for titles */
        }
    </style>
""", unsafe_allow_html=True)

# Stylized title using markdown + HTML
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: red; margin-top: 20px;'>
        EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

# Welcome subtitle
st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: black;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

# Image
# Load image from parent directory
try:
    st.image("ev-car-factory.jpg", use_container_width=True)
except FileNotFoundError:
    st.warning("Image 'ev-car-factory.jpg' not found. Displaying without image.")


# Instruction line
st.markdown("""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: black;'>
        Select a county and see the forecasted EV adoption trend for the next few years.
    </div>
""", unsafe_allow_html=True)


# === Load data (must contain historical values, features, etc.) ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Error: 'preprocessed_ev_data .csv' not found. Please ensure the data file is in the parent directory.")
        st.stop()

df = load_data()

# === County dropdown ===
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecast Horizon Slider ===
forecast_years = st.slider("Select Forecast Horizon (Years)", 1, 5, 3)
forecast_horizon = forecast_years * 12 # Convert years to months

# === Forecasting Logic ===
def generate_forecast(df_county, county_code_val, model, horizon):
    historical_ev = list(df_county['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev_series = list(df_county['Electric Vehicle (EV) Total'].cumsum().values)
    months_since_start = df_county['months_since_start'].max()
    latest_date = df_county['Date'].max()

    future_rows = []
    
    # Ensure we have enough history for lags, pad if necessary (e.g., for very short historical data)
    # This is a simplification; a robust solution might involve different feature engineering for early periods
    while len(historical_ev) < 3:
        historical_ev.insert(0, historical_ev[0] if historical_ev else 0) # Pad with first available or 0

    for i in range(1, horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        
        # Ensure sufficient history for lags before accessing
        lag1 = historical_ev[-1] if len(historical_ev) >= 1 else 0
        lag2 = historical_ev[-2] if len(historical_ev) >= 2 else 0
        lag3 = historical_ev[-3] if len(historical_ev) >= 3 else 0
        
        roll_mean = np.mean([lag1, lag2, lag3]) if len(historical_ev) >= 3 else lag1 # Use lag1 if not enough history
        
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        
        recent_cumulative = cumulative_ev_series[-6:] if len(cumulative_ev_series) >= 6 else cumulative_ev_series
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) >= 2 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code_val,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        # Predict and append to future_rows
        pred = model.predict(pd.DataFrame([new_row]))[0]
        # Ensure predictions are non-negative
        pred = max(0, round(pred))
        future_rows.append({"Date": forecast_date, "Predicted EV Total": pred})

        # Update historical_ev and cumulative_ev_series for next iteration's lags and slope
        historical_ev.append(pred)
        if len(historical_ev) > 6: # Keep only last 6 for rolling calculations
            historical_ev.pop(0)

        cumulative_ev_series.append(cumulative_ev_series[-1] + pred)
        # We don't pop from cumulative_ev_series to keep full history for accurate 'recent_cumulative'
        # but the `recent_cumulative` slice handles the windowing.

    return pd.DataFrame(future_rows)

# Generate forecast for the selected county
forecast_df_single = generate_forecast(county_df, county_code, model, forecast_horizon)

# === Combine Historical + Forecast for Cumulative Plot ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

# Calculate cumulative for forecast_df_single based on last historical cumulative
if not historical_cum.empty:
    last_historical_cumulative = historical_cum['Cumulative EV'].iloc[-1]
    forecast_df_single['Cumulative EV'] = forecast_df_single['Predicted EV Total'].cumsum() + last_historical_cumulative
else:
    forecast_df_single['Cumulative EV'] = forecast_df_single['Predicted EV Total'].cumsum()
    st.warning("No historical data found for this county. Forecast starts from zero cumulative EV.")

forecast_df_single['Source'] = 'Forecast'

combined_single_county = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df_single[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot Cumulative Graph (Plotly for interactivity) ===
st.subheader(f"📈 Cumulative EV Forecast for {county} County")

fig_single = px.line(combined_single_county, x='Date', y='Cumulative EV', color='Source',
                     title=f"Cumulative EV Trend - {county} ({forecast_years} Years Forecast)",
                     labels={'Cumulative EV': 'Cumulative EV Count'},
                     color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}) # Custom colors
fig_single.update_layout(
    plot_bgcolor='black', # Dark background for plot area
    paper_bgcolor='black', # Dark background for the entire figure
    font_color='black',
    title_font_color='white',
    legend_title_font_color='black'
)
fig_single.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.2)')
fig_single.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.2)')
st.plotly_chart(fig_single, use_container_width=True)

# === Compare historical and forecasted cumulative EVs ===
historical_total = historical_cum['Cumulative EV'].iloc[-1] if not historical_cum.empty else 0
forecasted_total = forecast_df_single['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend_emoji = "🚀" if forecast_growth_pct > 0 else "📉"
    st.success(f"Based on the graph, EV adoption in **{county}** is expected to show a **{trend_emoji} {forecast_growth_pct:.2f}%** growth over the next {forecast_years} years.")
else:
    st.warning("Historical EV total is zero, so percentage forecast change can't be computed. Forecast starts from zero.")

# Add download button for single county forecast
csv_single_county = combined_single_county.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast Data for Selected County",
    data=csv_single_county,
    file_name=f'{county}_ev_forecast.csv',
    mime='text/csv',
    help="Download the historical and forecasted EV data for the selected county."
)


# === New: Compare up to 3 counties ===
st.markdown("---")
st.header("🤝 Compare EV Adoption Trends for up to 5 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=5)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        if cty_df.empty:
            st.warning(f"No data found for {cty}. Skipping comparison for this county.")
            continue

        cty_code = cty_df['county_encoded'].iloc[0]

        # Generate forecast for current county in loop
        forecast_df_cty = generate_forecast(cty_df, cty_code, model, forecast_horizon)

        hist_cum_cty = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum_cty['Cumulative EV'] = hist_cum_cty['Electric Vehicle (EV) Total'].cumsum()

        if not hist_cum_cty.empty:
            last_historical_cumulative_cty = hist_cum_cty['Cumulative EV'].iloc[-1]
            forecast_df_cty['Cumulative EV'] = forecast_df_cty['Predicted EV Total'].cumsum() + last_historical_cumulative_cty
        else:
            forecast_df_cty['Cumulative EV'] = forecast_df_cty['Predicted EV Total'].cumsum()
            st.warning(f"No historical data found for {cty}. Forecast starts from zero cumulative EV for comparison.")

        combined_cty = pd.concat([
            hist_cum_cty[['Date', 'Cumulative EV']],
            forecast_df_cty[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    # Combine all counties data for plotting
    if comparison_data:
        comp_df = pd.concat(comparison_data, ignore_index=True)

        # Plotly Plot for Multi-County Comparison
        st.subheader("📊 Comparison of Cumulative EV Adoption Trends")
        fig_multi = px.line(comp_df, x='Date', y='Cumulative EV', color='County',
                            title=f"EV Adoption Trends: Historical + {forecast_years}-Year Forecast",
                            labels={'Cumulative EV': 'Cumulative EV Count'})
        fig_multi.update_layout(
            plot_bgcolor='#1c1c1c',
            paper_bgcolor='#1c1c1c',
            font_color='white',
            title_font_color='white',
            legend_title_font_color='white'
        )
        fig_multi.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,285,0.2)')
        fig_multi.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,285,0.2)')
        st.plotly_chart(fig_multi, use_container_width=True)

        # Display % growth for selected counties ===
        growth_summaries = []
        for cty_comp in multi_counties:
            # Filter the combined comparison dataframe for the current county
            cty_data_for_growth = comp_df[comp_df['County'] == cty_comp].reset_index(drop=True)
            
            # Find the point where forecast starts based on forecast_horizon
            # Ensure there's enough data; if not, historical_total might be incorrect or 0
            if len(cty_data_for_growth) > forecast_horizon:
                historical_total_comp = cty_data_for_growth['Cumulative EV'].iloc[len(cty_data_for_growth) - forecast_horizon - 1]
            else:
                historical_total_comp = 0 # No sufficient historical data for percentage comparison
                
            forecasted_total_comp = cty_data_for_growth['Cumulative EV'].iloc[-1]

            if historical_total_comp > 0:
                growth_pct_comp = ((forecasted_total_comp - historical_total_comp) / historical_total_comp) * 100
                growth_summaries.append(f"**{cty_comp}**: {growth_pct_comp:.2f}%")
            else:
                growth_summaries.append(f"**{cty_comp}**: N/A (no sufficient historical data for % growth)")

        # Join all in one sentence and show with st.success
        if growth_summaries:
            growth_sentence = " | ".join(growth_summaries)
            st.success(f"Forecasted EV adoption growth over next {forecast_years} years: {growth_sentence}")
        
        # Add download button for multi-county comparison
        csv_multi_county = comp_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Comparison Data",
            data=csv_multi_county,
            file_name='multi_county_ev_comparison.csv',
            mime='text/csv',
            help="Download the historical and forecasted EV data for all compared counties."
        )
    else:
        st.info("No data available for selected counties for comparison.")

st.markdown("---")
st.success("Forecast complete! ✨")



st.markdown("""
    <div style='text-align: center; font-size: 14px; padding-top: 10px; color: black;'>
        Disclaimer: This forecast is based on historical data and a trained machine learning model.
        Future outcomes may vary and are subject to unforeseen factors.
    </div>
""", unsafe_allow_html=True)
