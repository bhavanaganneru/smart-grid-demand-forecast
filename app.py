import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Smart Grid Dashboard",
                   layout="wide")

# ---------- PREMIUM COLORFUL GLASS THEME ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1 {
    color: #00FFFF;
    font-weight: 800;
}
h2, h3 {
    color: #00E5FF;
}
.glass {
    background: rgba(255, 255, 255, 0.10);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 17px;
}
.white-text {
    color: white !important;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ Smart Grid Electricity Forecast Dashboard</h1>", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load("model.pkl")

# ---------- Load Dataset ----------
df = pd.read_csv("data.csv", sep="\t")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['lag_1'] = df['DOM_MW'].shift(1)
df = df.dropna()

# ---------- Sidebar ----------
st.sidebar.header("📅 Select Date")
selected_date = st.sidebar.date_input("Choose Date")
predict_button = st.sidebar.button("🚀 Predict Demand")

if predict_button:

    predictions = []
    hours = list(range(24))
    last_value = df['DOM_MW'].iloc[-1]

    for hour in hours:
        input_datetime = datetime.combine(selected_date, datetime.min.time())
        input_datetime = input_datetime.replace(hour=hour)

        dayofweek = input_datetime.weekday()
        month = input_datetime.month
        year = input_datetime.year

        input_data = pd.DataFrame([[hour, dayofweek, month, year, last_value]],
                                  columns=['hour', 'dayofweek', 'month', 'year', 'lag_1'])

        pred = model.predict(input_data)[0]
        predictions.append(pred)
        last_value = pred

    prediction_df = pd.DataFrame({
        "Hour": hours,
        "Predicted_Demand_MW": predictions
    })

    # ---------- CO2 Calculation ----------
    EMISSION_FACTOR = 0.82
    daily_emission = sum(predictions) * 1000 * EMISSION_FACTOR
    peak_hour = prediction_df.loc[prediction_df['Predicted_Demand_MW'].idxmax()]
    peak_emission = peak_hour['Predicted_Demand_MW'] * 1000 * EMISSION_FACTOR

    # ---------- KPI Glass Cards ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="glass">
            <h3>🔝 Peak Demand</h3>
            <h2>{max(predictions):,.2f} MW</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass">
            <h3>🌍 Daily CO₂ Emission</h3>
            <h2>{daily_emission:,.0f} kg</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="glass">
            <h3>⚠ Peak Hour Emission</h3>
            <h2>{peak_emission:,.0f} kg</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Forecast Chart ----------
    st.markdown("<h2>📈 24-Hour Demand Forecast</h2>", unsafe_allow_html=True)

    fig = px.line(prediction_df,
                  x="Hour",
                  y="Predicted_Demand_MW",
                  markers=True,
                  template="plotly_dark")

    fig.add_scatter(
        x=[peak_hour['Hour']],
        y=[peak_hour['Predicted_Demand_MW']],
        mode="markers",
        marker=dict(size=16, color="red"),
        name="Peak Hour"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- AI Explanation ----------
    st.markdown("<h2>🧠 AI Demand Analysis</h2>", unsafe_allow_html=True)

    explanation = ""

    if selected_date.weekday() >= 5:
        explanation += "• Weekend detected — residential usage likely dominates demand.<br><br>"
    else:
        explanation += "• Weekday detected — industrial & commercial load increases demand.<br><br>"

    if selected_date.month in [12, 1, 2]:
        explanation += "• Winter season — heating systems may increase usage.<br><br>"
    elif selected_date.month in [6, 7, 8]:
        explanation += "• Summer season — air conditioning increases electricity demand.<br><br>"

    if max(predictions) > df['DOM_MW'].quantile(0.9):
        explanation += "• High peak detected — grid stress likely during peak hour.<br><br>"

    explanation += "• Estimated CO₂ impact calculated based on fossil fuel emission factor."

    st.markdown(f"""
    <div class="glass white-text">
        {explanation}
    </div>
    """, unsafe_allow_html=True)

# ---------- Historical Chart ----------
st.markdown("<h2>📊 Recent Historical Demand</h2>", unsafe_allow_html=True)

recent_data = df['DOM_MW'].tail(500).reset_index()

fig_hist = px.line(recent_data,
                   x="Datetime",
                   y="DOM_MW",
                   template="plotly_dark")

st.plotly_chart(fig_hist, use_container_width=True)

# ---------- Model Accuracy ----------
with open("mae.txt", "r") as f:
    mae = float(f.read())

st.markdown(f"""
<div class="glass white-text">
📌 Model Mean Absolute Error (MAE): {mae:,.2f}
</div>
""", unsafe_allow_html=True)
