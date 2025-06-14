import streamlit as st
import pandas as pd
import joblib

# === Load Models ===
so_regressor = joblib.load("strikeout_regression_model.pkl")
so_classifier = joblib.load("strikeout_model_over6.pkl")
bb_regressor = joblib.load("BB_regression_model.pkl")
bb_classifier = joblib.load("BB_model_over1.pkl")

# === App Title ===
st.title("⚾ Strikeout & Walk Prediction Dashboard")

st.markdown("""
This tool uses two sets of XGBoost models:
- **Regression models** for predicting exact values of Strikeouts and Walks.
- **Classification models** for predicting the probability of exceeding a threshold:
  - Strikeouts > 6
  - Walks > 1
""")

# === User Input ===
st.header("📥 Enter Game Metrics")

dr = st.number_input("Days Rest (DR):", min_value=0, max_value=10, value=4)
start_depth = st.number_input("Start Depth (GS-#):", min_value=0.0, max_value=10.0, value=1.0)
team_id = st.number_input("Team ID (0–29):", min_value=0, max_value=29, value=2)
opp_id = st.number_input("Opponent ID (0–29):", min_value=0, max_value=29, value=5)
rolling_so = st.number_input("Rolling Strikeouts (Last 5 Games):", value=7.0)
rolling_bb = st.number_input("Rolling Walks (Last 5 Games):", value=1.5)
rolling_ip = st.number_input("Rolling Innings Pitched (Last 5 Games):", value=5.8)

# === Prediction ===
if st.button("🧠 Predict All Models"):
    input_df = pd.DataFrame([{
        "DR": dr,
        "Start_Depth": start_depth,
        "Team_ID": team_id,
        "Opp_ID": opp_id,
        "Rolling_SO_5": rolling_so,
        "Rolling_BB_5": rolling_bb,
        "Rolling_IP_5": rolling_ip
    }])

    # Strikeout Predictions
    so_pred = so_regressor.predict(input_df)[0]
    so_prob = so_classifier.predict_proba(input_df)[0][1]

    # Walk Predictions
    bb_pred = bb_regressor.predict(input_df)[0]
    bb_prob = bb_classifier.predict_proba(input_df)[0][1]

    # === Output ===
    st.header("📊 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Strikeouts")
        st.metric("Expected SO", f"{so_pred:.2f}")
        st.metric("P(SO > 6)", f"{so_prob:.2%}")

    with col2:
        st.subheader("🎯 Walks (BB)")
        st.metric("Expected BB", f"{bb_pred:.2f}")
        st.metric("P(BB > 1)", f"{bb_prob:.2%}")
