import streamlit as st
import pandas as pd
import joblib

# Load models
so_regressor = joblib.load("strikeout_regression_model.pkl")
so_classifier = joblib.load("strikeout_model_over6.pkl")
bb_regressor = joblib.load("BB_regression_model.pkl")
bb_classifier = joblib.load("BB_model_over1.pkl")

# Load feature lists
so_features = joblib.load("so_model_features.pkl")
so_cls_features = joblib.load("so_model_cls_features.pkl")
bb_features = joblib.load("BB_model_features.pkl")
bb_cls_features = joblib.load("BB_model_cls_features.pkl")

# Title
st.title("⚾ Strikeout & Walk Prediction Dashboard")
st.markdown("""
**Two prediction types:**
- 🎯 Regression: Predicts exact values.
- ✅ Classification: Predicts probability of exceeding a threshold (SO > 6, BB > 1).
""")

# Inputs
st.header("📥 Enter Game Metrics")
dr = st.number_input("Days Rest (DR):", 0, 10, 4)
start_depth = st.number_input("Start Depth (GS-#):", 0.0, 10.0, 1.0)
team_id = st.number_input("Team ID (0–29):", 0, 29, 2)
opp_id = st.number_input("Opponent ID (0–29):", 0, 29, 5)
rolling_so = st.number_input("Rolling Strikeouts (Last 5 Games):", value=7.0)
rolling_bb = st.number_input("Rolling Walks (Last 5 Games):", value=1.5)
rolling_ip = st.number_input("Rolling Innings Pitched (Last 5 Games):", value=5.8)
pitcher_id = st.text_input("Pitcher ID (e.g. snellbl01):", value="snellbl01")

# Predict
if st.button("🧠 Predict All Models"):
    base_input = {
        "DR": dr,
        "Start_Depth": start_depth,
        "Team_ID": team_id,
        "Opp_ID": opp_id,
        "Rolling_SO_5": rolling_so,
        "Rolling_BB_5": rolling_bb,
        "Rolling_IP_5": rolling_ip,
        f"pitcher_id_{pitcher_id}": 1
    }

    df_input = pd.DataFrame([base_input])

    # Reindex inputs
    so_input = df_input.reindex(columns=so_features, fill_value=0)
    so_cls_input = df_input.reindex(columns=so_cls_features, fill_value=0)
    bb_input = df_input.reindex(columns=bb_features, fill_value=0)
    bb_cls_input = df_input.reindex(columns=bb_cls_features, fill_value=0)

    # Predict
    so_pred = so_regressor.predict(so_input)[0]
    so_prob = so_classifier.predict_proba(so_cls_input)[0][1]
    bb_pred = bb_regressor.predict(bb_input)[0]
    bb_prob = bb_classifier.predict_proba(bb_cls_input)[0][1]

    # Results
    st.header("📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strikeouts")
        st.metric("Expected SO", f"{so_pred:.2f}")
        st.metric("P(SO > 6)", f"{so_prob:.2%}")

    with col2:
        st.subheader("Walks (BB)")
        st.metric("Expected BB", f"{bb_pred:.2f}")
        st.metric("P(BB > 1)", f"{bb_prob:.2%}")
