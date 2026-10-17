import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prepare_data import load_clean_data, build_features

ROOT = os.path.join(os.path.dirname(__file__), "..")

st.set_page_config(page_title="Retention Lab", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    rf = joblib.load(os.path.join(ROOT, "src/churn_model.joblib"))
    feature_cols = joblib.load(os.path.join(ROOT, "src/feature_cols.joblib"))
    decision_threshold = joblib.load(os.path.join(ROOT, "src/decision_threshold.joblib"))
    targeting_threshold = joblib.load(os.path.join(ROOT, "src/targeting_threshold.joblib"))
    segment_model = joblib.load(os.path.join(ROOT, "src/segment_model.joblib"))
    segment_scaler = joblib.load(os.path.join(ROOT, "src/segment_scaler.joblib"))
    segment_label_map = joblib.load(os.path.join(ROOT, "src/segment_label_map.joblib"))
    return (rf, feature_cols, decision_threshold, targeting_threshold,
            segment_model, segment_scaler, segment_label_map)


@st.cache_data
def load_scored_data(_feature_cols):
    df = load_clean_data(os.path.join(ROOT, "data/telco_churn.csv"))
    df_feat, feature_cols = build_features(df)

    rf, _, _, _, segment_model, segment_scaler, segment_label_map = load_artifacts()
    df_feat["churn_risk"] = rf.predict_proba(df_feat[feature_cols])[:, 1]

    seg_features = df_feat[["tenure", "MonthlyCharges", "TotalCharges"]]
    seg_scaled = segment_scaler.transform(seg_features)
    df_feat["segment_id"] = segment_model.predict(seg_scaled)
    df_feat["segment_label"] = df_feat["segment_id"].map(segment_label_map)

    df_feat["customerID"] = df["customerID"]
    return df_feat


(rf, feature_cols, decision_threshold, targeting_threshold,
 segment_model, segment_scaler, segment_label_map) = load_artifacts()
scored = load_scored_data(tuple(feature_cols))

st.title("📉 Retention Lab")
st.caption(
    "Churn prediction + a simulated retention-offer experiment, built on the "
    "IBM Telco Customer Churn dataset. Pick a customer below to see their "
    "churn risk, which segment they fall into, and whether a retention offer "
    "is worth sending them."
)

tab_lookup, tab_experiment, tab_model = st.tabs(
    ["🔍 Customer Lookup", "🧪 A/B Test Results", "📊 Model Insights"]
)

# ---------------- Customer Lookup ----------------
with tab_lookup:
    col_select, col_random = st.columns([3, 1])
    with col_select:
        customer_id = st.selectbox("Choose a customer", scored["customerID"].tolist())
    with col_random:
        if st.button("🎲 Random customer"):
            customer_id = np.random.choice(scored["customerID"].tolist())
            st.session_state["picked"] = customer_id

    if "picked" in st.session_state:
        customer_id = st.session_state["picked"]

    row = scored[scored["customerID"] == customer_id].iloc[0]
    raw = load_clean_data(os.path.join(ROOT, "data/telco_churn.csv"))
    raw_row = raw[raw["customerID"] == customer_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn risk", f"{row['churn_risk']:.0%}")
    col2.metric("Segment", row["segment_label"])
    col3.metric("Tenure", f"{int(row['tenure'])} months")

    st.subheader("Recommendation")
    is_high_risk = row["churn_risk"] >= targeting_threshold
    if is_high_risk:
        st.success(
            f"**Send the retention offer.** This customer is in the top 25% "
            f"of churn risk (risk score {row['churn_risk']:.0%}, cutoff is "
            f"{targeting_threshold:.0%}) and belongs to the "
            f"**{row['segment_label']}** segment. Based on the simulated "
            f"A/B test, customers like this retained at a meaningfully "
            f"higher rate when offered a discount, see the A/B Test tab."
        )
    else:
        st.info(
            f"**Hold off on the offer.** Churn risk is {row['churn_risk']:.0%}, "
            f"below the {targeting_threshold:.0%} targeting cutoff. This "
            f"customer isn't flagged as high-risk right now, so a discount "
            f"would likely just be a cost with no retention benefit."
        )

    with st.expander("Customer details"):
        details = raw_row[[
            "gender", "SeniorCitizen", "Partner", "Dependents", "Contract",
            "InternetService", "PaymentMethod", "MonthlyCharges", "TotalCharges",
        ]]
        st.table(details)

# ---------------- A/B Test Results ----------------
with tab_experiment:
    st.subheader("Simulated retention offer experiment")
    st.caption(
        "The dataset doesn't include a real experiment, so this is simulated: "
        "high-risk customers were randomly split into treatment (got a "
        "retention offer) and control (didn't), with a 25% relative churn "
        "reduction applied to the treatment group's underlying risk. The "
        "point is the analysis, not the data: random assignment, a "
        "believable effect size, and the right test for a conversion-rate "
        "outcome."
    )

    ab_results = pd.read_csv(os.path.join(ROOT, "src/ab_test_results.csv"), index_col=0).squeeze()

    col1, col2, col3 = st.columns(3)
    col1.metric("Treatment retention", f"{float(ab_results['retention_treatment']):.1%}")
    col2.metric("Control retention", f"{float(ab_results['retention_control']):.1%}")
    col3.metric("Lift", f"{float(ab_results['lift']):+.1%}")

    p_val = float(ab_results["p_value"])
    significant = str(ab_results["significant"]) == "True"
    st.write(
        f"**p-value:** {p_val:.4f}  |  "
        f"**Statistically significant at α=0.05:** {'Yes' if significant else 'No'}  |  "
        f"**Sample size:** {int(float(ab_results['n_treatment']))} treatment, "
        f"{int(float(ab_results['n_control']))} control"
    )

    chart_df = pd.DataFrame({
        "group": ["Treatment (offer)", "Control (no offer)"],
        "retention_rate": [float(ab_results["retention_treatment"]), float(ab_results["retention_control"])],
    })
    st.bar_chart(chart_df.set_index("group"))

# ---------------- Model Insights ----------------
with tab_model:
    st.subheader("What drives churn risk")
    importance = pd.read_csv(os.path.join(ROOT, "src/feature_importance.csv"), index_col=0).squeeze()
    importance = importance.sort_values(ascending=True).tail(10)
    st.bar_chart(importance)

    st.subheader("Customer segments")
    seg_summary = pd.read_csv(os.path.join(ROOT, "src/segment_summary.csv"), index_col=0)
    st.dataframe(seg_summary, use_container_width=True)
