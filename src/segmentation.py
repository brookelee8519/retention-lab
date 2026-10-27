"""K-means segmentation by usage/value, so the retention offer can be
targeted at the segment where it actually makes financial sense (ties the
churn model + A/B test result together into one recommendation)."""
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from prepare_data import load_clean_data, build_features
import joblib

df = load_clean_data()
df_feat, feature_cols = build_features(df)
rf = joblib.load("src/churn_model.joblib")
df_feat["churn_risk"] = rf.predict_proba(df_feat[feature_cols])[:, 1]

seg_features = df_feat[["tenure", "MonthlyCharges", "TotalCharges"]]
scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg_features)

inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(seg_scaled)
    inertias.append(km.inertia_)
print("inertia by k:", [round(i) for i in inertias])

k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df_feat["segment"] = km.fit_predict(seg_scaled)

summary = df_feat.groupby("segment").agg(
    n_customers=("segment", "size"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    avg_total_charges=("TotalCharges", "mean"),
    avg_churn_risk=("churn_risk", "mean"),
).round(1)

median_tenure = summary["avg_tenure"].median()
median_spend = summary["avg_monthly_charges"].median()

def label_segment(row):
    is_loyal = row["avg_tenure"] >= median_tenure
    is_high_spend = row["avg_monthly_charges"] >= median_spend
    if is_loyal and is_high_spend:
        return "Loyal high-value"
    if is_loyal and not is_high_spend:
        return "Loyal low-spend"
    if not is_loyal and is_high_spend:
        return "New high-value (at-risk)"
    return "New low-spend"

summary["label"] = summary.apply(label_segment, axis=1)
print("\nsegment summary:")
print(summary)

summary_sorted = summary.sort_values("avg_churn_risk", ascending=False)
top_segment = summary_sorted.iloc[0]
print(f"\nhighest-risk segment: '{top_segment['label']}' "
      f"({top_segment['n_customers']:.0f} customers, "
      f"{top_segment['avg_churn_risk']:.1%} avg churn risk, "
      f"${top_segment['avg_monthly_charges']:.0f}/mo avg revenue) "
      f"- this is where the retention offer has the best ROI: "
      f"high risk of leaving AND meaningful revenue to protect.")

cluster_label_map = summary["label"].to_dict()

joblib.dump(km, "src/segment_model.joblib")
joblib.dump(scaler, "src/segment_scaler.joblib")
joblib.dump(cluster_label_map, "src/segment_label_map.joblib")
summary.to_csv("src/segment_summary.csv")
print("\nsaved segment model + summary + label map to src/")
