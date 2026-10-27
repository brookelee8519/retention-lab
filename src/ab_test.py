"""Simulates a retention intervention (a discount offer) targeted at
high-risk customers, and runs a proper hypothesis test on whether it worked.

This dataset doesn't have a real experiment in it, so I'm upfront about
simulating it, same move as the feature leakage finding in Movie Match.
I'm not claiming I ran a real A/B test, just showing I know how to set one
up and analyze it correctly: random assignment, a pre-registered effect
size, and the right significance test for a conversion-rate outcome.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from prepare_data import load_clean_data, build_features
import joblib

np.random.seed(42)

df = load_clean_data()
df_feat, feature_cols = build_features(df)
rf = joblib.load("src/churn_model.joblib")

df_feat["churn_risk"] = rf.predict_proba(df_feat[feature_cols])[:, 1]

targeting_threshold = df_feat["churn_risk"].quantile(0.75)
high_risk = df_feat[df_feat["churn_risk"] >= targeting_threshold].copy()
print(f"targeting threshold (75th pct of churn risk): {targeting_threshold:.3f}")
print("high-risk pool size:", len(high_risk), f"({len(high_risk)/len(df_feat):.1%} of customers)")

high_risk["group"] = np.random.choice(["treatment", "control"], size=len(high_risk), p=[0.5, 0.5])
print(high_risk["group"].value_counts())

RELATIVE_CHURN_REDUCTION = 0.25

def simulate_retained(row):
    base_churn_prob = row["churn_risk"]
    if row["group"] == "treatment":
        churn_prob = base_churn_prob * (1 - RELATIVE_CHURN_REDUCTION)
    else:
        churn_prob = base_churn_prob
    churned = np.random.random() < churn_prob
    return not churned  # retained = True if they did NOT churn

high_risk["retained"] = high_risk.apply(simulate_retained, axis=1)

treatment = high_risk[high_risk["group"] == "treatment"]
control = high_risk[high_risk["group"] == "control"]

retention_t = treatment["retained"].mean()
retention_c = control["retained"].mean()

print(f"\nretention rate, treatment (got offer): {retention_t:.3f}")
print(f"retention rate, control (no offer):    {retention_c:.3f}")
print(f"absolute lift: {retention_t - retention_c:+.3f}")

n_t, n_c = len(treatment), len(control)
x_t, x_c = treatment["retained"].sum(), control["retained"].sum()

p_pool = (x_t + x_c) / (n_t + n_c)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
z = (retention_t - retention_c) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"\nz-statistic: {z:.3f}")
print(f"p-value: {p_value:.4f}")
print("statistically significant at alpha=0.05:", p_value < 0.05)

diff = retention_t - retention_c
ci_margin = 1.96 * se
print(f"95% CI on lift: [{diff - ci_margin:.3f}, {diff + ci_margin:.3f}]")

contingency = pd.DataFrame({
    "retained": [x_t, x_c],
    "churned": [n_t - x_t, n_c - x_c],
}, index=["treatment", "control"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nchi-square cross-check, chi2: {chi2:.3f}, p-value: {chi_p:.4f}")

results = {
    "high_risk_pool_size": len(high_risk),
    "n_treatment": n_t,
    "n_control": n_c,
    "retention_treatment": round(retention_t, 4),
    "retention_control": round(retention_c, 4),
    "lift": round(diff, 4),
    "z_stat": round(z, 4),
    "p_value": round(p_value, 4),
    "significant": bool(p_value < 0.05),
}
pd.Series(results).to_csv("src/ab_test_results.csv")
joblib.dump(targeting_threshold, "src/targeting_threshold.joblib")
print("\nsaved results to src/ab_test_results.csv")
print("saved targeting threshold to src/targeting_threshold.joblib")
