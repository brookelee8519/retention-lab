"""Trains and compares two churn models:
  1. Logistic Regression: interpretable baseline
  2. Random Forest: stronger model, tuned toward recall on the churn class

Business framing: missing an actual churner (false negative) costs more than
flagging a loyal customer by mistake (false positive). A missed churner is
lost revenue, a false alarm just means someone gets an email/discount they
didn't need. So recall on the "will churn" class is the metric that matters
more than raw accuracy here.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, recall_score, precision_score
)
import joblib

sys.path.insert(0, "src")
from prepare_data import load_clean_data, build_features

df = load_clean_data()
df_feat, feature_cols = build_features(df)

X = df_feat[feature_cols]
y = df_feat["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("train size:", X_train.shape, "churn rate:", y_train.mean().round(3))
print("test size:", X_test.shape, "churn rate:", y_test.mean().round(3))

# ---- Model 1: Logistic Regression (scaled features, interpretable baseline) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, logreg_pred, target_names=["stay", "churn"]))
print("ROC-AUC:", round(roc_auc_score(y_test, logreg_proba), 3))

# ---- Model 2: Random Forest, class_weight balanced for recall on churn ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (class_weight=balanced) ===")
print(classification_report(y_test, rf_pred, target_names=["stay", "churn"]))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 3))

# Also show what happens with a lowered decision threshold, since the
# business cares more about recall than the default 0.5 cutoff gives us
threshold = 0.35
rf_pred_lowthresh = (rf_proba >= threshold).astype(int)
print(f"\n=== Random Forest @ threshold={threshold} (recall-optimized) ===")
print(classification_report(y_test, rf_pred_lowthresh, target_names=["stay", "churn"]))
print("recall:", round(recall_score(y_test, rf_pred_lowthresh), 3))
print("precision:", round(precision_score(y_test, rf_pred_lowthresh), 3))

# ---- Feature importance ----
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\ntop 10 features driving churn risk:")
print(importances.head(10).round(3))

# ---- Save artifacts for the Streamlit app ----
joblib.dump(rf, "src/churn_model.joblib")
joblib.dump(feature_cols, "src/feature_cols.joblib")
joblib.dump(threshold, "src/decision_threshold.joblib")
importances.to_csv("src/feature_importance.csv")
print("\nsaved model + feature list + importances to src/")
