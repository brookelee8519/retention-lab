"""Shared data loading + cleaning + feature prep used by both the model
training script and the Streamlit app."""
import pandas as pd


def load_clean_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def build_features(df):
    """Turns the raw columns into a model-ready feature matrix."""
    df = df.copy()

    binary_map = {"Yes": 1, "No": 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(binary_map)

    df["gender"] = (df["gender"] == "Male").astype(int)

    cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    drop_cols = ["customerID"]
    if "Churn" in df.columns:
        feature_cols = [c for c in df.columns if c not in drop_cols + ["Churn"]]
    else:
        feature_cols = [c for c in df.columns if c not in drop_cols]

    return df, feature_cols
