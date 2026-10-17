"""Quick EDA on the Telco churn dataset. Prints key stats used to decide
feature engineering and model framing."""
import pandas as pd

df = pd.read_csv("data/telco_churn.csv")

print("shape:", df.shape)
print("\nchurn rate overall:", (df["Churn"] == "Yes").mean().round(3))

# TotalCharges has some blank strings for brand-new customers (tenure=0); coerce
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nrows with bad TotalCharges:", df["TotalCharges"].isna().sum())
print(df[df["TotalCharges"].isna()][["tenure", "MonthlyCharges", "TotalCharges"]])

print("\nchurn rate by contract type:")
print(df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).round(3))

print("\nchurn rate by internet service:")
print(df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean()).round(3))

print("\nchurn rate by senior citizen:")
print(df.groupby("SeniorCitizen")["Churn"].apply(lambda s: (s == "Yes").mean()).round(3))

print("\ntenure stats by churn:")
print(df.groupby("Churn")["tenure"].describe()[["mean", "50%"]])

print("\nmonthly charges stats by churn:")
print(df.groupby("Churn")["MonthlyCharges"].describe()[["mean", "50%"]])

print("\npayment method churn rate:")
print(df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean()).round(3))
