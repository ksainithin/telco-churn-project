import pandas as pd


customer_df = pd.read_csv("data/telco_churn_clean.csv")

customer_df["ChurnValue"] = customer_df["Churn"].map(
    {"No": 0, "Yes": 1}
)

overall_churn_rate = customer_df["ChurnValue"].mean() * 100

contract_churn_rate = (
    customer_df.groupby("Contract")["ChurnValue"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("OVERALL CHURN RATE")
print(f"{overall_churn_rate:.2f}%")

print("\nCHURN RATE BY CONTRACT")
print(contract_churn_rate.round(2))