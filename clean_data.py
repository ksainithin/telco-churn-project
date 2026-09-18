import pandas as pd


input_path = "data/telco_churn.csv"
output_path = "data/telco_churn_clean.csv"

customer_df = pd.read_csv(input_path)

customer_df["TotalCharges"] = pd.to_numeric(
    customer_df["TotalCharges"],
    errors="coerce",
)

print("MISSING BEFORE CLEANING")
print(customer_df["TotalCharges"].isna().sum())

customer_df["TotalCharges"] = customer_df["TotalCharges"].fillna(0)

customer_df.to_csv(output_path, index=False)

print("\nMISSING AFTER CLEANING")
print(customer_df["TotalCharges"].isna().sum())

print("\nTOTAL CHARGES TYPE")
print(customer_df["TotalCharges"].dtype)

print("\nCLEANED DATA SAVED")
print(output_path)