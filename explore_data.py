import pandas as pd


data_path = "data/telco_churn.csv"

customer_df = pd.read_csv(data_path)

print("DATASET SIZE")
print(customer_df.shape)

print("\nFIRST FIVE CUSTOMERS")
print(customer_df.head())

print("\nCOLUMN NAMES")
print(customer_df.columns.tolist())

print("\nDATA TYPES AND MISSING VALUES")
customer_df.info()

print("\nDUPLICATE ROWS")
print(customer_df.duplicated().sum())

print("\nCHURN VALUES")
print(customer_df["Churn"].value_counts())

print("\nBLANK TOTAL CHARGES")
blank_total_charges = (customer_df["TotalCharges"].str.strip() == "").sum()
print(blank_total_charges)

print("\nCUSTOMERS WITH BLANK TOTAL CHARGES")
blank_rows = customer_df[customer_df["TotalCharges"].str.strip() == ""]

print(
    blank_rows[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)