import pandas as pd


customers = {
    "customer_id": ["C001", "C002", "C003", "C004", "C005"],
    "tenure_months": [2, 24, 5, 48, 12],
    "monthly_charges": [75.50, 45.00, 90.25, 60.00, 82.50],
    "contract": [
        "Month-to-month",
        "One year",
        "Month-to-month",
        "Two year",
        "Month-to-month",
    ],
    "churn": ["Yes", "No", "Yes", "No", "Yes"],
}


customer_df = pd.DataFrame(customers)

print("CUSTOMER DATA")
print(customer_df)

print("\nNUMBER OF CUSTOMERS")
print(len(customer_df))

print("\nCHURN COUNTS")
print(customer_df["churn"].value_counts())

print("\nAVERAGE MONTHLY CHARGE")
print(customer_df["monthly_charges"].mean())

print("\nAVERAGE CHARGE BY CHURN STATUS")
print(customer_df.groupby("churn")["monthly_charges"].mean())

churned_customers = customer_df[customer_df["churn"] == "Yes"]

churn_rate = len(churned_customers) / len(customer_df) * 100

print("\nCHURNED CUSTOMERS")
print(churned_customers)

print("\nCHURN RATE")
print(f"{churn_rate:.2f}%")