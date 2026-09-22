from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


customer_df = pd.read_csv("data/telco_churn_clean.csv")

customer_df["ChurnValue"] = customer_df["Churn"].map(
    {"No": 0, "Yes": 1}
)

contract_churn = (
    customer_df.groupby("Contract", as_index=False)["ChurnValue"]
    .mean()
)

contract_churn["ChurnRate"] = contract_churn["ChurnValue"] * 100

contract_order = [
    "Month-to-month",
    "One year",
    "Two year",
]

sns.set_theme(style="whitegrid")

figure, axis = plt.subplots(figsize=(9, 6))

sns.barplot(
    data=contract_churn,
    x="Contract",
    y="ChurnRate",
    order=contract_order,
    color="#3478C0",
    ax=axis,
)

axis.set_title("Customer Churn Rate by Contract Type")
axis.set_xlabel("Contract Type")
axis.set_ylabel("Churn Rate (%)")
axis.set_ylim(0, 50)

for container in axis.containers:
    axis.bar_label(container, fmt="%.1f%%", padding=3)

figure.tight_layout()

output_directory = Path("reports/figures")
output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "churn_by_contract.png"

figure.savefig(output_path, dpi=150)

print(f"Chart saved to: {output_path}")

plt.show()

figure_2, axis_2 = plt.subplots(figsize=(8, 6))

sns.boxplot(
    data=customer_df,
    x="Churn",
    y="MonthlyCharges",
    order=["No", "Yes"],
    hue="Churn",
    palette={
        "No": "#55A868",
        "Yes": "#C44E52",
    },
    legend=False,
    ax=axis_2,
)

axis_2.set_title("Monthly Charges by Churn Status")
axis_2.set_xlabel("Customer Churned")
axis_2.set_ylabel("Monthly Charges ($)")

figure_2.tight_layout()

output_path_2 = output_directory / "monthly_charges_by_churn.png"

figure_2.savefig(output_path_2, dpi=150)

print(f"Chart saved to: {output_path_2}")

plt.show()

tenure_bins = [-1, 12, 24, 48, 72]

tenure_labels = [
    "0-12 months",
    "13-24 months",
    "25-48 months",
    "49-72 months",
]

customer_df["TenureGroup"] = pd.cut(
    customer_df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels,
)

tenure_churn = (
    customer_df.groupby(
        "TenureGroup",
        observed=True,
    )["ChurnValue"]
    .mean()
    .reset_index()
)

tenure_churn["ChurnRate"] = tenure_churn["ChurnValue"] * 100

figure_3, axis_3 = plt.subplots(figsize=(9, 6))

sns.barplot(
    data=tenure_churn,
    x="TenureGroup",
    y="ChurnRate",
    color="#8172B2",
    ax=axis_3,
)

axis_3.set_title("Customer Churn Rate by Tenure")
axis_3.set_xlabel("Customer Tenure")
axis_3.set_ylabel("Churn Rate (%)")
axis_3.set_ylim(0, 50)

for container in axis_3.containers:
    axis_3.bar_label(container, fmt="%.1f%%", padding=3)

figure_3.tight_layout()

output_path_3 = output_directory / "churn_by_tenure.png"

figure_3.savefig(output_path_3, dpi=150)

print(f"Chart saved to: {output_path_3}")

plt.show()