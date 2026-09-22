import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


customer_df = pd.read_csv("data/telco_churn_clean.csv")

feature_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
]

X = customer_df[feature_columns]

y = customer_df["Churn"].map({
    "No": 0,
    "Yes": 1,
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    "Contract",
]

data_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", data_preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("TRAINING CUSTOMERS")
print(len(X_train))

print("\nTESTING CUSTOMERS")
print(len(X_test))

print("\nMODEL RESULTS")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 score:  {f1:.2%}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))