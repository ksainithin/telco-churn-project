import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path

import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


customer_df = pd.read_csv("data/telco_churn_clean.csv")

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

feature_columns = numeric_features + categorical_features

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

number_of_stayers = (y_train == 0).sum()
number_of_churners = (y_train == 1).sum()

churn_weight = number_of_stayers / number_of_churners

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
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                scale_pos_weight=churn_weight,
                random_state=42,
                eval_metric="logloss",
            ),
        ),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("XGBOOST MODEL RESULTS")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.2%}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

model_directory = Path("models")
model_directory.mkdir(parents=True, exist_ok=True)

model_path = model_directory / "churn_xgboost_pipeline.joblib"

joblib.dump(model, model_path)

print("\nSAVED MODEL")
print(model_path)


feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance_values = (
    model.named_steps["classifier"]
    .feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance_values,
})

feature_importance_df["Feature"] = (
    feature_importance_df["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

feature_importance_df = feature_importance_df.sort_values(
    "Importance",
    ascending=False,
)

print("\nTOP 10 IMPORTANT FEATURES")
print(feature_importance_df.head(10).to_string(index=False))

top_features = (
    feature_importance_df.head(10)
    .sort_values("Importance")
)

figure, axis = plt.subplots(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    color="#4C72B0",
    ax=axis,
)

axis.set_title("Top 10 Features Used by the XGBoost Model")
axis.set_xlabel("Importance")
axis.set_ylabel("Customer Feature")

figure.tight_layout()

importance_chart_path = (
    Path("reports/figures")
    / "xgboost_feature_importance.png"
)

figure.savefig(importance_chart_path, dpi=150)

print("\nFEATURE IMPORTANCE CHART SAVED")
print(importance_chart_path)

plt.show()