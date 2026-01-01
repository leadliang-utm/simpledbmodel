import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("sales.csv")

X = df[["units_sold", "region", "product"]]
y = df["revenue"]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["region", "product"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

joblib.dump(model, "revenue_model.pkl")
print("Model saved to revenue_model.pkl")
