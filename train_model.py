# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop rows with missing target (price)
df = df.dropna(subset=["price"])

# Drop text-heavy columns not used directly
df = df.drop(columns=["name", "description"])

# Separate target and features
X = df.drop(columns=["price"])
y = df["price"]

# Feature types
num_features = ["year", "mileage", "cylinders", "doors"]
cat_features = [col for col in X.columns if col not in num_features]

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Full pipeline with model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, "vehicle_price_model.pkl")
print("✅ Model saved as vehicle_price_model.pkl")
