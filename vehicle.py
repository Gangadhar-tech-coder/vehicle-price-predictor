import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Drop rows with missing target
df = df.dropna(subset=["price"])

# Drop unused text columns
df = df.drop(columns=["description", "name"])

# Define target and features
X = df.drop(columns=["price"])
y = df["price"]

# Separate numerical and categorical features
num_features = ["year", "mileage", "cylinders", "doors"]
cat_features = [col for col in X.columns if col not in num_features]

# Preprocessing pipelines
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

# Full pipeline with Random Forest Regressor
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = model_pipeline.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance plot
cat_encoder = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
all_feature_names = num_features + list(cat_feature_names)

importances = model_pipeline.named_steps["regressor"].feature_importances_
feature_importance = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

# Plot top 15 important features
plt.figure(figsize=(10, 6))
feature_importance.head(15).plot(kind='barh')
plt.title("Top 15 Important Features for Vehicle Price Prediction")
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Save model to disk
joblib.dump(model_pipeline, "vehicle_price_model.pkl")
print("\nModel saved as vehicle_price_model.pkl")
