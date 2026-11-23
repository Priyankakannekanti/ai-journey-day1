import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_data():
    """Loads the California Housing dataset and returns features and target."""
    print("1. Loading dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y

def preprocess_and_split(X, y):
    """Splits data into train/test sets and applies feature scaling."""
    print("2. Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Trains a Random Forest Regressor model."""
    print("3. Training Random Forest model (takes 5-10 seconds)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluates train vs test R2 score, test errors, and feature importances."""
    print("4. Evaluating model performance...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    print("\n--- Model Evaluation Results ---")
    print(f"Train R2 Score: {train_r2:.4f} ({train_r2 * 100:.2f}%)")
    print(f"Test R2 Score : {test_r2:.4f} ({test_r2 * 100:.2f}%)")
    print(f"Mean Absolute Error (MAE): ${mae * 100000:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse * 100000:,.2f}")

    # Feature Importance (sorted highest to lowest)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False)

    print("\n--- Feature Importance (Most Impactful First) ---")
    for feature, score in top_features.items():
        print(f"{feature:12s}: {score * 100:.2f}%")

    return train_r2, test_r2

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test, X.columns)

    # Save the trained model and scaler for future predictions
    joblib.dump(model, "house_price_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nModel and scaler saved successfully!")