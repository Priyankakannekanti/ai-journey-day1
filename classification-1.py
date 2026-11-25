import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data():
    """Loads built-in breast cancer classification dataset as DataFrame."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Target counts:\n{y.value_counts().to_dict()}")
    return X, y


def preprocess_and_split(X, y):
    """Splits data into train/test (80/20) and scales features without leakage."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    """Trains model and prints train/test accuracy plus confusion matrix."""
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n--- {name} ---")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, test_preds))
    print("Classification Report:")
    print(classification_report(y_test, test_preds, digits=4))


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(X, y)

    # Baseline classifier
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    evaluate_classifier("Logistic Regression (Baseline)", log_reg, X_train, X_test, y_train, y_test)

    # Ensemble classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    evaluate_classifier("Random Forest Classifier", rf_clf, X_train, X_test, y_train, y_test)