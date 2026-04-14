import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """Split a DataFrame into train and test sets with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics from true and predicted labels."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """Run stratified k-fold cross-validation with LogisticRegression."""
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight="balanced",
    )

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    print(f"Loaded {len(df)} rows")

    numeric_cols = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen", "has_partner",
        "has_dependents"
    ]
    df_numeric = df[numeric_cols + ["churned"]]

    result = split_data(df_numeric)
    if result is not None:
        X_train, X_test, y_train, y_test = result
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)
        print(metrics)

        cv_results = run_cross_validation(X_train, y_train)
        print(f"CV: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")