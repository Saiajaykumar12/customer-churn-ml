from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 20


def tune_model(name, classifier, param_distributions, X, y, cv):
    print(f"\n===== TUNING {name} =====")

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X, y)

    print(f"\nBest ROC-AUC: {search.best_score_:.4f}")
    print("\nBest Parameters:")

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    return search


def main():
    train_path = Path("data/processed/train.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_csv(train_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Random Forest
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_params = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced"],
    }

    rf_search = tune_model(
        "Random Forest",
        rf,
        rf_params,
        X,
        y,
        cv,
    )

    # XGBoost
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )

    xgb_params = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
    }

    xgb_search = tune_model(
        "XGBoost",
        xgb,
        xgb_params,
        X,
        y,
        cv,
    )

    # Save tuning results
    results = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Best ROC-AUC": rf_search.best_score_,
                "Best Parameters": str(rf_search.best_params_),
            },
            {
                "Model": "XGBoost",
                "Best ROC-AUC": xgb_search.best_score_,
                "Best Parameters": str(xgb_search.best_params_),
            },
        ]
    )

    output_path = Path("data/processed/hyperparameter_results.csv")
    results.to_csv(output_path, index=False)

    print("\n===== TUNING COMPLETE =====")
    print(results.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()