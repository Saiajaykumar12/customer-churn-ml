from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


MODEL_PATH = Path("models/final_model.joblib")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    train_path = Path("data/processed/train.csv")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    df = pd.read_csv(train_path)

    X = df.drop(columns=["Churn"])

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    print("\n===== SHAP ANALYSIS =====")
    print("Model:", type(classifier).__name__)
    print("Features:", X_transformed_df.shape[1])

    # Explicit TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
    )

    shap_values = explainer.shap_values(X_transformed_df)

    output_dir = Path("models/shap")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global feature importance
    print("\nGenerating global feature importance...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        plot_type="bar",
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    global_path = output_dir / "global_feature_importance.png"

    plt.savefig(
        global_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {global_path}")

    # SHAP summary / beeswarm plot
    print("Generating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    summary_path = output_dir / "shap_summary.png"

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {summary_path}")

    print("\n===== SHAP COMPLETE =====")


if __name__ == "__main__":
    main()