from pathlib import Path
import json
import shutil
import joblib

SOURCE_MODEL = Path("models/final_model.joblib")
VERSION = "v1"
VERSION_DIR = Path("models/versions") / VERSION


def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Final model not found: {SOURCE_MODEL}"
        )

    VERSION_DIR.mkdir(parents=True, exist_ok=True)

    versioned_model = VERSION_DIR / "model.joblib"
    shutil.copy2(SOURCE_MODEL, versioned_model)

    metadata = {
        "model_version": VERSION,
        "model_file": "model.joblib",
        "source_model": str(SOURCE_MODEL),
        "framework": "scikit-learn",
        "purpose": "Customer churn prediction",
        "dataset": "IBM Telco Customer Churn",
        "random_state": 42
    }

    metadata_path = VERSION_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Verify that the copied model can be loaded
    loaded_model = joblib.load(versioned_model)

    print("\n===== MODEL VERSION CREATED =====")
    print(f"Version      : {VERSION}")
    print(f"Model path   : {versioned_model}")
    print(f"Metadata     : {metadata_path}")
    print(f"Model type   : {type(loaded_model).__name__}")
    print("Model loading : SUCCESS")


if __name__ == "__main__":
    main()