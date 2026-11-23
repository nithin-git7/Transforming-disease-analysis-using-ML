"""Load the active MediAI checkpoints and verify their inference contracts."""

from src.inference import DIABETES_FIELDS, KIDNEY_FIELDS, load_image_model, load_tabular


def main():
    load_tabular("diabetes_model_optimized.joblib", "diabetes_scaler.joblib", DIABETES_FIELDS)
    print("Diabetes checkpoint and scaler loaded")
    load_tabular("kidney_model.joblib", "kidney_scaler.joblib", tuple(KIDNEY_FIELDS.values()))
    print("Kidney checkpoint and scaler loaded")
    load_image_model("lung_disease_model.h5", 224, 2)
    print("Lung checkpoint loaded")
    load_image_model("breast_cancer_image_model_improved.h5", 150, 1)
    print("Breast checkpoint loaded")


if __name__ == "__main__":
    main()
