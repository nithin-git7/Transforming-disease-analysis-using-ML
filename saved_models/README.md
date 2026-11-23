# Model checkpoints

Git LFS tracks the trained checkpoint files in this directory. After cloning, run `git lfs pull` from the project root to materialize them. Load serialized models only from a trusted source.

| Streamlit page | Checkpoint files | Class order |
| --- | --- | --- |
| Lung | `lung_disease_model.h5` | Normal, Pneumonia |
| Diabetes | `diabetes_model_optimized.joblib`, `diabetes_scaler.joblib` | Low Risk, High Risk |
| Kidney | `kidney_model.joblib`, `kidney_scaler.joblib` | Normal, CKD Detected |
| Breast | `breast_cancer_image_model_improved.h5` | Benign, Malignant |

The image pages resize PNG or JPEG inputs to RGB, scale pixels by 255, and use the models' 224×224 (lung) or 150×150 (breast) inputs. The breast checkpoint outputs the malignant score in a single sigmoid unit. The remaining checkpoints and metrics table are archived experiments from the original project. The app loads the six files listed above.

## Kidney model

The kidney classifier and scaler were refit with categorical features encoded explicitly and missing values handled with medians learned from the training split. Run the command below from the project root with a compatible CSV. The training dataset is kept outside the repository.

```sh
python -m scripts.train_kidney path/to/kidney_disease.csv --output-dir saved_models
```

The bundled pair used a stratified split of a 400-row UCI-style dataset with 320 training rows and 80 held-out rows. On that split, the fitted model classified 30 of 30 non-CKD rows and 50 of 50 CKD rows correctly. This single dataset split has no independent validation and provides no evidence of clinical performance.
