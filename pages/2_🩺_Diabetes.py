from src.demo_pages import tabular_page
from src.inference import DIABETES_FIELDS

CONTROLS = (
    ("Pregnancies", 0, 20, 2, 1),
    ("Glucose (mg/dL)", 0, 300, 120, 1),
    ("Diastolic blood pressure (mmHg)", 0, 150, 80, 1),
    ("Skin thickness (mm)", 0, 100, 25, 1),
    ("Insulin (mu U/mL)", 0, 1000, 100, 1),
    ("BMI", 0.0, 100.0, 25.5, 0.1),
    ("Diabetes pedigree function", 0.0, 3.0, 0.5, 0.01),
    ("Age (years)", 1, 120, 30, 1),
)
tabular_page(
    "Diabetes Risk Assessment", "diabetes", "Assess Diabetes Risk", DIABETES_FIELDS,
    CONTROLS, ("diabetes_model_optimized.joblib", "diabetes_scaler.joblib"),
    ("Low Risk", "High Risk"),
)
