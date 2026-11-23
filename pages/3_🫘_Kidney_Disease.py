from src.demo_pages import tabular_page
from src.inference import KIDNEY_FIELDS

CONTROLS = (
    ("Age (years)", 1, 120, 45, 1),
    ("Blood pressure (mmHg)", 30, 200, 80, 1),
    ("Specific gravity", 1.000, 1.050, 1.020, 0.005),
    ("Albumin", 0, 5, 0, 1), ("Sugar", 0, 5, 0, 1),
    ("Red blood cells in urine",), ("Pus cells",), ("Pus cell clumps",), ("Bacteria",),
    ("Random blood glucose (mg/dL)", 0, 600, 120, 1),
    ("Blood urea (mg/dL)", 0, 500, 25, 1),
    ("Serum creatinine (mg/dL)", 0.0, 100.0, 1.0, 0.1),
    ("Sodium (mEq/L)", 0, 200, 140, 1),
    ("Potassium (mEq/L)", 0.0, 20.0, 4.0, 0.1),
    ("Hemoglobin (g/dL)", 0.0, 25.0, 12.5, 0.1),
    ("Packed cell volume (%)", 0, 75, 40, 1),
    ("White blood cell count (cells/uL)", 0, 50000, 8000, 100),
    ("Red blood cell count (millions/uL)", 0.0, 10.0, 4.5, 0.1),
    ("Hypertension",), ("Diabetes mellitus",), ("Coronary artery disease",),
    ("Appetite",), ("Pedal edema",), ("Anemia",),
)
tabular_page(
    "Kidney Disease Analysis", "kidney", "Analyze Kidney Function", tuple(KIDNEY_FIELDS),
    CONTROLS, ("kidney_model.joblib", "kidney_scaler.joblib"),
    ("Normal", "CKD Detected"), aliases=KIDNEY_FIELDS,
)
