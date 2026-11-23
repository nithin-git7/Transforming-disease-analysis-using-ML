from src.demo_pages import image_page

image_page(
    "Breast Cancer Analysis", "breast_cancer", "Upload a public mammogram sample",
    "Analyze Mammogram", "breast_cancer_image_model_improved.h5", 150,
    ("Benign", "Malignant"), sigmoid=True,
)
