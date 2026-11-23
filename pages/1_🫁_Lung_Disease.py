from src.demo_pages import image_page

image_page(
    "Lung Disease Analysis", "lung", "Upload a public chest X-ray sample", "Analyze X-Ray",
    "lung_disease_model.h5", 224, ("Normal", "Pneumonia"),
)
