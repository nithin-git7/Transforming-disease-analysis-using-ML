"""Input contracts and inference for the four bundled model checkpoints."""

import csv
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"
DIABETES_FIELDS = (
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
)
KIDNEY_FIELDS = {
    "age": "age", "blood_pressure": "bp", "specific_gravity": "sg",
    "albumin": "al", "sugar": "su", "red_blood_cells": "rbc",
    "pus_cell": "pc", "pus_cell_clumps": "pcc", "bacteria": "ba",
    "blood_glucose_random": "bgr", "blood_urea": "bu", "serum_creatinine": "sc",
    "sodium": "sod", "potassium": "pot", "hemoglobin": "hemo",
    "packed_cell_volume": "pcv", "white_blood_cell_count": "wc",
    "red_blood_cell_count": "rc", "hypertension": "htn", "diabetes_mellitus": "dm",
    "coronary_artery_disease": "cad", "appetite": "appet", "pedal_edema": "pe",
    "anemia": "ane",
}
KIDNEY_CATEGORIES = {
    "rbc": {"normal": 0, "abnormal": 1}, "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1}, "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1}, "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1}, "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1}, "ane": {"no": 0, "yes": 1},
}


def encode_kidney_category(value, field):
    text = str(value).strip().lower()
    return KIDNEY_CATEGORIES[field].get(text, text)


def read_inputs(payload, fields, aliases=None):
    """Load CSV rows by name and reject ambiguous headers and incomplete inputs."""
    try:
        reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
        headers = next(reader, [])
        return parse_rows(reader, headers, fields, aliases)
    except (csv.Error, UnicodeError):
        raise ValueError("Upload a well-formed CSV file encoded as UTF-8.") from None


def parse_rows(reader, headers, fields, aliases):
    if not headers:
        raise ValueError("The CSV needs a header and at least one data row.")
    normalized = [name.strip().lower() for name in headers]
    if len(set(normalized)) != len(normalized):
        raise ValueError("The CSV contains duplicate column names.")
    by_name = {name: index for index, name in enumerate(normalized)}
    indices = []
    for field in fields:
        choices = {field.lower(), (aliases or {}).get(field, field).lower()}
        matches = [by_name[name] for name in choices if name in by_name]
        if len(matches) != 1:
            raise ValueError(f"Provide exactly one column for {field}.")
        indices.append(matches[0])

    rows = []
    for row in reader:
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(headers):
            raise ValueError(f"CSV row {reader.line_num} has an incorrect number of columns.")
        rows.append([row[index] for index in indices])
    if not rows:
        raise ValueError("The CSV has no data rows.")
    rows = [validate_inputs(row, fields, aliases) for row in rows]
    return pd.DataFrame(rows, columns=fields)


def validate_inputs(values, fields, aliases=None):
    if len(values) != len(fields):
        raise ValueError(f"Expected {len(fields)} input values.")
    numbers = []
    for field, value in zip(fields, values, strict=True):
        model_field = (aliases or {}).get(field)
        if model_field in KIDNEY_CATEGORIES:
            value = encode_kidney_category(value, model_field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} needs a valid value.") from None
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{field} needs a finite non-negative value.")
        if model_field in KIDNEY_CATEGORIES and number not in (0, 1):
            raise ValueError(f"{field} needs a named category or a 0/1 value.")
        numbers.append(number)
    return numbers


def load_tabular(model_name, scaler_name, fields):
    model = joblib.load(MODEL_DIR / model_name)
    scaler = joblib.load(MODEL_DIR / scaler_name)
    if tuple(scaler.feature_names_in_) != tuple(fields) or model.n_features_in_ != len(fields):
        raise ValueError("The checkpoint and feature schema do not match.")
    if tuple(model.classes_) != (0, 1):
        raise ValueError("The checkpoint has unexpected class labels.")
    return model, scaler


def predict_tabular(values, model, scaler, labels):
    features = pd.DataFrame([values], columns=scaler.feature_names_in_)
    scaled = scaler.transform(features)
    scores = np.asarray(model.predict_proba(scaled), dtype=float)
    return decode_scores(scores, labels)


def load_image_model(name, size, output_units):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_DIR / name, compile=False)
    if model.input_shape != (None, size, size, 3) or model.output_shape != (None, output_units):
        raise ValueError("The checkpoint has an unexpected input or output shape.")
    return model


def predict_image(image, model, labels, *, sigmoid=False):
    _, height, width, _ = model.input_shape
    resized = image.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    scores = np.asarray(model(pixels[np.newaxis, ...], training=False), dtype=float)
    if sigmoid:
        if scores.shape != (1, 1):
            raise ValueError("The image model returned an unexpected output shape.")
        score = scores[0, 0]
        scores = np.array([[1 - score, score]])
    return decode_scores(scores, labels)


def decode_scores(scores, labels):
    if (
        scores.shape != (1, len(labels))
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or not np.isclose(scores.sum(), 1, atol=1e-5)
    ):
        raise ValueError("The model did not return valid class scores.")
    scores = scores[0]
    index = int(np.argmax(scores))
    return labels[index], float(scores[index]), dict(zip(labels, map(float, scores), strict=True))
