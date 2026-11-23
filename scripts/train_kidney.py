"""Train and evaluate the educational CKD model from the UCI-style CSV."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.inference import KIDNEY_CATEGORIES, KIDNEY_FIELDS, encode_kidney_category


def train(dataset, output_dir):
    fields = list(KIDNEY_FIELDS.values())
    data = pd.read_csv(dataset)
    labels = data["classification"].str.strip().str.lower().map({"ckd": 1, "notckd": 0})
    if labels.isna().any():
        raise ValueError("The dataset contains an unrecognized target label.")
    features = data.loc[:, fields].copy()
    for field in fields:
        values = features[field]
        if field in KIDNEY_CATEGORIES:
            values = values.map(lambda value: encode_kidney_category(value, field))
        features[field] = pd.to_numeric(values, errors="coerce")
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.transform(test_x)
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.fit_transform(train_x), train_y)
    predictions = model.predict(scaler.transform(test_x))
    metrics = {
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "accuracy": accuracy_score(test_y, predictions),
        "confusion_matrix_labels": ["notckd", "ckd"],
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=[0, 1]).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "kidney_model.joblib", compress=3)
    joblib.dump(scaler, output_dir / "kidney_scaler.joblib", compress=3)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("saved_models"))
    args = parser.parse_args()
    train(args.dataset, args.output_dir)
