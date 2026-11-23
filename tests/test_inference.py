from io import BytesIO
import csv

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src import inference


KIDNEY_ROW = [
    48, 80, 1.02, 0, 0, "normal", "normal", "notpresent", "notpresent",
    120, 26, 1.1, 138, 4.4, 14.5, 42, 7800, 4.7, "no", "no", "no",
    "good", "no", "no",
]


def csv_bytes(headers, row):
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerow(row)
    return output.getvalue().encode("utf-8")


def sample_png():
    output = BytesIO()
    Image.new("RGB", (290, 233), color=(63, 89, 114)).save(output, format="PNG")
    return output.getvalue()


def test_kidney_accepts_shuffled_headers_and_dataset_codes():
    fields = inference.KIDNEY_FIELDS
    expected = inference.validate_inputs(KIDNEY_ROW, fields, fields)
    headers = list(fields.values())[::-1]
    result = inference.read_inputs(csv_bytes(headers, KIDNEY_ROW[::-1]), fields, fields)
    assert result.iloc[0].tolist() == expected


@pytest.mark.parametrize("payload", [b'"unclosed', b"\xff\xfe"])
def test_csv_reports_invalid_syntax_and_encoding(payload):
    with pytest.raises(ValueError, match="well-formed CSV"):
        inference.read_inputs(payload, inference.DIABETES_FIELDS)


@pytest.mark.parametrize("defect", [
    "missing", "duplicate", "ambiguous", "nonfinite", "category", "empty", "width", "negative"
])
def test_kidney_rejects_invalid_csv(defect):
    fields = inference.KIDNEY_FIELDS
    headers, values = list(fields), list(KIDNEY_ROW)
    if defect == "missing":
        headers[1] = "other"
    elif defect == "duplicate":
        headers[1] = " Age "
    elif defect == "ambiguous":
        headers.append("bp")
        values.append(80)
    elif defect == "nonfinite":
        values[10] = "inf"
    elif defect == "category":
        values[5] = 2
    elif defect == "empty":
        values[7] = ""
    elif defect == "width":
        values.pop()
    else:
        values[0] = -1
    with pytest.raises(ValueError):
        inference.read_inputs(csv_bytes(headers, values), fields, fields)


def test_kidney_calls_saved_scaler_and_uses_trained_categories():
    fields = inference.KIDNEY_FIELDS
    model, scaler = inference.load_tabular(
        "kidney_model.joblib", "kidney_scaler.joblib", tuple(fields.values())
    )
    sample = inference.validate_inputs(KIDNEY_ROW, fields, fields)
    assert scaler.mean_[list(fields.values()).index("htn")] > 0
    frame = pd.DataFrame([sample], columns=scaler.feature_names_in_)
    expected = model.predict_proba(scaler.transform(frame))[0]
    actual = inference.predict_tabular(sample, model, scaler, ("Normal", "CKD Detected"))[2]
    assert list(actual.values()) == pytest.approx(expected)
    assert not np.allclose(expected, model.predict_proba([sample])[0])


def test_invalid_model_output_stops_inference():
    for values in ([0.2, 0.2], [np.nan, 0.2], [-0.1, 1.1], [0.2, 0.2, 0.6]):
        with pytest.raises(ValueError):
            inference.decode_scores(np.asarray([values]), ("Normal", "Pneumonia"))


def test_lung_softmax_uses_both_classes():
    class TwoClassModel:
        input_shape = (None, 224, 224, 3)

        def __call__(self, pixels, training):
            assert pixels.shape == (1, 224, 224, 3)
            assert not training
            return [[0.12, 0.88]]

    with Image.open(BytesIO(sample_png())) as image:
        result = inference.predict_image(image, TwoClassModel(), ("Normal", "Pneumonia"))
    assert result == ("Pneumonia", 0.88, {"Normal": 0.12, "Pneumonia": 0.88})
