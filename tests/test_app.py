from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import inference
from test_inference import KIDNEY_ROW, csv_bytes, sample_png


ROOT = Path(__file__).resolve().parents[1]
PAGES = {path.name.split("_")[0] or path.stem: path.relative_to(ROOT).as_posix()
         for path in (ROOT / "pages").glob("*.py")}


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'demo.db').as_posix()}")


def app_at(page=None):
    app = AppTest.from_file(str(ROOT / "Home.py"), default_timeout=80).run()
    if page:
        app.switch_page(PAGES[page]).run()
    assert not app.exception
    return app


def click(app, text):
    next(widget for widget in app.button if text in widget.label).click().run()
    assert not app.exception
    return app


def assert_report(app, title, payloads):
    assert not app.error
    assert app.metric
    assert any(button.proto.label == "Download PDF report" for button in app.get("download_button"))
    assert pdf_contains(title, payloads[-1])


def pdf_contains(title, artifact):
    pdf = PdfReader(BytesIO(artifact))
    page = pdf.pages[0].extract_text()
    return title in page and "Educational model report" in page


@pytest.fixture
def pdfs(monkeypatch):
    artifacts = []
    download = st.download_button

    def collect(*args, **kwargs):
        if kwargs.get("mime") == "application/pdf":
            artifacts.append(kwargs["data"])
        return download(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", collect)
    return artifacts


@pytest.mark.parametrize(("page", "action", "title"), [
    ("1", "Analyze X-Ray", "Lung Disease Analysis"),
    ("2", "Assess Diabetes Risk", "Diabetes Risk Assessment"),
    ("3", "Analyze Kidney Function", "Kidney Disease Analysis"),
    ("4", "Analyze Mammogram", "Breast Cancer Analysis"),
])
def test_real_model_and_pdf_for_each_page(page, action, title, pdfs):
    with patch.object(st, "file_uploader", side_effect=lambda *args, **kwargs: BytesIO(sample_png())):
        app = click(app_at(page), action)
    assert_report(app, title, pdfs)


@pytest.mark.parametrize(("page", "fields", "row", "title"), [
    ("2", inference.DIABETES_FIELDS, [2, 120, 80, 25, 100, 25.5, 0.5, 30], "Diabetes Risk Assessment"),
    ("3", tuple(inference.KIDNEY_FIELDS), KIDNEY_ROW, "Kidney Disease Analysis"),
])
def test_csv_success_and_invalid_upload_clears_previous_result(page, fields, row, title, pdfs):
    app = app_at(page)
    with patch.object(st, "file_uploader", side_effect=lambda *a, **kw: BytesIO(csv_bytes(fields, row))):
        app.radio[0].set_value("Upload CSV File").run()
        click(app, "Selected Patient")
    assert_report(app, title, pdfs)
    invalid = csv_bytes(["wrong_header", *fields[1:]], row)
    with patch.object(st, "file_uploader", side_effect=lambda *a, **kw: BytesIO(invalid)):
        app.run()
    assert app.error and not app.metric and len(pdfs) == 1


def test_bad_image_cannot_generate_output(pdfs):
    with patch.object(st, "file_uploader", side_effect=lambda *a, **kw: BytesIO(b"not an image")):
        app = app_at("1")
    assert app.error and not app.metric and not pdfs


@pytest.mark.parametrize("page", ["1", "2", "3", "4"])
def test_missing_checkpoint_stops_page(page, monkeypatch, tmp_path, pdfs):
    from src.demo_pages import image_checkpoint, tabular_checkpoint

    image_checkpoint.clear()
    tabular_checkpoint.clear()
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    app = app_at(page)
    assert any("model is unavailable" in error.value for error in app.error)
    assert not app.metric and not pdfs
    assert all(button.label == "Home" for button in app.button)


def test_inference_error_clears_previous_output(pdfs):
    from src import demo_pages

    app = click(app_at("3"), "Analyze Kidney Function")
    assert app.metric and pdfs
    with patch.object(demo_pages, "predict_tabular", side_effect=ValueError("bad estimator output")):
        click(app, "Analyze Kidney Function")
    assert app.error and not app.metric and len(pdfs) == 1


def test_register_sign_in_save_export_and_sign_out(pdfs):
    from src.database import get_prediction_operations

    app = app_at("5")
    assert app.warning
    app.switch_page(PAGES["_Register"]).run()
    for widget, value in zip(app.text_input, [
        "audit_student", "Demo Student", "student@example.test", "AuditDemo42", "AuditDemo42"
    ], strict=True):
        widget.set_value(value)
    app.checkbox[0].check()
    with patch.object(st, "switch_page") as navigate:
        click(app, "Create Account")
    navigate.assert_called_once_with(PAGES["_Login"])
    assert not app.error
    app.switch_page(PAGES["_Login"]).run()
    app.text_input[0].set_value("audit_student")
    app.text_input[1].set_value("Incorrect42")
    click(app, "Sign In")
    assert app.error and not app.session_state.filtered_state.get("authenticated", False)
    app.text_input[1].set_value("AuditDemo42")
    with patch.object(st, "switch_page") as navigate:
        click(app, "Sign In")
    navigate.assert_called_once_with("Home.py")
    assert not app.error and app.session_state["authenticated"]
    app.switch_page(PAGES["2"]).run()
    click(app, "Assess Diabetes Risk")
    assert_report(app, "Diabetes Risk Assessment", pdfs)
    operations, connection = get_prediction_operations()
    try:
        rows = operations.get_user_predictions("audit_student")
        assert len(rows) == 1
        assert list(rows[0]["test_data"]) == ["probabilities"]
        assert not operations.get_user_predictions("another_student")
    finally:
        connection.close()
    app.switch_page(PAGES["5"]).run()
    assert not app.error
    click(app, "Export History as CSV")
    assert any(item.proto.label == "💾 Download CSV" for item in app.get("download_button"))
    with patch.object(st, "switch_page"):
        click(app, "Logout")
    assert not app.session_state["authenticated"]
    assert "diabetes_results" not in app.session_state
