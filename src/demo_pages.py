"""Streamlit controls shared by the educational prediction pages."""

from contextlib import closing
import hashlib
from io import BytesIO
import logging
import warnings

from fpdf import FPDF
import pandas as pd
from PIL import Image
import streamlit as st

from .auth import auth
from .database import get_prediction_operations
from .inference import (
    KIDNEY_CATEGORIES, load_image_model, load_tabular, predict_image,
    predict_tabular, read_inputs, validate_inputs,
)

LOGGER = logging.getLogger(__name__)


def page_header(title):
    st.set_page_config(page_title=f"{title} | MediAI", layout="wide")
    if st.sidebar.button("Home"):
        st.switch_page("Home.py")
    if auth.is_authenticated() and st.sidebar.button("Logout"):
        auth.logout_user()
        st.switch_page("Home.py")
    st.title(title)
    st.caption("Educational model demo. Use synthetic records or public sample images only.")
    if auth.is_authenticated():
        st.caption("Model labels and scores from this session are saved to your prediction history.")


def model_or_stop(loader, *args):
    try:
        return loader(*args)
    except Exception as error:
        LOGGER.warning("Model could not load (%s)", type(error).__name__)
        st.error("The trained model is unavailable. Verify the model files and try again.")
        st.stop()


def save_result(disease_type, label, score, scores):
    if not auth.is_authenticated():
        return
    try:
        operations, connection = get_prediction_operations()
        with closing(connection):
            operations.save_prediction(
                username=auth.get_current_user()["username"],
                disease_type=disease_type,
                prediction_result=label,
                confidence=score,
                test_data={"probabilities": scores},
            )
    except Exception as error:
        LOGGER.warning("History write failed (%s)", type(error).__name__)
        st.warning("The model ran successfully. The result could not be saved to your history.")


def assess(disease_type, state_key, input_key, predict):
    st.session_state.pop(state_key, None)
    try:
        label, score, scores = predict()
    except Exception as error:
        LOGGER.warning("Prediction failed for %s (%s)", disease_type, type(error).__name__)
        st.error("The model could not process this input. No result was generated.")
        return
    st.session_state[state_key] = (input_key, label, score, scores)
    save_result(disease_type, label, score, scores)


def result_pdf(title, label, scores):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MediAI | Educational model report", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, title, ln=1)
    pdf.cell(0, 8, f"Model output: {label}", ln=1)
    for name, value in scores.items():
        pdf.cell(0, 7, f"{name}: {value:.1%}", ln=1)
    pdf.ln(4)
    pdf.multi_cell(0, 6, "For educational use only. Scores are model outputs and cannot establish a diagnosis.")
    return pdf.output(dest="S").encode("latin-1")


def show_result(title, disease_type, state_key, input_key):
    result = st.session_state.get(state_key)
    if not result or result[0] != input_key:
        return
    _, label, score, scores = result
    st.subheader("Model output")
    st.metric(label, f"{score:.1%}")
    chart = pd.DataFrame({"Class": list(scores), "Score": list(scores.values())})
    st.bar_chart(chart, x="Class", y="Score", horizontal=True)
    st.caption("Scores describe the model output for this example. They are not clinical probabilities.")
    st.download_button(
        "Download PDF report", data=result_pdf(title, label, scores),
        file_name=f"{disease_type}_report.pdf", mime="application/pdf",
    )


@st.cache_resource
def tabular_checkpoint(model, scaler, fields):
    return load_tabular(model, scaler, fields)


def tabular_page(title, disease_type, action, fields, controls, checkpoints, labels, aliases=None):
    page_header(title)
    model_fields = tuple(aliases.values()) if aliases else fields
    model, scaler = model_or_stop(tabular_checkpoint, *checkpoints, model_fields)
    state_key = f"{disease_type}_results"
    mode = st.radio("Choose input method:", ["Manual Input", "Upload CSV File"])
    if mode == "Manual Input":
        columns = st.columns(2)
        values = []
        for index, (field, control) in enumerate(zip(fields, controls, strict=True)):
            with columns[index % 2]:
                if aliases and aliases[field] in KIDNEY_CATEGORIES:
                    values.append(st.selectbox(control[0], list(KIDNEY_CATEGORIES[aliases[field]])))
                else:
                    caption, low, high, default, step = control
                    values.append(st.number_input(caption, min_value=low, max_value=high, value=default, step=step))
        values = validate_inputs(values, fields, aliases)
        input_key = (mode, *values)
    else:
        st.caption("Required columns include " + ", ".join(fields) + ".")
        if aliases:
            st.caption("The original dataset column abbreviations are also accepted.")
        uploaded = st.file_uploader("Upload example data", type=["csv"])
        st.download_button(
            "Download CSV header template", ",".join(fields) + "\n",
            file_name=f"{disease_type}_template.csv", mime="text/csv",
        )
        if uploaded is None:
            return
        try:
            data = read_inputs(uploaded.getvalue(), fields, aliases)
        except ValueError as error:
            st.session_state.pop(state_key, None)
            st.error(str(error))
            return
        selected = st.selectbox("Select a row", range(len(data)), format_func=lambda index: f"Row {index + 1}")
        values = data.iloc[selected].to_list()
        input_key = (mode, selected, *values)
        action = "Assess Selected Patient" if disease_type == "diabetes" else "Analyze Selected Patient"
    if st.button(action, type="primary"):
        assess(disease_type, state_key, input_key, lambda: predict_tabular(values, model, scaler, labels))
    show_result(title, disease_type, state_key, input_key)


@st.cache_resource
def image_checkpoint(name, size, units):
    return load_image_model(name, size, units)


def image_page(title, disease_type, upload_label, action, checkpoint, size, labels, *, sigmoid=False):
    page_header(title)
    model = model_or_stop(image_checkpoint, checkpoint, size, 1 if sigmoid else len(labels))
    state_key = f"{disease_type}_results"
    uploaded = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    payload = uploaded.getvalue()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload), formats=("JPEG", "PNG")) as opened:
                image = opened.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.session_state.pop(state_key, None)
        st.error("Upload a readable PNG or JPEG sample.")
        return
    st.image(image, width=320)
    input_key = hashlib.sha256(payload).hexdigest()
    if st.button(action, type="primary"):
        assess(disease_type, state_key, input_key, lambda: predict_image(image, model, labels, sigmoid=sigmoid))
    show_result(title, disease_type, state_key, input_key)
