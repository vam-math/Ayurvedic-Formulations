from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from predictor import AyurvedaClassifier


st.set_page_config(
    page_title="Ayurvedic Formulation Classifier",
    page_icon="🧪",
    layout="centered",
)


@st.cache_resource
def load_classifier() -> AyurvedaClassifier:
    return AyurvedaClassifier()


classifier = load_classifier()
app_dir = Path(__file__).resolve().parent
cover_image_path = app_dir / "ayurvedic cover.png"

if cover_image_path.exists():
    encoded_image = base64.b64encode(cover_image_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(15, 23, 42, 0.18), rgba(15, 23, 42, 0.18)),
                url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        .page-shell {{
            max-width: 860px;
            margin: 2rem auto;
            padding: 1rem 1rem 2rem 1rem;
        }}
        .hero-title {{
            margin: 0;
            padding: 0;
            font-size: 2.4rem;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            letter-spacing: 0.02em;
            text-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
        }}
        label, .stMarkdown, .stMetric label, [data-testid="stFileUploaderDropzoneInstructions"] small {{
            color: #ffffff !important;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderDeleteBtn"],
        [data-testid="stFileUploaderDeleteBtn"] svg,
        [data-testid="stBaseButton-secondary"] svg,
        [data-testid="stBaseButton-secondary"] {{
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }}
        [data-testid="stBaseButton-secondary"] {{
            color: #111111 !important;
            font-weight: 800 !important;
            text-shadow: none !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 18px;
            backdrop-filter: blur(4px);
        }}
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            min-height: 132px;
            backdrop-filter: blur(4px);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: #ffffff !important;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.45);
        }}
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricValue"] > div {{
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            line-height: 1.2 !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.15rem !important;
        }}
        .stApp, .stApp label, .stApp p, .stApp div, .stApp span, .stApp h1 {{
            font-weight: 800 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="page-shell">', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Ayurvedic Formulations Authentication</h1>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["tsv"],
)

if uploaded_file is not None:
    try:
        result = classifier.predict(uploaded_file)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Formulation", result.formulation)
        col2.metric("Predicted Brand", result.brand)

        st.metric("Accuracy", f"{result.confidence_percent:.2f}%")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("</div>", unsafe_allow_html=True)
