# Ayurvedic Formulation Classifier

This folder contains a standalone Streamlit app for free hosting on Streamlit Community Cloud.

## What it does

- Upload one Raman spectroscopy `.tsv` sample.
- Rebuild the feature vector on the saved universal wavenumber axis.
- Run a 5-model ensemble using KNN, Random Forest, MLP, SVM, and XGBoost.
- Predict the full formulation and brand name.
- Show one ensemble confidence score based on model agreement.

## Files

- `app.py`: Streamlit user interface.
- `predictor.py`: model loading, preprocessing, and inference.
- `models/`: saved model artifacts required for hosting.
- `requirements.txt`: Python packages for Streamlit Cloud.
