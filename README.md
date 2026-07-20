# Ayurvedic Formulation Classifier

## What it does

- Upload one Raman spectroscopy `.tsv` sample.
- Rebuild the feature vector on the saved universal wavenumber axis.
- Run one KNN model for prediction.
- Predict the full formulation and brand name.
- Show the saved KNN test accuracy as the confidence score.

## Files

- `app.py`: Streamlit user interface.
- `predictor.py`: model loading, preprocessing, and inference.
- `models/`: saved model artifacts required for hosting.
- `requirements.txt`: Python packages for Streamlit Cloud.
