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

## Run locally

```bash
cd deploy_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Free hosting without your laptop running

Use Streamlit Community Cloud.

### Steps

1. Create a new GitHub repository.
2. Upload the full `deploy_streamlit_app` folder contents into that repository.
3. Sign in to Streamlit Community Cloud.
4. Choose `New app`.
5. Select your GitHub repository.
6. Set the main file path to `app.py`.
7. Deploy.

After deployment:

- your laptop does not need to stay on
- Streamlit hosts the app for free
- anyone with the app link can open and use it

## Notes

- This app uses the saved best-fold non-CNN models from `best_fold_per_model.csv`.
- Ensemble confidence is based on vote agreement. For example, `5/5 = 100%`, `4/5 = 80%`, `3/5 = 60%`.
- The app expects the same `.tsv` structure used in `Datasets_processed`.
