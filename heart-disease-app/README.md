# Heart Disease Prediction App

Streamlit web app that loads pre-trained models and predicts presence of heart disease.

## Contents
- `app.py` - Streamlit app
- `models/` - pretrained models and scaler (`.pkl`)
- `data/` - optional cleaned dataset for EDA and metrics
- `metadata.json` - expected features & order
- `requirements.txt` - Python dependencies

## Run locally
1. Create virtual env and install:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
