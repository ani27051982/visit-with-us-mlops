# Visit with Us — Streamlit Deployment

## Runtime
- Python 3.11
- scikit-learn 1.5.2
- Streamlit 1.41.1

## Entry Point
app/streamlit_app.py

## Required Artifacts
- published-models/model_a_champion.joblib
- published-models/model_b_champion.joblib
- published-models/champion_manifest.json
- app/config/app_config.json
- app/config/feature_metadata.json

## Local Installation

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Select the GitHub repository.
3. Choose branch `main`.
4. Set entry point to `app/streamlit_app.py`.
5. Select Python 3.11 if available.
6. Deploy and inspect the application logs.
7. Test Model A and Model B predictions.
8. Record the public deployment URL.

## Security

Only deserialize trusted model artifacts.
Verify SHA-256 checksums before loading.
Never expose GitHub tokens or customer-level data.

## Compatibility

Model artifacts were generated using scikit-learn 1.5.2.
Deploy with compatible dependency versions.
