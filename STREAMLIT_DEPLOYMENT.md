# Visit with Us — Streamlit Deployment

## Runtime Environment

- Python: 3.11
- NumPy: 1.26.4
- scikit-learn: 1.5.2
- PyArrow: 17.0.0
- Streamlit: 1.41.1

## Application Entry Point

`app/streamlit_app.py`

## Required Champion Artifacts

- `published-models/model_a_champion.joblib`
- `published-models/model_b_champion.joblib`
- `published-models/champion_manifest.json`

## Local Execution

Use a Python 3.11 virtual environment.

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

1. Connect the GitHub repository.
2. Select the main branch.
3. Use `app/streamlit_app.py` as the entry point.
4. Select Python 3.11 where supported.
5. Deploy and inspect application logs.
6. Verify predictions for Model A and Model B.

## Hugging Face Spaces

- SDK: Docker
- Application port: 7860
- Dockerfile: repository root
- Space README: Docker SDK metadata

## Security

- Load only trusted published champion artifacts.
- Verify SHA-256 checksums before deserialization.
- Do not log or persist customer input records.
- Keep credentials in secure secret stores.

## Compatibility

The published models were trained with scikit-learn 1.5.2.
Inference should use the matching runtime dependencies.
