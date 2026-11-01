# Visit with Us — Enterprise MLOps

## Business Objective
Predict customer purchases of the Wellness Tourism Package
using reproducible machine learning pipelines.

## Prediction Models
- Model A: Historical Purchase Prediction
- Model B: Pre-Contact Purchase Prediction

## Technology
Python, scikit-learn, XGBoost, GitHub Actions,
Streamlit and Google Colab.

## MLOps Workflow
1. Dataset registration and validation
2. Data preparation and stratified splitting
3. Model training and hyperparameter optimization
4. Experiment tracking and quality validation
5. Controlled model publishing
6. Streamlit deployment

## Security
Customer-level data, predictions, logs and credentials
must not be committed to public repositories.

## Status
GitHub Actions CI/CD implementation completed.

<!-- DEPLOYMENT_LINKS_START -->

## 🌐 Live Cloud Deployments

### 🚀 Streamlit Community Cloud

**Live Application:** [https://visit-with-us-wellness-ai.streamlit.app/](https://visit-with-us-wellness-ai.streamlit.app/)

Supports both prediction scenarios, interactive customer
forms, probability visualizations and business interpretations.

### 🤗 Hugging Face Spaces

**Live Docker Space:** [https://huggingface.co/spaces/animesh27051982/visit-with-us-wellness-ai](https://huggingface.co/spaces/animesh27051982/visit-with-us-wellness-ai)

Docker-based deployment using verified champion models,
Python 3.11 and pinned dependencies.

<!-- DEPLOYMENT_LINKS_END -->

<!-- CICD_STATUS_START -->

## ⚙️ GitHub Actions — CI/CD Implementation

**Implementation Status: Completed**

The project implements an automated five-job MLOps pipeline:

1. **Register Dataset** — Validates the approved tourism dataset.
2. **Prepare Dataset** — Performs leakage-safe data preparation.
3. **Train & Optimize Models** — Trains and tunes both classifiers.
4. **Validate Model Artifacts** — Applies quality and integrity checks.
5. **Publish Approved Models** — Publishes validated champion artifacts when manually authorized.

### Automation

- Automatic CI execution is configured for relevant source changes on `main`.
- Manual execution remains available through `workflow_dispatch`.
- Champion model publishing requires explicit authorization.
- GitHub Actions uses Python 3.11 and Ubuntu 24.04.
- Automated application validation is configured for Streamlit changes.

### Published Champion Models

- `published-models/model_a_champion.joblib`
- `published-models/model_b_champion.joblib`
- `published-models/champion_manifest.json`

Both champion artifacts are protected by SHA-256 integrity verification.

<!-- CICD_STATUS_END -->
