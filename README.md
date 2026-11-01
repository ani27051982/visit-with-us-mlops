# Visit with Us — Enterprise MLOps

<!-- GITHUB_ACTIONS_BADGES_START -->

## 🔄 CI/CD Pipeline Status

The following badges display the current GitHub Actions
workflow status for the `main` branch.

[![MLOps Pipeline](https://github.com/ani27051982/visit-with-us-mlops/actions/workflows/pipeline.yml/badge.svg?branch=main)](https://github.com/ani27051982/visit-with-us-mlops/actions/workflows/pipeline.yml)
[![Streamlit Validation](https://github.com/ani27051982/visit-with-us-mlops/actions/workflows/streamlit-validation.yml/badge.svg?branch=main)](https://github.com/ani27051982/visit-with-us-mlops/actions/workflows/streamlit-validation.yml)

### Automated Workflow Verification

| Workflow | Execution | Latest Verified Run |
|---|---|---|
| MLOps Pipeline | Push + Manual | [Successful automatic run](https://github.com/ani27051982/visit-with-us-mlops/actions/runs/41280048719) |
| Streamlit Validation | Push + Manual | [Successful automatic run](https://github.com/ani27051982/visit-with-us-mlops/actions/runs/41281006920) |

**Model publishing:** Manual approval is required
through `publish_models = true`. Automatic push
events perform validation without publishing champions.

<!-- GITHUB_ACTIONS_BADGES_END -->


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

### Streamlit Community Cloud

[🚀 Open Live Streamlit Application](https://visit-with-us-wellness-ai.streamlit.app/)

Interactive customer purchase prediction using
Model A and Model B.

### Hugging Face Spaces

[🤗 Open Hugging Face Docker Space](https://huggingface.co/spaces/animesh27051982/visit-with-us-wellness-ai)

Docker-based deployment using validated champion models.

### GitHub Repository

[📂 View Project Source Code](https://github.com/ani27051982/visit-with-us-mlops)

<!-- DEPLOYMENT_LINKS_END -->

<!-- CICD_STATUS_START -->

## ⚙️ GitHub Actions — Automated CI/CD

**Status: Implemented and Verified**

Both project workflows support automatic execution
when relevant files are pushed to the `main` branch.

### 🤖 MLOps Pipeline

The five-job pipeline includes:

1. Register Dataset
2. Prepare Dataset
3. Train & Optimize Models
4. Validate Model Artifacts
5. Publish Approved Models

**Automatic execution:** First four jobs.

**Champion publishing:** Manual approval required.

**Verified automatic runs:** 1 successful run.

[View successful automatic MLOps run](https://github.com/ani27051982/visit-with-us-mlops/actions/runs/41280048719)

### 🚀 Streamlit Application Validation

The application workflow automatically validates
relevant Streamlit and model integration changes.

**Verified automatic runs:** 3 successful runs.

[View successful Streamlit validation run](https://github.com/ani27051982/visit-with-us-mlops/actions/runs/41281006920)

### 🔐 Engineering Controls

- Automatic `push` triggers on `main`.
- Manual `workflow_dispatch` execution.
- Controlled champion publishing.
- Python 3.11.
- Ubuntu 24.04 runners.
- SHA-256 model integrity verification.
- Scoped GitHub Actions permissions.

<!-- CICD_STATUS_END -->
