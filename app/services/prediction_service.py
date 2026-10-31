"""Secure, reusable prediction services for Visit with Us."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("visit_with_us.prediction")


class PredictionServiceError(Exception):
    """Base exception for prediction service failures."""


class ModelIntegrityError(PredictionServiceError):
    """Raised when a model artifact fails integrity checks."""


class InputValidationError(PredictionServiceError):
    """Raised when customer inputs are invalid."""


class ModelInferenceError(PredictionServiceError):
    """Raised when model inference fails."""


def sha256_file(path):
    """Calculate SHA-256 without loading the entire file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    """Load and validate a JSON configuration object."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required configuration file missing: {path.name}"
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionServiceError(
            f"Unable to load configuration: {path.name}"
        ) from exc

    if not isinstance(data, dict):
        raise PredictionServiceError(
            f"Configuration must be a JSON object: {path.name}"
        )

    return data


def validate_model_manifest(manifest):
    """Validate the expected published champion manifest."""

    champions = manifest.get("champions")

    if not isinstance(champions, dict):
        raise ModelIntegrityError(
            "Champion manifest is missing model definitions."
        )

    if set(champions) != {"model_a", "model_b"}:
        raise ModelIntegrityError(
            "Expected Model A and Model B in manifest."
        )

    for track, metadata in champions.items():

        required = {
            "algorithm",
            "feature_names",
            "sha256"
        }

        missing = required - set(metadata)

        if missing:
            raise ModelIntegrityError(
                f"{track}: missing metadata fields: "
                f"{sorted(missing)}"
            )

        features = metadata["feature_names"]

        if (
            not isinstance(features, list)
            or not features
            or len(features) != len(set(features))
            or not all(
                isinstance(name, str) and name.strip()
                for name in features
            )
        ):
            raise ModelIntegrityError(
                f"{track}: invalid feature schema."
            )

        checksum = metadata["sha256"]

        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(
                char not in "0123456789abcdef"
                for char in checksum.lower()
            )
        ):
            raise ModelIntegrityError(
                f"{track}: invalid SHA-256 metadata."
            )

    return champions


def load_champion_models(
    repository_root,
    manifest_path=None,
    trusted_source=False
):
    """Load verified champion pipelines from trusted artifacts.

    WARNING:
        joblib deserialization can execute arbitrary code.
        Only use artifacts from a trusted publishing workflow.
        A checksum does not independently prove provenance.
    """

    if not trusted_source:
        raise ModelIntegrityError(
            "Model loading requires a trusted artifact source."
        )

    repository_root = Path(repository_root).resolve()

    models_dir = (
        repository_root / "published-models"
    ).resolve()

    if manifest_path is None:
        manifest_path = (
            models_dir / "champion_manifest.json"
        )

    manifest_path = Path(manifest_path).resolve()

    if manifest_path.parent != models_dir:
        raise ModelIntegrityError(
            "Manifest must be located in published-models."
        )

    manifest = load_json(manifest_path)
    champions = validate_model_manifest(manifest)

    loaded_models = {}

    for track in ("model_a", "model_b"):

        metadata = champions[track]

        model_path = (
            models_dir / f"{track}_champion.joblib"
        ).resolve()

        if model_path.parent != models_dir:
            raise ModelIntegrityError(
                f"{track}: invalid model artifact location."
            )

        if not model_path.is_file():
            raise ModelIntegrityError(
                f"{track}: published model file is missing."
            )

        actual_hash = sha256_file(model_path)

        if actual_hash.lower() != metadata["sha256"].lower():
            raise ModelIntegrityError(
                f"{track}: model checksum verification failed."
            )

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            LOGGER.exception(
                "Trusted model loading failed: %s", track
            )
            raise ModelIntegrityError(
                f"{track}: unable to load the published model."
            ) from exc

        if not callable(getattr(model, "predict_proba", None)):
            raise ModelIntegrityError(
                f"{track}: model lacks probability prediction."
            )

        if not hasattr(model, "classes_"):
            raise ModelIntegrityError(
                f"{track}: fitted class metadata is missing."
            )

        if 1 not in list(model.classes_):
            raise ModelIntegrityError(
                f"{track}: positive class 1 is unavailable."
            )

        # Verify fitted pipeline feature names where exposed.
        if hasattr(model, "feature_names_in_"):
            fitted_features = list(model.feature_names_in_)

            if fitted_features != metadata["feature_names"]:
                raise ModelIntegrityError(
                    f"{track}: fitted feature schema mismatch."
                )

        loaded_models[track] = model

        LOGGER.info(
            "Verified champion loaded successfully: %s",
            track
        )

    return loaded_models, manifest


def validate_customer_input(
    customer_data,
    feature_metadata,
    expected_features
):
    """Validate and normalize one customer input record."""

    if not isinstance(customer_data, dict):
        raise InputValidationError(
            "Customer input must be a dictionary."
        )

    if not isinstance(feature_metadata, dict):
        raise InputValidationError(
            "Feature metadata is invalid."
        )

    expected = list(expected_features)

    missing = [
        name for name in expected
        if name not in customer_data
    ]

    unexpected = [
        name for name in customer_data
        if name not in expected
    ]

    if missing:
        raise InputValidationError(
            "Missing required customer fields: "
            + ", ".join(missing)
        )

    if unexpected:
        raise InputValidationError(
            "Unexpected customer fields: "
            + ", ".join(unexpected)
        )

    normalized = {}

    for feature in expected:

        if feature not in feature_metadata:
            raise InputValidationError(
                f"Feature metadata missing: {feature}"
            )

        specification = feature_metadata[feature]
        value = customer_data[feature]
        feature_type = specification.get("type")

        if feature_type == "numeric":

            if value is None or value == "":
                normalized[feature] = np.nan
                continue

            if isinstance(value, (bool, np.bool_)):
                raise InputValidationError(
                    f"{feature} must be numeric."
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    f"{feature} must contain a valid number."
                ) from exc

            if not np.isfinite(numeric_value):
                raise InputValidationError(
                    f"{feature} must be finite."
                )

            normalized[feature] = numeric_value

        elif feature_type == "categorical":

            if value is None or pd.isna(value):
                normalized[feature] = np.nan
                continue

            category = str(value).strip()

            if not category:
                normalized[feature] = np.nan
                continue

            allowed = [
                str(item)
                for item in specification.get("categories", [])
            ]

            if allowed and category not in allowed:
                raise InputValidationError(
                    f"{feature} contains an unsupported category."
                )

            normalized[feature] = category

        else:
            raise InputValidationError(
                f"Unsupported feature type: {feature}"
            )

    return pd.DataFrame(
        [normalized],
        columns=expected
    )


def predict_customer(
    model,
    customer_data,
    feature_metadata,
    expected_features,
    threshold=0.50
):
    """Generate one validated purchase prediction."""

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "Classification threshold must be numeric."
        ) from exc

    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise InputValidationError(
            "Classification threshold must be between 0 and 1."
        )

    features = validate_customer_input(
        customer_data,
        feature_metadata,
        expected_features
    )

    try:
        probabilities = np.asarray(
            model.predict_proba(features),
            dtype=float
        )

        classes = list(model.classes_)

        if 1 not in classes:
            raise ModelInferenceError(
                "Model does not support the positive class."
            )

        positive_index = classes.index(1)

        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != 1
            or probabilities.shape[1] != len(classes)
        ):
            raise ModelInferenceError(
                "Unexpected prediction probability shape."
            )

        probability = float(
            probabilities[0, positive_index]
        )

        if not np.isfinite(probability):
            raise ModelInferenceError(
                "Model returned a non-finite probability."
            )

        if not 0 <= probability <= 1:
            raise ModelInferenceError(
                "Model returned a probability outside [0, 1]."
            )

    except ModelInferenceError:
        raise

    except Exception as exc:
        LOGGER.exception("Customer prediction failed.")
        raise ModelInferenceError(
            "Prediction could not be completed. "
            "Please verify the input and model configuration."
        ) from exc

    predicted_class = int(
        probability >= threshold
    )

    return {
        "purchase_probability": probability,
        "non_purchase_probability": 1.0 - probability,
        "predicted_class": predicted_class,
        "predicted_label": (
            "Likely Purchaser"
            if predicted_class == 1
            else "Unlikely Purchaser"
        ),
        "decision_threshold": threshold
    }
