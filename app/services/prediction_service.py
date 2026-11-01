"""Verified champion loading and customer prediction services."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("visit_with_us.prediction")


class PredictionServiceError(Exception):
    """Base prediction-service exception."""


class ModelIntegrityError(PredictionServiceError):
    """Raised when a champion artifact cannot be verified."""


class InputValidationError(PredictionServiceError):
    """Raised for invalid customer input."""


class ModelInferenceError(PredictionServiceError):
    """Raised when prediction execution fails."""


def sha256_file(path):
    """Calculate a file's SHA-256 checksum in chunks."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    """Load a JSON configuration object."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required configuration missing: {path.name}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionServiceError(
            f"Unable to read configuration: {path.name}"
        ) from exc

    if not isinstance(data, dict):
        raise PredictionServiceError(
            f"Invalid configuration structure: {path.name}"
        )

    return data


def validate_model_manifest(manifest):
    """Validate the published champion manifest structure."""

    if not isinstance(manifest, dict):
        raise ModelIntegrityError(
            "Champion manifest must be a JSON object."
        )

    champions = manifest.get("champions")

    if not isinstance(champions, dict):
        raise ModelIntegrityError(
            "Champion model definitions are missing."
        )

    expected_tracks = {"model_a", "model_b"}

    if set(champions) != expected_tracks:
        raise ModelIntegrityError(
            "Manifest must contain Model A and Model B."
        )

    for track, metadata in champions.items():

        if not isinstance(metadata, dict):
            raise ModelIntegrityError(
                f"{track}: invalid model metadata."
            )

        required = {
            "algorithm",
            "feature_names",
            "sha256"
        }

        missing = required - set(metadata)

        if missing:
            raise ModelIntegrityError(
                f"{track}: missing metadata {sorted(missing)}"
            )

        features = metadata["feature_names"]

        if (
            not isinstance(features, list)
            or not features
            or not all(
                isinstance(name, str) and name.strip()
                for name in features
            )
            or len(features) != len(set(features))
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
                f"{track}: invalid SHA-256 checksum."
            )

    return champions


def load_champion_models(
    repository_root,
    trusted_source=False
):
    """Load SHA-256-verified champions from a trusted source.

    WARNING:
        joblib uses pickle-based deserialization, which
        can execute code. Checksums detect changes relative
        to the manifest but do not establish its provenance.
        Only deploy artifacts from a trusted CI/CD workflow.
    """

    if not trusted_source:
        raise ModelIntegrityError(
            "Trusted artifact source confirmation is required."
        )

    root = Path(repository_root).resolve()

    models_dir = (
        root / "published-models"
    ).resolve()

    manifest_path = (
        models_dir / "champion_manifest.json"
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
                f"{track}: unsafe model path."
            )

        if not model_path.is_file():
            raise ModelIntegrityError(
                f"{track}: published champion is missing."
            )

        actual_sha = sha256_file(model_path)

        if actual_sha.lower() != metadata["sha256"].lower():
            raise ModelIntegrityError(
                f"{track}: SHA-256 verification failed."
            )

        try:
            model = joblib.load(model_path)

        except Exception as exc:
            LOGGER.exception(
                "Champion loading failed: %s", track
            )
            raise ModelIntegrityError(
                f"{track}: model could not be loaded."
            ) from exc

        if not callable(
            getattr(model, "predict_proba", None)
        ):
            raise ModelIntegrityError(
                f"{track}: predict_proba unavailable."
            )

        classes = list(
            getattr(model, "classes_", [])
        )

        if 1 not in classes:
            raise ModelIntegrityError(
                f"{track}: positive class 1 is missing."
            )

        if hasattr(model, "feature_names_in_"):

            fitted_features = list(
                model.feature_names_in_
            )

            if fitted_features != metadata["feature_names"]:
                raise ModelIntegrityError(
                    f"{track}: fitted feature schema mismatch."
                )

        loaded_models[track] = model

        LOGGER.info(
            "Verified champion loaded: %s", track
        )

    return loaded_models, manifest


def validate_customer_input(
    customer_data,
    feature_metadata,
    expected_features
):
    """Normalize and validate one customer feature record."""

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
            "Missing customer fields: "
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
                f"Missing feature metadata: {feature}"
            )

        specification = feature_metadata[feature]
        value = customer_data[feature]
        feature_type = specification.get("type")

        if feature_type == "numeric":

            if value is None or (
                isinstance(value, str)
                and not value.strip()
            ):
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

            if value is None:
                normalized[feature] = np.nan
                continue

            if not isinstance(value, str):
                raise InputValidationError(
                    f"{feature} must be a text category."
                )

            category = value.strip()

            if not category:
                normalized[feature] = np.nan
                continue

            allowed = [
                str(item)
                for item in specification.get(
                    "categories", []
                )
            ]

            if allowed and category not in allowed:
                raise InputValidationError(
                    f"{feature} contains an unsupported category."
                )

            normalized[feature] = category

        else:
            raise InputValidationError(
                f"Unsupported feature type for {feature}."
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
    """Generate one validated purchase probability."""

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "Decision threshold must be numeric."
        ) from exc

    if (
        not np.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise InputValidationError(
            "Decision threshold must be between 0 and 1."
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
                "Positive class is unavailable."
            )

        positive_index = classes.index(1)

        if (
            probabilities.ndim != 2
            or probabilities.shape != (1, len(classes))
        ):
            raise ModelInferenceError(
                "Invalid probability output shape."
            )

        probability = float(
            probabilities[0, positive_index]
        )

        if (
            not np.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ModelInferenceError(
                "Model returned an invalid probability."
            )

    except ModelInferenceError:
        raise

    except Exception as exc:
        LOGGER.exception(
            "Prediction inference failed."
        )
        raise ModelInferenceError(
            "Prediction could not be completed."
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
