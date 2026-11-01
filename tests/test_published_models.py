"""Integration tests for trusted published champions."""

from pathlib import Path

import numpy as np
import pytest

from app.services.prediction_service import (
    load_champion_models,
    load_json,
    predict_customer
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_context():

    models, manifest = load_champion_models(
        ROOT,
        trusted_source=True
    )

    feature_metadata = load_json(
        ROOT / "app/config/feature_metadata.json"
    )

    return models, manifest, feature_metadata


@pytest.mark.parametrize(
    "track",
    ["model_a", "model_b"]
)
def test_champion_inference(track, model_context):

    models, manifest, feature_metadata = model_context

    expected_features = manifest[
        "champions"
    ][track]["feature_names"]

    sample = {
        feature: feature_metadata[
            track
        ][feature].get("default")
        for feature in expected_features
    }

    result = predict_customer(
        model=models[track],
        customer_data=sample,
        feature_metadata=feature_metadata[track],
        expected_features=expected_features,
        threshold=0.50
    )

    probability = result["purchase_probability"]

    assert np.isfinite(probability)
    assert 0 <= probability <= 1
    assert result["predicted_class"] in (0, 1)
    assert result["predicted_label"] in (
        "Likely Purchaser",
        "Unlikely Purchaser"
    )
