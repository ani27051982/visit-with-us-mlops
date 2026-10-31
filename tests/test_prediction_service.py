"""Unit tests for prediction service validation."""

import numpy as np
import pytest

from app.services.prediction_service import (
    InputValidationError,
    validate_customer_input
)


def test_numeric_customer_input():
    metadata = {
        "Age": {
            "type": "numeric",
            "default": 35,
            "minimum": 18,
            "maximum": 80
        }
    }

    result = validate_customer_input(
        {"Age": 35},
        metadata,
        ["Age"]
    )

    assert result.shape == (1, 1)
    assert float(result.loc[0, "Age"]) == 35


def test_invalid_numeric_input():
    metadata = {
        "Age": {"type": "numeric"}
    }

    with pytest.raises(InputValidationError):
        validate_customer_input(
            {"Age": "invalid"},
            metadata,
            ["Age"]
        )


def test_missing_required_feature():
    metadata = {
        "Age": {"type": "numeric"}
    }

    with pytest.raises(InputValidationError):
        validate_customer_input(
            {},
            metadata,
            ["Age"]
        )


def test_unexpected_feature():
    metadata = {
        "Age": {"type": "numeric"}
    }

    with pytest.raises(InputValidationError):
        validate_customer_input(
            {"Age": 35, "Unknown": 1},
            metadata,
            ["Age"]
        )


def test_supported_categorical_input():
    metadata = {
        "Occupation": {
            "type": "categorical",
            "categories": ["Salaried", "Business"]
        }
    }

    result = validate_customer_input(
        {"Occupation": "Salaried"},
        metadata,
        ["Occupation"]
    )

    assert result.loc[0, "Occupation"] == "Salaried"


def test_unsupported_categorical_input():
    metadata = {
        "Occupation": {
            "type": "categorical",
            "categories": ["Salaried", "Business"]
        }
    }

    with pytest.raises(InputValidationError):
        validate_customer_input(
            {"Occupation": "Unknown"},
            metadata,
            ["Occupation"]
        )
