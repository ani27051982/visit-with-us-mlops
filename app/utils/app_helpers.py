"""Presentation helpers for Visit with Us Streamlit."""

import logging

LOGGER = logging.getLogger("visit_with_us.app_helpers")


def format_probability(probability):
    """Format probability as a percentage."""

    return f"{float(probability) * 100:.1f}%"


def prediction_interpretation(result):
    """Provide a business-oriented prediction explanation."""

    probability = float(
        result["purchase_probability"]
    )

    threshold = float(
        result["decision_threshold"]
    )

    if probability >= threshold:
        return (
            "The model estimates a higher purchase "
            "likelihood under the selected classification "
            "threshold. Consider the result alongside "
            "business eligibility and outreach capacity."
        )

    return (
        "The model estimates a lower purchase likelihood "
        "under the selected classification threshold. "
        "This does not mean the customer will never purchase."
    )


def friendly_error_message(error):
    """Convert internal exceptions to safe UI messages."""

    error_name = type(error).__name__

    if error_name == "InputValidationError":
        return str(error)

    if error_name == "ModelIntegrityError":
        return (
            "The prediction model could not be verified. "
            "Please contact the application administrator."
        )

    return (
        "The prediction could not be completed. "
        "Please try again later."
    )
