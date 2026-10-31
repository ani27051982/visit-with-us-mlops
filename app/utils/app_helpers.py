"""Reusable presentation helpers for Visit with Us."""

import logging

LOGGER = logging.getLogger("visit_with_us.app_helpers")


def format_probability(probability):
    """Format a probability as a percentage."""

    return f"{float(probability) * 100:.1f}%"


def prediction_interpretation(result):
    """Return a non-causal, business-oriented explanation."""

    probability = float(result["purchase_probability"])
    threshold = float(result["decision_threshold"])

    if probability >= threshold:
        return (
            "The model estimates that this customer is "
            "more likely to purchase under the selected "
            "classification threshold. Consider the result "
            "alongside business eligibility and outreach capacity."
        )

    return (
        "The model estimates a lower purchase likelihood "
        "under the selected classification threshold. "
        "This does not mean the customer will never purchase."
    )


def friendly_error_message(error):
    """Return a safe, readable application error message."""

    error_name = type(error).__name__

    if error_name == "InputValidationError":
        return str(error)

    if error_name == "ModelIntegrityError":
        return (
            "The prediction model could not be verified. "
            "Please contact the application administrator."
        )

    return (
        "The requested prediction could not be completed. "
        "Please try again or contact support."
    )
