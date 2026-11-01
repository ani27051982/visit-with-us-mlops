"""Visit with Us — Wellness Tourism AI application."""

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.prediction_service import (
    InputValidationError,
    ModelIntegrityError,
    ModelInferenceError,
    load_champion_models,
    load_json,
    predict_customer
)

from app.utils.app_helpers import (
    format_probability,
    prediction_interpretation,
    friendly_error_message
)

LOGGER = logging.getLogger("visit_with_us.streamlit")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------
# 1. Load Application Configuration
# ------------------------------------------------------------

CONFIG = load_json(
    ROOT / "app/config/app_config.json"
)

FEATURE_METADATA = load_json(
    ROOT / "app/config/feature_metadata.json"
)

st.set_page_config(
    page_title=CONFIG["ui"]["page_title"],
    page_icon=CONFIG["ui"]["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

THEME = CONFIG["ui"]["theme"]

# ------------------------------------------------------------
# 2. Professional Application Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(
            120deg, #176B87, #20A39E
        );
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #F0FBFC;
        margin-bottom: 0;
    }

    .result-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: #F0F7FA;
        border: 1px solid #D6E8EF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 3. Load Trusted Champion Models
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner="Verifying champion models..."
)
def get_models():
    return load_champion_models(
        repository_root=ROOT,
        trusted_source=True
    )

try:
    MODELS, MANIFEST = get_models()

except (ModelIntegrityError, OSError) as exc:
    LOGGER.exception("Champion initialization failed.")
    st.error(friendly_error_message(exc))
    st.stop()

# ------------------------------------------------------------
# 4. Application Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🌍 Visit with Us — Wellness Tourism AI</h1>
        <p>
            Intelligent customer purchase prediction,
            wellness tourism insights and
            data-driven marketing support.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 5. Sidebar — Prediction Scenario
# ------------------------------------------------------------

with st.sidebar:

    st.title("🧭 Prediction Center")

    model_choice = st.radio(
        "Select Prediction Scenario",
        options=["model_b", "model_a"],
        format_func=lambda track: (
            CONFIG["models"][track]["display_name"]
        )
    )

    model_metadata = MANIFEST[
        "champions"
    ][model_choice]

    st.divider()
    st.subheader("🤖 Champion Model")

    st.write(
        "**Algorithm:**",
        model_metadata["algorithm"]
    )

    st.write(
        "**Features:**",
        len(model_metadata["feature_names"])
    )

    st.write(
        "**Variant:**",
        model_metadata.get("variant", "champion")
    )

    st.divider()

    st.caption(
        "Predictions are statistical estimates, "
        "not guaranteed customer outcomes."
    )

# ------------------------------------------------------------
# 6. Validate Model Feature Configuration
# ------------------------------------------------------------

model_config = CONFIG["models"][model_choice]

feature_names = model_metadata["feature_names"]

feature_specs = FEATURE_METADATA[model_choice]

if list(feature_specs) != feature_names:
    st.error(
        "Feature configuration does not match the "
        "published champion model."
    )
    st.stop()

st.subheader(
    model_config["display_name"]
)

st.write(
    model_config["description"]
)

st.info(
    "Enter customer information below. "
    "The form uses the model's registered feature schema."
)

# ------------------------------------------------------------
# 7. Dynamic Customer Input Form
# ------------------------------------------------------------

customer_inputs = {}

with st.form(
    key=f"prediction_form_{model_choice}",
    clear_on_submit=False
):

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(feature_names):

        specification = feature_specs[feature]

        field_label = (
            feature.replace("_", " ").strip()
        )

        with (
            left_column
            if index % 2 == 0
            else right_column
        ):

            if specification["type"] == "numeric":

                default_value = float(
                    specification.get("default", 0.0)
                )

                minimum = specification.get("minimum")
                maximum = specification.get("maximum")

                customer_inputs[feature] = (
                    st.number_input(
                        label=field_label,
                        value=default_value,
                        step=1.0,
                        help=(
                            "Observed training range: "
                            f"{minimum} to {maximum}. "
                            "This range is informational."
                        ),
                        key=f"{model_choice}_{feature}"
                    )
                )

            elif specification["type"] == "categorical":

                categories = specification.get(
                    "categories", []
                )

                if not categories:
                    st.error(
                        f"No categories configured for "
                        f"{field_label}."
                    )
                    st.stop()

                default_category = specification.get(
                    "default"
                )

                default_index = (
                    categories.index(default_category)
                    if default_category in categories
                    else 0
                )

                customer_inputs[feature] = (
                    st.selectbox(
                        label=field_label,
                        options=categories,
                        index=default_index,
                        key=f"{model_choice}_{feature}"
                    )
                )

            else:
                st.error(
                    f"Unsupported feature type: {feature}"
                )
                st.stop()

    submitted = st.form_submit_button(
        "🔮 Predict Purchase Likelihood",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------------------------
# 8. Prediction Execution
# ------------------------------------------------------------

if submitted:

    try:
        with st.spinner(
            "Analyzing customer purchase likelihood..."
        ):

            result = predict_customer(
                model=MODELS[model_choice],
                customer_data=customer_inputs,
                feature_metadata=feature_specs,
                expected_features=feature_names,
                threshold=model_config.get(
                    "threshold", 0.50
                )
            )

        probability = float(
            result["purchase_probability"]
        )

        threshold = float(
            result["decision_threshold"]
        )

        st.divider()
        st.subheader("📊 Prediction Results")

        # ----------------------------------------------------
        # 9. Prediction Summary Cards
        # ----------------------------------------------------

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Purchase Probability",
            format_probability(probability)
        )

        metric_2.metric(
            "Non-Purchase Probability",
            format_probability(1 - probability)
        )

        metric_3.metric(
            "Decision Threshold",
            format_probability(threshold)
        )

        # ----------------------------------------------------
        # 10. Probability Gauge
        # ----------------------------------------------------

        gauge_color = (
            THEME["success"]
            if result["predicted_class"] == 1
            else THEME["warning"]
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={
                    "text": "Estimated Purchase Likelihood"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": gauge_color
                    },
                    "steps": [
                        {
                            "range": [0, 35],
                            "color": "#FDE8E5"
                        },
                        {
                            "range": [35, 65],
                            "color": "#FFF2D8"
                        },
                        {
                            "range": [65, 100],
                            "color": "#DDF4EA"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "#16324F",
                            "width": 4
                        },
                        "thickness": 0.8,
                        "value": threshold * 100
                    }
                }
            )
        )

        gauge.update_layout(
            height=330,
            margin={
                "l": 30,
                "r": 30,
                "t": 60,
                "b": 20
            },
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        st.caption(
            "The threshold marker represents the "
            "classification decision boundary. "
            "Gauge bands are illustrative."
        )

        # ----------------------------------------------------
        # 11. Probability Comparison Chart
        # ----------------------------------------------------

        comparison = go.Figure(
            go.Bar(
                x=[
                    "Purchase",
                    "Non-Purchase"
                ],
                y=[
                    probability * 100,
                    (1 - probability) * 100
                ],
                marker_color=[
                    THEME["success"],
                    THEME["danger"]
                ],
                text=[
                    f"{probability:.1%}",
                    f"{1 - probability:.1%}"
                ],
                textposition="outside",
                name="Predicted Probability"
            )
        )

        comparison.update_layout(
            title=(
                "Purchase vs Non-Purchase Probability"
            ),
            xaxis_title="Prediction Outcome",
            yaxis_title="Estimated Probability (%)",
            yaxis_range=[0, 110],
            height=350,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            comparison,
            use_container_width=True
        )

        # ----------------------------------------------------
        # 12. Business Interpretation
        # ----------------------------------------------------

        st.subheader("🎯 Business Interpretation")

        if result["predicted_class"] == 1:
            st.success(
                result["predicted_label"]
            )
        else:
            st.warning(
                result["predicted_label"]
            )

        st.write(
            prediction_interpretation(result)
        )

        # ----------------------------------------------------
        # 13. Safe Customer Input Summary
        # ----------------------------------------------------
        #
        # Normalize all display values to strings to avoid
        # mixed-type PyArrow serialization problems.
        # ----------------------------------------------------

        with st.expander(
            "📋 Customer Input Summary"
        ):

            summary_rows = []

            for feature, value in customer_inputs.items():

                if value is None:
                    display_value = "Not provided"
                else:
                    display_value = str(value)

                summary_rows.append({
                    "Feature": str(feature),
                    "Value": display_value
                })

            summary_df = pd.DataFrame(
                summary_rows,
                columns=["Feature", "Value"]
            ).astype("string")

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This application does not intentionally "
            "persist customer inputs or prediction results."
        )

        LOGGER.info(
            "Prediction completed for %s.",
            model_choice
        )

    except InputValidationError as exc:
        st.error(
            friendly_error_message(exc)
        )

    except (
        ModelIntegrityError,
        ModelInferenceError
    ) as exc:

        LOGGER.exception(
            "Prediction service failure."
        )

        st.error(
            friendly_error_message(exc)
        )

    except Exception:
        LOGGER.exception(
            "Unexpected Streamlit prediction failure."
        )

        st.error(
            "An unexpected error occurred. "
            "Please try again later."
        )

# ------------------------------------------------------------
# 14. Application Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "🌍 Visit with Us — Enterprise MLOps | "
    "Python • scikit-learn • GitHub Actions • Streamlit"
)
