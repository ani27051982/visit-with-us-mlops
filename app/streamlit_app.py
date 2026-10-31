"""Visit with Us — Wellness Tourism AI Streamlit application."""

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
# 1. Application Configuration
# ------------------------------------------------------------

CONFIG = load_json(ROOT / "app/config/app_config.json")
FEATURE_METADATA = load_json(
    ROOT / "app/config/feature_metadata.json"
)

st.set_page_config(
    page_title=CONFIG["ui"]["page_title"],
    page_icon=CONFIG["ui"]["page_icon"],
    layout="wide"
)

THEME = CONFIG["ui"]["theme"]

# ------------------------------------------------------------
# 2. Application Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
    }
    .hero {
        background: linear-gradient(120deg, #176B87, #20A39E);
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
# 3. Trusted Model Loading
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Verifying champion models...")
def get_models():
    return load_champion_models(
        repository_root=ROOT,
        trusted_source=True
    )

try:
    MODELS, MANIFEST = get_models()

except (ModelIntegrityError, OSError) as exc:
    LOGGER.exception("Model initialization failed.")
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
        Intelligent customer purchase prediction and
        data-driven wellness tourism marketing.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 5. Sidebar Navigation
# ------------------------------------------------------------

with st.sidebar:
    st.title("🧭 Prediction Center")

    model_choice = st.radio(
        "Prediction Scenario",
        ["model_b", "model_a"],
        format_func=lambda key: (
            CONFIG["models"][key]["display_name"]
        )
    )

    metadata = MANIFEST["champions"][model_choice]

    st.divider()
    st.subheader("🤖 Champion Model")
    st.write("**Algorithm:**", metadata["algorithm"])
    st.write("**Features:**", len(metadata["feature_names"]))
    st.write("**Variant:**", metadata.get("variant", "champion"))

    st.caption(
        "Predictions are estimates and do not guarantee "
        "customer purchasing behavior."
    )

# ------------------------------------------------------------
# 6. Model-Specific Feature Configuration
# ------------------------------------------------------------

model_config = CONFIG["models"][model_choice]
feature_names = metadata["feature_names"]
feature_specs = FEATURE_METADATA[model_choice]

if list(feature_specs) != feature_names:
    st.error("Model feature configuration mismatch.")
    st.stop()

st.subheader(model_config["display_name"])
st.write(model_config["description"])

# ------------------------------------------------------------
# 7. Dynamic Customer Input Form
# ------------------------------------------------------------

customer_inputs = {}

with st.form(f"prediction_form_{model_choice}"):

    left, right = st.columns(2)

    for index, feature in enumerate(feature_names):

        spec = feature_specs[feature]
        label = feature.replace("_", " ").strip()

        with (left if index % 2 == 0 else right):

            if spec["type"] == "numeric":

                default = float(spec.get("default", 0.0))

                customer_inputs[feature] = st.number_input(
                    label,
                    value=default,
                    step=1.0,
                    help=(
                        "Training range: "
                        f"{spec.get('minimum')} to "
                        f"{spec.get('maximum')}"
                    ),
                    key=f"{model_choice}_{feature}"
                )

            elif spec["type"] == "categorical":

                options = spec.get("categories", [])

                if not options:
                    st.error(f"No categories available for {label}.")
                    st.stop()

                default = spec.get("default")
                selected_index = (
                    options.index(default)
                    if default in options else 0
                )

                customer_inputs[feature] = st.selectbox(
                    label,
                    options=options,
                    index=selected_index,
                    key=f"{model_choice}_{feature}"
                )

            else:
                st.error(f"Unsupported feature type: {feature}")
                st.stop()

    submitted = st.form_submit_button(
        "🔮 Predict Purchase Likelihood",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------------------------
# 8. Prediction & Results
# ------------------------------------------------------------

if submitted:
    try:
        result = predict_customer(
            model=MODELS[model_choice],
            customer_data=customer_inputs,
            feature_metadata=feature_specs,
            expected_features=feature_names,
            threshold=model_config.get("threshold", 0.50)
        )

        probability = result["purchase_probability"]
        threshold = result["decision_threshold"]

        st.divider()
        st.subheader("📊 Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Purchase Probability",
            format_probability(probability)
        )
        c2.metric(
            "Non-Purchase Probability",
            format_probability(1 - probability)
        )
        c3.metric(
            "Decision Threshold",
            format_probability(threshold)
        )

        # ----------------------------------------------------
        # Probability Gauge
        # ----------------------------------------------------

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Estimated Purchase Likelihood"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": (
                        THEME["success"]
                        if result["predicted_class"] == 1
                        else THEME["warning"]
                    )
                },
                "steps": [
                    {"range": [0, 35], "color": "#FDE8E5"},
                    {"range": [35, 65], "color": "#FFF2D8"},
                    {"range": [65, 100], "color": "#DDF4EA"}
                ],
                "threshold": {
                    "line": {
                        "color": "#16324F",
                        "width": 4
                    },
                    "value": threshold * 100
                }
            }
        ))

        gauge.update_layout(
            height=330,
            margin=dict(l=30, r=30, t=60, b=20)
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.caption(
            "The threshold marker indicates the classification "
            "decision boundary. Gauge bands are illustrative."
        )

        # ----------------------------------------------------
        # Purchase Probability Comparison
        # ----------------------------------------------------

        comparison = go.Figure(go.Bar(
            x=["Purchase", "Non-Purchase"],
            y=[probability * 100, (1 - probability) * 100],
            marker_color=[
                THEME["success"],
                THEME["danger"]
            ],
            text=[
                f"{probability:.1%}",
                f"{1 - probability:.1%}"
            ],
            textposition="outside",
            name="Estimated Probability"
        ))

        comparison.update_layout(
            title="Purchase vs Non-Purchase Probability",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 110],
            height=350,
            showlegend=True
        )

        st.plotly_chart(
            comparison,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Business Interpretation
        # ----------------------------------------------------

        st.subheader("🎯 Business Interpretation")

        if result["predicted_class"] == 1:
            st.success(result["predicted_label"])
        else:
            st.warning(result["predicted_label"])

        st.write(prediction_interpretation(result))

        with st.expander("📋 Customer Input Summary"):

            summary_rows = [
                {
                    "Feature": str(feature),
                    "Value": (
                        "Not provided"
                        if value is None or pd.isna(value)
                        else str(value)
                    )
                }
                for feature, value in customer_inputs.items()
            ]

            summary_df = pd.DataFrame(
                summary_rows,
                columns=["Feature", "Value"]
            ).astype("string")

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        LOGGER.info(
            "Prediction completed for model track %s.",
            model_choice
        )

    except InputValidationError as exc:
        st.error(friendly_error_message(exc))

    except (ModelIntegrityError, ModelInferenceError) as exc:
        LOGGER.exception("Prediction service failed.")
        st.error(friendly_error_message(exc))

    except Exception:
        LOGGER.exception("Unexpected prediction failure.")
        st.error(
            "An unexpected error occurred. "
            "Please try again later."
        )

# ------------------------------------------------------------
# 9. Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Visit with Us — Enterprise MLOps | "
    "Python • scikit-learn • GitHub Actions • Streamlit"
)
