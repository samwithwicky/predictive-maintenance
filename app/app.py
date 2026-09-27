import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from src.inference import InferenceEngine
from src.explainability import RULExplainer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* ----------------------------------------------------------
   Header
---------------------------------------------------------- */

.main-title {
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 2rem;
}

/* ----------------------------------------------------------
   Section headings
---------------------------------------------------------- */

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 0.35rem;
}

.section-description {
    color: #9ca3af;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 1.1rem;
}

/* ----------------------------------------------------------
   Metric cards
---------------------------------------------------------- */

.metric-card {
    padding: 1.25rem 1.35rem;
    border: 1px solid #30343b;
    border-radius: 12px;
    background: #15171c;
    min-height: 125px;
}

.metric-label {
    color: #9ca3af;
    font-size: 0.82rem;
    margin-bottom: 0.55rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
}

/* ----------------------------------------------------------
   Status
---------------------------------------------------------- */

.status-normal {
    color: #34d399;
    font-weight: 700;
    font-size: 1.35rem;
}

.status-monitor {
    color: #fbbf24;
    font-weight: 700;
    font-size: 1.35rem;
}

.status-attention {
    color: #f87171;
    font-weight: 700;
    font-size: 1.35rem;
}

/* ----------------------------------------------------------
   Factor cards
---------------------------------------------------------- */

.factor-card {
    padding: 1rem 1.2rem;
    border: 1px solid #30343b;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    background: #15171c;
}

.factor-name {
    font-weight: 600;
    font-size: 0.98rem;
    margin-bottom: 0.35rem;
}

.factor-up {
    color: #34d399;
    font-weight: 600;
    font-size: 0.9rem;
}

.factor-down {
    color: #f87171;
    font-weight: 600;
    font-size: 0.9rem;
}

.factor-magnitude {
    color: #9ca3af;
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

/* ----------------------------------------------------------
   Interpretation
---------------------------------------------------------- */

.interpretation-box {
    padding: 1.2rem 1.3rem;
    border-radius: 10px;
    background: #15171c;
    border: 1px solid #30343b;
    line-height: 1.65;
    color: #d1d5db;
}

/* ----------------------------------------------------------
   Footer
---------------------------------------------------------- */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #25282e;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Predictive Maintenance — Aircraft Engine RUL'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Remaining Useful Life estimation based on historical '
    'aircraft engine sensor measurements and operating-cycle data.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource
def load_system():

    inference = InferenceEngine()

    explainer = RULExplainer()

    data = inference.prepare(
        inference.load_data()
    )

    return inference, explainer, data


try:

    inference, explainer, data = load_system()

except Exception as e:

    st.error(
        f"Unable to initialize the predictive maintenance system: {e}"
    )

    st.stop()


# ============================================================
# EQUIPMENT SELECTION
# ============================================================

engine_ids = inference.get_engine_ids(data)

selected_engine = st.selectbox(
    "Select engine",
    engine_ids,
    format_func=lambda x: f"Engine {x}",
)


# ============================================================
# PREDICTION
# ============================================================

try:

    result = inference.predict_engine(
        data,
        selected_engine,
    )

except Exception as e:

    st.error(
        f"Unable to generate prediction: {e}"
    )

    st.stop()


engine_id = result["engine_id"]
latest_cycle = result["latest_cycle"]
predicted_rul = result["predicted_rul"]
risk = result["risk"]


# ============================================================
# EQUIPMENT STATUS
# ============================================================

st.markdown(
    '<div class="section-title">Equipment Status</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Current operating condition and estimated remaining useful life.'
    '</div>',
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Equipment
# ------------------------------------------------------------

with col1:

    st.markdown(
        f"""
<div class="metric-card">
<div class="metric-label">Equipment</div>
<div class="metric-value">Engine {engine_id}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# RUL
# ------------------------------------------------------------

with col2:

    st.markdown(
        f"""
<div class="metric-card">
<div class="metric-label">Estimated Remaining Useful Life</div>
<div class="metric-value">{predicted_rul:.1f} cycles</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Maintenance status
# ------------------------------------------------------------

with col3:

    if risk == "Normal":

        status_class = "status-normal"

    elif risk == "Monitor":

        status_class = "status-monitor"

    else:

        status_class = "status-attention"

    st.markdown(
        f"""
<div class="metric-card">
<div class="metric-label">Maintenance Status</div>
<div class="{status_class}">{risk}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.caption(
    f"Latest recorded operating cycle: {latest_cycle}"
)


# ============================================================
# TOP FACTORS AFFECTING RUL
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Top Factors Affecting Estimated RUL'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'The strongest factors influencing the current RUL estimate. '
    'Direction indicates whether each factor increases or decreases '
    'the model\'s estimated remaining useful life.'
    '</div>',
    unsafe_allow_html=True,
)


try:

    engine_history = data[
        data["unit"] == selected_engine
    ].sort_values("cycle")

    latest_row = engine_history.tail(1)

    top_factors = explainer.top_contributors(
        latest_row,
        n=10,
    )

except Exception as e:

    st.error(
        f"Unable to generate explanation: {e}"
    )

    top_factors = None


# ============================================================
# DISPLAY TOP FACTORS
# ============================================================

if top_factors is not None:

    for _, row in top_factors.iterrows():

        feature = row["feature"]

        shap_value = float(
            row["shap_value"]
        )

        if shap_value > 0:

            direction_class = "factor-up"

            direction_text = (
                "↑ Increases estimated RUL"
            )

        else:

            direction_class = "factor-down"

            direction_text = (
                "↓ Decreases estimated RUL"
            )

        st.markdown(
            f"""<div class="factor-card">
<div class="factor-name">{feature}</div>
<div class="{direction_class}">{direction_text}</div>
<div class="factor-magnitude">
Contribution magnitude: {abs(shap_value):.3f}
</div>
</div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">Maintenance Overview</div>',
    unsafe_allow_html=True,
)

if risk == "Normal":

    interpretation = (
        f"The current prediction estimates approximately "
        f"<strong>{predicted_rul:.1f} cycles</strong> "
        "of remaining useful life. The engine is currently "
        "<strong>Normal</strong> under the configured "
        "maintenance thresholds."
    )

elif risk == "Monitor":

    interpretation = (
        f"The current prediction estimates approximately "
        f"<strong>{predicted_rul:.1f} cycles</strong> "
        "of remaining useful life. The engine is currently "
        "<strong>Monitor</strong>, indicating that its condition "
        "should be observed more closely."
    )

else:

    interpretation = (
        f"The current prediction estimates approximately "
        f"<strong>{predicted_rul:.1f} cycles</strong> "
        "of remaining useful life. The engine currently requires "
        "<strong>Maintenance Attention</strong> under the "
        "configured maintenance thresholds."
    )


st.markdown(
    f"""<div class="interpretation-box">
{interpretation}

<br><br>

The factors above represent the strongest contributors to the
current RUL estimate and provide context for the prediction.
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Predictive Maintenance • Aircraft Engine Health Monitoring • '
    'Remaining Useful Life Estimation'
    '</div>',
    unsafe_allow_html=True,
)