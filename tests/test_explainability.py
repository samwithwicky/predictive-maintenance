from src.inference import InferenceEngine
from src.explainability import RULExplainer


def test_explainer_loads():
    engine = InferenceEngine()

    explainer = RULExplainer(engine.model)

    assert explainer is not None


def test_shap_generates_for_engine():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    observation = engine.get_latest_observation(
        data,
        2
    )

    explainer = RULExplainer(engine.model)

    result = explainer.top_contributors(
        observation,
        10
    )

    assert not result.empty


def test_shap_returns_expected_columns():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    observation = engine.get_latest_observation(
        data,
        2
    )

    explainer = RULExplainer(engine.model)

    result = explainer.top_contributors(
        observation,
        10
    )

    expected_columns = {
        "feature",
        "feature_value",
        "shap_value",
        "direction"
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_shap_returns_requested_number_of_features():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    observation = engine.get_latest_observation(
        data,
        2
    )

    explainer = RULExplainer(engine.model)

    result = explainer.top_contributors(
        observation,
        10
    )

    assert len(result) == 10