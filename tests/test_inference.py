from src.inference import InferenceEngine


def test_inference_engine_loads():
    engine = InferenceEngine()

    assert engine is not None


def test_test_dataset_loads():
    engine = InferenceEngine()

    data = engine.load_data()

    assert not data.empty


def test_all_engines_are_available():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    engines = engine.get_engine_ids(data)

    assert len(engines) == 100


def test_single_engine_prediction():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    result = engine.predict_engine(data, 2)

    assert result["engine_id"] == 2
    assert result["latest_cycle"] > 0
    assert result["predicted_rul"] >= 0
    assert result["risk"] in {
        "Normal",
        "Monitor",
        "Maintenance Attention"
    }


def test_multiple_engines_can_be_predicted():
    engine = InferenceEngine()

    data = engine.prepare(engine.load_data())

    for engine_id in [1, 2, 31, 50, 100]:

        result = engine.predict_engine(
            data,
            engine_id
        )

        assert result["predicted_rul"] >= 0