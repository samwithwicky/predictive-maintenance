import numpy as np

from src.model import RULModel


def test_model_loads():
    model = RULModel()

    assert model.model is not None


def test_model_has_expected_feature_count():
    model = RULModel()

    assert len(model.features) == 60


def test_features_are_unique():
    model = RULModel()

    assert len(model.features) == len(set(model.features))


def test_model_has_60_input_features():
    model = RULModel()

    assert model.model.n_features_in_ == 60