import copy

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from adaptive_nof1.inference.physical_exercise_model import PhysicalExerciseModel
from .contextual_fixtures import ACTIONS, context, physical_model


@pytest.mark.parametrize("bad_type", [-1, 2, 0.5, np.nan, np.inf])
@pytest.mark.parametrize("fitting", [False, True])
def test_invalid_categories_fail_before_pymc(bad_type, fitting, monkeypatch):
    actions = copy.deepcopy(ACTIONS)
    actions[0]["type"] = bad_type
    model = physical_model(actions=actions)
    monkeypatch.setattr(
        pm, "set_data", lambda *args, **kwargs: pytest.fail("invalid data reached PyMC")
    )
    with pytest.raises(ValueError):
        if fitting:
            model.update_posterior(
                pd.DataFrame([{**context(2), **actions[0], "pain_reduction": 1.0}]), 2
            )
        else:
            model.approximate_max_probabilities(2, context(8))


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "intensity",
        "duration",
        "current_pain",
        "mean_duration",
        "mean_intensity",
        "pain_reduction",
    ],
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "invalid"])
def test_invalid_training_inputs_do_not_replace_model_state(
    field, bad_value, monkeypatch
):
    model = physical_model()
    before = model.trace
    row = {**context(2), **ACTIONS[0], "pain_reduction": 1.0}
    row[field] = bad_value
    monkeypatch.setattr(
        pm, "set_data", lambda *args, **kwargs: pytest.fail("invalid data reached PyMC")
    )
    with pytest.raises(ValueError):
        model.update_posterior(pd.DataFrame([row]), 2)
    assert model.trace is before


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "intensity",
        "duration",
        "current_pain",
        "mean_duration",
        "mean_intensity",
        "pain_reduction",
    ],
)
def test_missing_training_columns_are_rejected(field):
    row = {**context(2), **ACTIONS[0], "pain_reduction": 1.0}
    del row[field]
    with pytest.raises(ValueError):
        PhysicalExerciseModel(2, ACTIONS).update_posterior(pd.DataFrame([row]), 2)


@pytest.mark.parametrize("field", ["current_pain", "mean_duration", "mean_intensity"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "invalid"])
def test_invalid_scoring_context_is_rejected(field, bad_value):
    supplied = context(8)
    supplied[field] = bad_value
    with pytest.raises(ValueError):
        physical_model().approximate_max_probabilities(2, supplied)


@pytest.mark.parametrize("count", [0, 1, 3, 2.0, "2", True, None])
def test_invalid_candidate_counts_are_rejected(count):
    with pytest.raises(ValueError, match="number_of_treatments"):
        physical_model().approximate_max_probabilities(count, context(8))


def test_empty_candidate_catalog_is_rejected():
    with pytest.raises(ValueError, match="number_of_treatments"):
        physical_model(actions=[]).approximate_max_probabilities(0, context(8))


@pytest.mark.parametrize("dimension", [0, -1, 2.0, True, None])
def test_invalid_category_dimension_is_rejected(dimension):
    with pytest.raises(ValueError, match="dimension_for_type"):
        PhysicalExerciseModel(dimension, ACTIONS)


@pytest.mark.parametrize("as_strings", [False, True])
def test_valid_numeric_categories_are_normalized_for_fitting_and_scoring(
    as_strings, monkeypatch
):
    actions = copy.deepcopy(ACTIONS)
    for action in actions:
        action["type"] = str(action["type"]) if as_strings else float(action["type"])
    model = physical_model(actions=actions)
    trace = model.trace
    monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: trace)
    model.update_posterior(
        pd.DataFrame([{**context(2), **actions[0], "pain_reduction": 1.0}]), 2
    )
    np.testing.assert_array_equal(
        model.approximate_max_probabilities(2, context(8)), [0, 1]
    )
    assert model.model["types"].get_value().dtype.kind == "i"


def test_missing_candidate_descriptor_is_not_filled_from_previous_context():
    actions = copy.deepcopy(ACTIONS)
    del actions[1]["duration"]
    with pytest.raises(ValueError):
        physical_model(actions=actions).approximate_max_probabilities(
            2, {**context(8), **ACTIONS[0]}
        )
