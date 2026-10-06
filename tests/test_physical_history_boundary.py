import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from adaptive_nof1.basic_types import History, Observation
from adaptive_nof1.inference.physical_exercise_model import PhysicalExerciseModel
from .contextual_fixtures import ACTIONS, context


@pytest.mark.parametrize("as_history", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_fit_accepts_dataframe_and_library_history(as_history, empty, monkeypatch):
    observations = (
        []
        if empty
        else [Observation(context(2), ACTIONS[0], {"pain_reduction": 1.0}, 0, 7)]
    )
    history = History(observations)
    data = history if as_history else history.to_df()
    before = data.to_df() if as_history else data.copy(deep=True)
    model = PhysicalExerciseModel(2, ACTIONS)
    trace = object()
    monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: trace)
    model.update_posterior(data, 2)
    assert model.trace is trace
    np.testing.assert_array_equal(
        model.model["observed_outcomes"].get_value(), [] if empty else [1.0]
    )
    assert model.model["types"].get_value().dtype.kind == "i"
    pd.testing.assert_frame_equal(data.to_df() if as_history else data, before)


def test_refit_replaces_training_rows_after_prediction(monkeypatch):
    from .contextual_fixtures import physical_model

    model = physical_model()
    trace = model.trace
    monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: trace)
    rows = pd.DataFrame(
        [
            {**context(2), **ACTIONS[0], "pain_reduction": 1.0},
            {**context(8), **ACTIONS[1], "pain_reduction": 2.0},
            {**context(4), **ACTIONS[0], "pain_reduction": 3.0},
        ]
    )
    model.update_posterior(rows.iloc[:1], 2)
    model.approximate_max_probabilities(2, context(8))
    model.update_posterior(rows, 2)
    np.testing.assert_array_equal(
        model.model["observed_outcomes"].get_value(), [1.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(model.model["pains"].get_value(), [2.0, 8.0, 4.0])
    assert model._latest_posterior_predictive is None


@pytest.mark.sampling
@pytest.mark.parametrize("empty", [False, True])
def test_native_fit_can_predict_with_different_row_count(empty, monkeypatch):
    native_sample = pm.sample

    def diagnostic_sample(*args, **kwargs):
        return native_sample(
            100,
            tune=100,
            chains=2,
            cores=1,
            random_seed=605,
            progressbar=False,
            compute_convergence_checks=False,
        )

    monkeypatch.setattr(pm, "sample", diagnostic_sample)
    model = PhysicalExerciseModel(2, ACTIONS)
    rows = (
        []
        if empty
        else [
            Observation(
                context(pain), ACTIONS[action], {"pain_reduction": reward}, t, 7
            )
            for t, (pain, action, reward) in enumerate(
                [(2, 0, 1.0), (8, 1, 2.0), (4, 0, 1.5)]
            )
        ]
    )
    model.update_posterior(History(rows), 2)
    assert isinstance(model.trace, az.InferenceData)
    assert model.trace.posterior.sizes["draw"] == 100
    scores = model.approximate_max_probabilities(2, context(8))
    assert len(scores) == 2
    assert np.isfinite(scores).all()
    assert scores.sum() == pytest.approx(1.0)
