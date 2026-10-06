import arviz as az
import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from adaptive_nof1.basic_types import History, Observation
from adaptive_nof1.inference.bayes import (
    BernoulliLogItInferenceModel,
    LinearAdditiveInferenceModel,
)
from adaptive_nof1.inference.interlinked_additive_model import InterlinkedAdditiveModel
from .contextual_fixtures import ACTIONS, context, physical_model


@pytest.fixture(autouse=True)
def predictive_seed(monkeypatch):
    native = pm.sample_posterior_predictive

    def seeded(*args, **kwargs):
        return native(*args, **kwargs, random_seed=731, progressbar=False)

    monkeypatch.setattr(pm, "sample_posterior_predictive", seeded)


def test_fresh_physical_context_preserves_fitted_trace():
    model = physical_model()
    before = model.trace.copy()
    scores = [model.approximate_max_probabilities(2, context(p)) for p in (2, 8, 2)]
    np.testing.assert_array_equal(scores, [[1, 0], [0, 1], [1, 0]])
    assert model.trace.groups() == before.groups()
    assert model.trace.posterior.identical(before.posterior)


def test_refit_discards_previous_predictive_bounds(monkeypatch):
    model = physical_model()
    model.approximate_max_probabilities(2, context(8))
    replacement = model.trace.copy()
    monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: replacement)
    model.update_posterior(
        pd.DataFrame([{**context(2), **ACTIONS[0], "pain_reduction": 1.0}]), 2
    )
    assert model.trace is replacement
    assert model._latest_posterior_predictive is None


def test_legacy_serialized_trace_remains_readable():
    model = physical_model()
    model.trace.add_groups(
        posterior_predictive=az.from_dict(
            posterior_predictive={"outcome": np.tile([20.0, 0.0], (1, 30, 1))}
        ).posterior_predictive
    )
    del model._latest_posterior_predictive
    assert (
        model.get_upper_confidence_bounds("outcome")[0]
        > model.get_upper_confidence_bounds("outcome")[1]
    )


@pytest.mark.parametrize("kind", ["linear", "logit", "interlinked"])
def test_other_contextual_scorers_use_fresh_predictions(kind, monkeypatch):
    if kind == "interlinked":
        model = InterlinkedAdditiveModel([2], ["treatment"], [["pain"]])
        model.setup_model()
        model.trace = az.from_dict(
            posterior={
                "intercept_treatment": np.tile([0.0, -100.0], (1, 30, 1)),
                "slopes_treatment": np.tile([[0.0], [20.0]], (1, 30, 1, 1)),
            }
        )
    else:
        cls = (
            LinearAdditiveInferenceModel
            if kind == "linear"
            else BernoulliLogItInferenceModel
        )
        model = cls(["pain"], 1.0, 1.0)
        trace = az.from_dict(
            posterior={
                "intercept": np.tile([0.0, -5.0], (1, 30, 1)),
                "slopes": np.tile([[0.0], [1.0]], (1, 30, 1, 1)),
            }
        )
        monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: trace)
        model.update_posterior(
            History(
                [Observation({"pain": 0.0}, {"treatment": 0}, {"outcome": 0.0}, 0, 7)]
            ),
            2,
        )
    before = model.trace.copy()
    scores = [
        model.approximate_max_probabilities(2, {"pain": p}) for p in (2.0, 8.0, 2.0)
    ]
    np.testing.assert_array_equal(scores, [[1, 0], [0, 1], [1, 0]])
    assert model.trace.groups() == before.groups()
    assert model.trace.posterior.identical(before.posterior)


@pytest.mark.parametrize("after_prediction", [False, True])
def test_cloudpickle_roundtrip_uses_new_context(after_prediction):
    model = physical_model()
    if after_prediction:
        model.approximate_max_probabilities(2, context(2))
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    np.testing.assert_array_equal(
        restored.approximate_max_probabilities(2, context(8)), [0, 1]
    )
    np.testing.assert_array_equal(
        restored.approximate_max_probabilities(2, context(2)), [1, 0]
    )


def test_existing_predictive_group_does_not_override_new_scores():
    model = physical_model()
    model.trace.add_groups(
        posterior_predictive=az.from_dict(
            posterior_predictive={"outcome": np.tile([20.0, 0.0], (1, 30, 1))}
        ).posterior_predictive
    )
    np.testing.assert_array_equal(
        model.approximate_max_probabilities(2, context(8)), [0, 1]
    )
    np.testing.assert_array_equal(
        model.trace.posterior_predictive.outcome[0, 0], [20, 0]
    )


def test_reordering_candidates_reorders_scores():
    model = physical_model()
    before = model.approximate_max_probabilities(2, context(2))
    model.possible_actions.reverse()
    after = model.approximate_max_probabilities(2, context(2))
    np.testing.assert_array_equal(before, after[::-1])


def test_upper_bounds_use_latest_predictive_values():
    model = physical_model()
    model.approximate_max_probabilities(2, context(2))
    low_pain = model.get_upper_confidence_bounds("outcome")
    model.approximate_max_probabilities(2, context(8))
    high_pain = model.get_upper_confidence_bounds("outcome")
    assert low_pain[0] > low_pain[1]
    assert high_pain[1] > high_pain[0]
    assert "posterior_predictive" not in model.trace.groups()
