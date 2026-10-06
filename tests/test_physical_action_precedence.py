import copy

import numpy as np
import pytest

from .contextual_fixtures import ACTIONS, context, physical_model


@pytest.mark.parametrize(
    "previous_action",
    [ACTIONS[0], dict(type=1, intensity=9.0, duration=9.0)],
)
def test_candidate_descriptors_override_previous_action_context(previous_action):
    model = physical_model()
    supplied_context = {**context(8), **previous_action}
    before = copy.deepcopy(supplied_context)
    np.testing.assert_array_equal(
        model.approximate_max_probabilities(2, supplied_context), [0, 1]
    )
    np.testing.assert_array_equal(model.model["types"].get_value(), [0, 1])
    np.testing.assert_array_equal(model.model["intensities"].get_value(), [1.0, 2.0])
    np.testing.assert_array_equal(model.model["durations"].get_value(), [1.0, 1.0])
    assert supplied_context == before
    assert model.possible_actions == ACTIONS


def test_previous_action_metadata_does_not_change_candidate_order():
    model = physical_model(actions=ACTIONS[::-1])
    np.testing.assert_array_equal(
        model.approximate_max_probabilities(2, {**context(8), **ACTIONS[0]}), [1, 0]
    )
