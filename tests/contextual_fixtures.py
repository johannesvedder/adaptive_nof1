"""Controlled posteriors for exercising real PyMC predictive graphs."""

import copy

import arviz as az
import numpy as np

from adaptive_nof1.inference.physical_exercise_model import PhysicalExerciseModel


ACTIONS = [
    dict(type=0, intensity=1.0, duration=1.0),
    dict(type=1, intensity=2.0, duration=1.0),
]


def context(pain):
    return dict(current_pain=float(pain), mean_intensity=0.0, mean_duration=0.0)


def physical_model(draws=30, actions=None):
    model = PhysicalExerciseModel(
        2, copy.deepcopy(ACTIONS if actions is None else actions)
    )
    model.setup_model()
    model.trace = az.from_dict(
        posterior={
            "type_intercept": np.tile([0.0, -5.0], (1, draws, 1)),
            "intensity_coefficients": np.zeros((1, draws, 2)),
            "duration_coefficients": np.zeros((1, draws, 2)),
            "pain_coefficients": np.tile([0.0, 1.0], (1, draws, 1)),
            "sigma": np.full((1, draws), 0.001),
        }
    )
    return model
