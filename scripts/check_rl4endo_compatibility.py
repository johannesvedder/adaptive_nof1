"""Exercise the downstream fit/serialize/score functions without a Redis service.

Run with PYTHONPATH=src and --downstream /path/to/rl4endo_agent-main/main.py.
The sampler uses two chains and one core. --draws defaults to the app's 2000.
The job lookup is controlled; Flask routes, Redis and RQ workers are not tested.
"""

import argparse
import ast
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from adaptive_nof1.inference.physical_exercise_model import PhysicalExerciseModel


ACTIONS = [
    dict(type=0, intensity=1.0, duration=1.0),
    dict(type=1, intensity=2.0, duration=1.0),
]


def context(pain):
    return dict(current_pain=float(pain), mean_intensity=0.0, mean_duration=0.0)


def posterior_hash(model):
    digest = hashlib.sha256()
    for name in sorted(model.trace.posterior.data_vars):
        values = np.ascontiguousarray(model.trace.posterior[name].values)
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def restore_result(path):
    model = cloudpickle.loads(path.read_bytes())
    scores = model.approximate_max_probabilities(2, context(8))
    assert np.isfinite(scores).all() and np.isclose(scores.sum(), 1.0)
    return dict(posterior_sha256=posterior_hash(model), scores=scores.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--restore-file", type=Path)
    args = parser.parse_args()
    if args.restore_file:
        print(json.dumps(restore_result(args.restore_file)))
        return
    if not args.downstream or args.draws < 1:
        parser.error("--downstream and a positive --draws are required")

    source = args.downstream.read_text()
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"sample_bayesian_model", "create_recommendation"}
    ]
    assert (
        len(functions) == 2
    ), "Expected the downstream fit and recommendation functions"
    result_bytes = None

    class Job:
        @staticmethod
        def fetch(id, connection):
            assert id == "staging-job"
            return SimpleNamespace(return_value=lambda: result_bytes)

    namespace = dict(
        pandas=pd,
        numpy=np,
        cloudpickle=cloudpickle,
        random=random,
        PhysicalExerciseModel=PhysicalExerciseModel,
        Job=Job,
        conn=None,
    )
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(args.downstream), "exec"
        ),
        namespace,
    )
    native_sample = pm.sample

    def staging_sample(draws, **kwargs):
        assert draws == 2000, "The downstream library sampling contract changed"
        return native_sample(
            args.draws,
            chains=2,
            cores=1,
            random_seed=605,
            compute_convergence_checks=False,
            **kwargs
        )

    pm.sample = staging_sample
    rows = [
        {**context(pain), **ACTIONS[action], "pain_reduction": reward}
        for pain, action, reward in [(2, 0, 1.0), (8, 1, 2.0), (4, 0, 1.5)]
    ]
    result_bytes = namespace["sample_bayesian_model"](rows, ACTIONS)
    pm.sample = native_sample
    model = cloudpickle.loads(result_bytes)
    assert model.trace.posterior.sizes["draw"] == args.draws
    with tempfile.TemporaryDirectory(prefix="adaptive-nof1-roundtrip-") as output:
        artifact = Path(output) / "synthetic-model.cloudpickle"
        artifact.write_bytes(result_bytes)
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--restore-file",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        restored = json.loads(child.stdout.strip().splitlines()[-1])
    assert restored["posterior_sha256"] == posterior_hash(model)
    native_recommendation = namespace["create_recommendation"](
        "staging-job", ACTIONS, context(8)
    )
    assert (
        native_recommendation["action_description"]
        == ACTIONS[native_recommendation["action"]]
    )

    # A controlled crossover makes stale predictions and conflicting descriptors
    # observable independently of inference uncertainty in the diagnostic fit.
    model.trace = az.from_dict(
        posterior={
            "type_intercept": np.tile([0.0, -5.0], (1, 30, 1)),
            "intensity_coefficients": np.zeros((1, 30, 2)),
            "duration_coefficients": np.zeros((1, 30, 2)),
            "pain_coefficients": np.tile([0.0, 1.0], (1, 30, 1)),
            "sigma": np.full((1, 30), 0.001),
        }
    )
    model.approximate_max_probabilities(2, context(2))
    result_bytes = cloudpickle.dumps(model)
    winners = [
        namespace["create_recommendation"](
            "staging-job", ACTIONS, {**context(pain), **ACTIONS[0]}
        )["action"]
        for pain in (8, 2, 8)
    ]
    assert winners == [1, 0, 1]
    print(
        json.dumps(
            dict(
                downstream=str(args.downstream),
                downstream_sha256=hashlib.sha256(source.encode()).hexdigest(),
                draws_per_chain=args.draws,
                chains=2,
                fresh_interpreter=restored,
                controlled_winners=winners,
                transport="controlled Job.fetch; no Redis/RQ service",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
