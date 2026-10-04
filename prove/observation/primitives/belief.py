"""Explicit multi-hypothesis belief update."""

from __future__ import annotations

from typing import Any, Mapping


def update_multi_hypothesis(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    prior = {str(key): float(value) for key, value in inputs["prior"].items()}
    likelihood = {
        str(key): float(value) for key, value in inputs["likelihood"].items()
    }
    keys = sorted(set(prior) | set(likelihood))
    if not keys:
        raise ValueError("multi-hypothesis update requires candidates")
    unnormalized = {
        key: max(0.0, prior.get(key, 0.0)) * max(0.0, likelihood.get(key, 0.0))
        for key in keys
    }
    total = sum(unnormalized.values())
    if total <= 0:
        posterior = {key: 1.0 / len(keys) for key in keys}
    else:
        posterior = {key: value / total for key, value in unnormalized.items()}
    winner = max(keys, key=lambda key: (posterior[key], key))
    ordered = sorted(posterior.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return {
        "posterior": posterior,
        "leading_hypothesis": winner,
        "belief_margin": float(margin),
    }
