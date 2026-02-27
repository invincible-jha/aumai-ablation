"""Shared test fixtures for aumai-ablation."""

from __future__ import annotations

import pytest

from aumai_ablation.models import AblationConfig, AblationResult, AblationRun, Component


@pytest.fixture()
def two_components() -> list[Component]:
    return [
        Component(name="retriever", enabled=True, config={"top_k": 5}),
        Component(name="reranker", enabled=True),
    ]


@pytest.fixture()
def three_components() -> list[Component]:
    return [
        Component(name="retriever", enabled=True),
        Component(name="reranker", enabled=True),
        Component(name="generator", enabled=True),
    ]


@pytest.fixture()
def basic_config(two_components) -> AblationConfig:
    return AblationConfig(
        base_components=two_components,
        metrics_to_track=["accuracy", "latency_ms"],
    )


@pytest.fixture()
def completed_result(basic_config) -> AblationResult:
    """An AblationResult with metrics filled in."""
    from aumai_ablation.core import AblationStudy

    study = AblationStudy()
    runs = study.generate_runs(basic_config)

    # Fill metrics: baseline is best
    for run in runs:
        if run.disabled_component is None:
            run.metrics = {"accuracy": 0.90, "latency_ms": 100.0}
        elif run.disabled_component == "retriever":
            run.metrics = {"accuracy": 0.70, "latency_ms": 90.0}
        elif run.disabled_component == "reranker":
            run.metrics = {"accuracy": 0.85, "latency_ms": 95.0}

    return AblationResult(config=basic_config, runs=runs)


SAMPLE_YAML_COMPONENTS = """\
- name: retriever
  enabled: true
  config:
    top_k: 5
- name: reranker
  enabled: true
"""
