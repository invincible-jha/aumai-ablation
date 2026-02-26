"""Pydantic v2 models for aumai-ablation.

Provides typed structures for ablation study configuration, individual runs,
and aggregated results used by the ablation engine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Component",
    "AblationConfig",
    "AblationRun",
    "AblationResult",
]


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Component(BaseModel):
    """A named, toggle-able component in an agent or ML system.

    Example::

        c = Component(name="retriever", enabled=True, config={"top_k": 5})
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    name: str = Field(min_length=1, max_length=128)
    enabled: bool = Field(default=True)
    config: dict[str, Any] = Field(default_factory=dict)


class AblationConfig(BaseModel):
    """Configuration for an ablation study.

    Example::

        config = AblationConfig(
            base_components=[Component(name="retriever")],
            metrics_to_track=["accuracy", "latency_ms"],
            repetitions=3,
        )
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    base_components: list[Component] = Field(
        min_length=1,
        description="Full component set for the baseline (all enabled).",
    )
    metrics_to_track: list[str] = Field(
        min_length=1,
        description="Names of metrics the study will record.",
    )
    repetitions: int = Field(
        default=1,
        ge=1,
        description="Number of repetitions per ablation run for variance reduction.",
    )


class AblationRun(BaseModel):
    """A single ablation run with one component disabled (or the baseline).

    Example::

        run = AblationRun(
            run_id="run-001",
            disabled_component="retriever",
            components=[Component(name="retriever", enabled=False)],
            metrics={"accuracy": 0.72, "latency_ms": 120.0},
        )
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    run_id: str = Field(min_length=1)
    disabled_component: str | None = Field(
        default=None,
        description="Name of the disabled component, or None for the baseline run.",
    )
    components: list[Component] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class AblationResult(BaseModel):
    """Aggregated results of a complete ablation study.

    Example::

        result = AblationResult(
            config=config,
            runs=[baseline_run, run_a, run_b],
            component_importance={"retriever": 0.15, "reranker": 0.08},
        )
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    config: AblationConfig
    runs: list[AblationRun] = Field(default_factory=list)
    component_importance: dict[str, float] = Field(
        default_factory=dict,
        description="Maps component name to its contribution delta vs. baseline.",
    )
