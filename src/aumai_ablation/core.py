"""Core logic for aumai-ablation.

Implements ablation study configuration, run generation, component importance
computation, and ranking.  All logic is heuristic/structural — the actual
metric values are supplied externally (e.g. by running a real evaluation).
"""

from __future__ import annotations

import copy
import uuid

from aumai_ablation.models import (
    AblationConfig,
    AblationResult,
    AblationRun,
    Component,
)

__all__ = ["AblationStudy"]

_BASELINE_RUN_ID_PREFIX = "baseline"


class AblationStudy:
    """Ablation study engine for agent component evaluation.

    Generates a structured set of runs (one per component plus a baseline),
    and computes importance scores from observed metric deltas.

    Example::

        study = AblationStudy()
        config = study.configure(
            components=[Component(name="retriever"), Component(name="reranker")],
            metrics=["accuracy", "latency_ms"],
        )
        runs = study.generate_runs(config)
        # ... fill in run.metrics from real evaluations ...
        result = AblationResult(config=config, runs=runs)
        importance = study.compute_importance(result)
        ranking = study.rank_components(result)
    """

    def configure(self, components: list[Component], metrics: list[str]) -> AblationConfig:
        """Build an ``AblationConfig`` from a component list and metric names.

        Args:
            components: All components in the system (should all be enabled).
            metrics: Names of evaluation metrics to track.

        Returns:
            A validated ``AblationConfig``.
        """
        return AblationConfig(
            base_components=components,
            metrics_to_track=metrics,
        )

    def generate_runs(self, config: AblationConfig) -> list[AblationRun]:
        """Generate the full set of ablation runs.

        Produces one baseline run (all components enabled) plus one run per
        enabled component (that component disabled, all others enabled).

        Args:
            config: The ablation study configuration.

        Returns:
            A list of ``AblationRun`` objects with empty ``metrics`` dicts
            ready to be filled in by external evaluation code.
        """
        runs: list[AblationRun] = []

        # Baseline — all components enabled
        baseline_components = [copy.deepcopy(c) for c in config.base_components]
        baseline_run = AblationRun(
            run_id=f"{_BASELINE_RUN_ID_PREFIX}-{str(uuid.uuid4())[:8]}",
            disabled_component=None,
            components=baseline_components,
        )
        runs.append(baseline_run)

        # One run per enabled component
        for component in config.base_components:
            if not component.enabled:
                continue  # Skip already-disabled components

            ablated_components: list[Component] = []
            for comp in config.base_components:
                ablated = copy.deepcopy(comp)
                if ablated.name == component.name:
                    ablated.enabled = False
                ablated_components.append(ablated)

            run = AblationRun(
                run_id=f"ablate-{component.name}-{str(uuid.uuid4())[:8]}",
                disabled_component=component.name,
                components=ablated_components,
            )
            runs.append(run)

        return runs

    def compute_importance(self, result: AblationResult) -> dict[str, float]:
        """Compute component importance as performance delta vs. the baseline.

        For each ablation run (where one component is disabled), the average
        metric delta compared to the baseline is computed.  A positive delta
        means the component *helps* overall performance; negative means it
        *hurts*.  Importance = baseline_avg - ablated_avg.

        Args:
            result: A completed ``AblationResult`` whose ``runs`` have
                    populated ``metrics`` dicts.

        Returns:
            A dict mapping component name to its importance score.
        """
        # Find the baseline run
        baseline_run = next(
            (r for r in result.runs if r.disabled_component is None), None
        )
        if baseline_run is None or not baseline_run.metrics:
            return {}

        # Average of all metrics in the baseline
        baseline_avg = (
            sum(baseline_run.metrics.values()) / len(baseline_run.metrics)
            if baseline_run.metrics
            else 0.0
        )

        importance: dict[str, float] = {}
        for run in result.runs:
            if run.disabled_component is None:
                continue  # skip baseline
            if not run.metrics:
                importance[run.disabled_component] = 0.0
                continue

            ablated_avg = sum(run.metrics.values()) / len(run.metrics)
            # Higher importance = bigger drop when component is removed
            delta = baseline_avg - ablated_avg
            importance[run.disabled_component] = round(delta, 6)

        return importance

    def rank_components(self, result: AblationResult) -> list[tuple[str, float]]:
        """Rank components from most to least important.

        Populates ``result.component_importance`` as a side effect and returns
        a sorted list of ``(component_name, importance_score)`` tuples.

        Args:
            result: A completed ``AblationResult``.

        Returns:
            Descending-sorted list of ``(component_name, importance_score)``.
        """
        importance = self.compute_importance(result)
        result.component_importance = importance
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
