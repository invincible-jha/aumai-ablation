"""Comprehensive tests for aumai_ablation core logic."""

from __future__ import annotations

import pytest

from aumai_ablation.core import AblationStudy
from aumai_ablation.models import AblationConfig, AblationResult, AblationRun, Component


# ---------------------------------------------------------------------------
# AblationStudy.configure tests
# ---------------------------------------------------------------------------


class TestAblationStudyConfigure:
    def test_configure_returns_ablation_config(self, two_components) -> None:
        study = AblationStudy()
        config = study.configure(components=two_components, metrics=["accuracy"])
        assert isinstance(config, AblationConfig)

    def test_configure_stores_components(self, two_components) -> None:
        study = AblationStudy()
        config = study.configure(components=two_components, metrics=["accuracy"])
        assert len(config.base_components) == 2

    def test_configure_stores_metrics(self, two_components) -> None:
        study = AblationStudy()
        config = study.configure(
            components=two_components, metrics=["accuracy", "latency_ms"]
        )
        assert "accuracy" in config.metrics_to_track
        assert "latency_ms" in config.metrics_to_track

    def test_configure_default_repetitions(self, two_components) -> None:
        study = AblationStudy()
        config = study.configure(components=two_components, metrics=["accuracy"])
        assert config.repetitions == 1


# ---------------------------------------------------------------------------
# AblationStudy.generate_runs tests
# ---------------------------------------------------------------------------


class TestAblationStudyGenerateRuns:
    def test_generate_runs_returns_list(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        assert isinstance(runs, list)

    def test_generate_runs_count(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        # 1 baseline + 1 per enabled component = 1 + 2 = 3
        assert len(runs) == 3

    def test_generate_runs_baseline_first(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        assert runs[0].disabled_component is None

    def test_generate_runs_baseline_all_enabled(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        baseline = runs[0]
        assert all(c.enabled for c in baseline.components)

    def test_generate_runs_each_ablation_disables_one(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        ablation_runs = [r for r in runs if r.disabled_component is not None]
        for run in ablation_runs:
            disabled = [c for c in run.components if not c.enabled]
            assert len(disabled) == 1
            assert disabled[0].name == run.disabled_component

    def test_generate_runs_ablation_others_remain_enabled(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        ablation_runs = [r for r in runs if r.disabled_component is not None]
        for run in ablation_runs:
            enabled = [c for c in run.components if c.enabled]
            assert len(enabled) == len(run.components) - 1

    def test_generate_runs_unique_run_ids(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        run_ids = [r.run_id for r in runs]
        assert len(run_ids) == len(set(run_ids))

    def test_generate_runs_three_components(self, three_components) -> None:
        study = AblationStudy()
        config = AblationConfig(
            base_components=three_components, metrics_to_track=["accuracy"]
        )
        runs = study.generate_runs(config)
        assert len(runs) == 4  # 1 baseline + 3

    def test_generate_runs_skips_disabled_components(self) -> None:
        components = [
            Component(name="active", enabled=True),
            Component(name="inactive", enabled=False),
        ]
        config = AblationConfig(base_components=components, metrics_to_track=["accuracy"])
        study = AblationStudy()
        runs = study.generate_runs(config)
        # Only 1 ablation run for the active component
        assert len(runs) == 2

    def test_generate_runs_components_are_deep_copies(self, basic_config) -> None:
        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        # Mutating a run's component should not affect another run
        runs[1].components[0].enabled = False
        assert runs[0].components[0].enabled is True


# ---------------------------------------------------------------------------
# AblationStudy.compute_importance tests
# ---------------------------------------------------------------------------


class TestAblationStudyComputeImportance:
    def test_compute_importance_returns_dict(self, completed_result) -> None:
        study = AblationStudy()
        importance = study.compute_importance(completed_result)
        assert isinstance(importance, dict)

    def test_compute_importance_has_all_components(self, completed_result) -> None:
        study = AblationStudy()
        importance = study.compute_importance(completed_result)
        assert "retriever" in importance
        assert "reranker" in importance

    def test_compute_importance_retriever_higher_than_reranker(
        self, completed_result
    ) -> None:
        study = AblationStudy()
        importance = study.compute_importance(completed_result)
        # Retriever: baseline_avg=95 vs ablated_avg=80, delta=15
        # Reranker: baseline_avg=95 vs ablated_avg=90, delta=5
        assert importance["retriever"] > importance["reranker"]

    def test_compute_importance_no_baseline_returns_empty(self, basic_config) -> None:
        runs = [
            AblationRun(
                run_id="ablate-retriever-abc",
                disabled_component="retriever",
                components=[],
                metrics={"accuracy": 0.8},
            )
        ]
        result = AblationResult(config=basic_config, runs=runs)
        study = AblationStudy()
        importance = study.compute_importance(result)
        assert importance == {}

    def test_compute_importance_empty_metrics_returns_zero(self, basic_config) -> None:
        from aumai_ablation.core import AblationStudy

        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        # Give baseline metrics but leave ablations empty
        runs[0].metrics = {"accuracy": 0.9}
        result = AblationResult(config=basic_config, runs=runs)
        importance = study.compute_importance(result)
        for component_name, score in importance.items():
            assert score == 0.0


# ---------------------------------------------------------------------------
# AblationStudy.rank_components tests
# ---------------------------------------------------------------------------


class TestAblationStudyRankComponents:
    def test_rank_components_returns_sorted_list(self, completed_result) -> None:
        study = AblationStudy()
        ranking = study.rank_components(completed_result)
        assert isinstance(ranking, list)
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_rank_components_includes_all_components(self, completed_result) -> None:
        study = AblationStudy()
        ranking = study.rank_components(completed_result)
        names = [name for name, _ in ranking]
        assert "retriever" in names
        assert "reranker" in names

    def test_rank_components_updates_result_importance(self, completed_result) -> None:
        study = AblationStudy()
        study.rank_components(completed_result)
        assert completed_result.component_importance != {}

    def test_rank_components_most_important_first(self, completed_result) -> None:
        study = AblationStudy()
        ranking = study.rank_components(completed_result)
        assert ranking[0][0] == "retriever"

    def test_rank_empty_metrics_result(self, basic_config) -> None:
        from aumai_ablation.core import AblationStudy

        study = AblationStudy()
        runs = study.generate_runs(basic_config)
        result = AblationResult(config=basic_config, runs=runs)
        ranking = study.rank_components(result)
        # No metrics → importance is empty → ranking is empty
        assert ranking == []
