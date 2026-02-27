"""Quickstart examples for aumai-ablation.

Demonstrates the full ablation study workflow:
  1. Basic study with three components and simulated metrics
  2. Detecting harmful components (negative importance scores)
  3. Loading results from a CLI-generated JSON file
  4. Variance reduction with repetitions
  5. Working directly with AblationRun objects

Run this file to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aumai_ablation import (
    AblationConfig,
    AblationResult,
    AblationRun,
    AblationStudy,
    Component,
)


# ---------------------------------------------------------------------------
# Demo 1: Basic three-component ablation study
# ---------------------------------------------------------------------------


def demo_basic_study() -> None:
    """Run a basic ablation study with three components and simulated metrics."""
    print("=" * 60)
    print("Demo 1: Basic ablation study")
    print("=" * 60)

    components = [
        Component(name="retriever", config={"top_k": 5}),
        Component(name="reranker"),
        Component(name="query_rewriter"),
    ]

    study = AblationStudy()
    config = study.configure(components=components, metrics=["accuracy", "latency_ms"])

    print(f"Configured study with {len(config.base_components)} components")
    print(f"Metrics to track: {config.metrics_to_track}")

    # Generate run plan
    runs = study.generate_runs(config)
    print(f"\nGenerated {len(runs)} runs:")
    for run in runs:
        label = f"(baseline)" if run.disabled_component is None else f"(ablate {run.disabled_component})"
        print(f"  {run.run_id}  {label}")

    # Simulate evaluation results — in practice these come from your harness
    simulated_metrics: list[dict[str, float]] = [
        {"accuracy": 0.87, "latency_ms": 320.0},  # baseline
        {"accuracy": 0.71, "latency_ms": 210.0},  # retriever removed
        {"accuracy": 0.84, "latency_ms": 290.0},  # reranker removed
        {"accuracy": 0.86, "latency_ms": 300.0},  # query_rewriter removed
    ]
    for run, metrics in zip(runs, simulated_metrics):
        run.metrics = metrics

    # Analyse
    result = AblationResult(config=config, runs=runs)
    ranking = study.rank_components(result)

    print("\nComponent importance ranking (higher = more important):")
    for rank, (name, score) in enumerate(ranking, start=1):
        bar = "#" * max(0, int(score * 100))
        print(f"  {rank}. {name:<20} {score:+.4f}  {bar}")


# ---------------------------------------------------------------------------
# Demo 2: Detecting harmful components
# ---------------------------------------------------------------------------


def demo_harmful_component() -> None:
    """Demonstrate how to detect a component that is hurting performance."""
    print("\n" + "=" * 60)
    print("Demo 2: Detecting harmful components")
    print("=" * 60)

    components = [
        Component(name="core_retriever"),
        Component(name="experimental_noise_filter"),
        Component(name="answer_generator"),
    ]

    study = AblationStudy()
    config = study.configure(components=components, metrics=["accuracy"])
    runs = study.generate_runs(config)

    # The noise_filter is actually hurting accuracy
    runs[0].metrics = {"accuracy": 0.72}  # baseline (includes the harmful filter)
    runs[1].metrics = {"accuracy": 0.45}  # core_retriever removed — catastrophic
    runs[2].metrics = {"accuracy": 0.81}  # noise_filter removed — accuracy improved!
    runs[3].metrics = {"accuracy": 0.70}  # answer_generator removed — small drop

    result = AblationResult(config=config, runs=runs)
    importance = study.compute_importance(result)

    print("\nImportance scores:")
    for name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        if score < 0:
            verdict = "HARMFUL — removing it improves performance"
        elif score < 0.02:
            verdict = "negligible contribution"
        else:
            verdict = "contributes positively"
        print(f"  {name:<30} {score:+.4f}  ({verdict})")


# ---------------------------------------------------------------------------
# Demo 3: Round-tripping through a JSON file (simulating the CLI workflow)
# ---------------------------------------------------------------------------


def demo_json_roundtrip() -> None:
    """Simulate the CLI configure + analyze workflow using Python."""
    print("\n" + "=" * 60)
    print("Demo 3: JSON round-trip (simulating CLI workflow)")
    print("=" * 60)

    components = [Component(name="embedder"), Component(name="ranker")]
    study = AblationStudy()
    config = study.configure(components=components, metrics=["score"])
    runs = study.generate_runs(config)

    # Simulate the configure command output
    output_data = {
        "config": config.model_dump(),
        "runs": [r.model_dump() for r in runs],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(output_data, tmp, indent=2)
        tmp_path = tmp.name

    print(f"Config written to temporary file: {tmp_path}")

    # Simulate filling in metrics (as a user would edit the JSON)
    loaded = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
    loaded["runs"][0]["metrics"] = {"score": 0.90}
    loaded["runs"][1]["metrics"] = {"score": 0.75}
    loaded["runs"][2]["metrics"] = {"score": 0.85}
    Path(tmp_path).write_text(json.dumps(loaded, indent=2), encoding="utf-8")

    # Load back and analyze
    data = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
    loaded_config = AblationConfig.model_validate(data["config"])
    loaded_runs = [AblationRun.model_validate(r) for r in data["runs"]]

    result = AblationResult(config=loaded_config, runs=loaded_runs)
    ranking = study.rank_components(result)

    print("Ranking from reloaded data:")
    for name, score in ranking:
        print(f"  {name}: {score:+.4f}")

    Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Demo 4: Variance reduction with repetitions metadata
# ---------------------------------------------------------------------------


def demo_repetitions() -> None:
    """Show how repetitions are recorded in the config for variance reduction."""
    print("\n" + "=" * 60)
    print("Demo 4: Repetitions metadata")
    print("=" * 60)

    components = [Component(name="sampler"), Component(name="scorer")]
    study = AblationStudy()
    config = study.configure(components=components, metrics=["quality"])
    config.repetitions = 5

    print(f"Repetitions configured: {config.repetitions}")
    print("(Your evaluation harness should run each configuration 5 times")
    print(" and store the average in run.metrics before calling analyze.)")

    runs = study.generate_runs(config)
    print(f"Run IDs generated ({len(runs)} runs):")
    for run in runs:
        print(f"  {run.run_id}")


# ---------------------------------------------------------------------------
# Demo 5: Manual AblationRun construction
# ---------------------------------------------------------------------------


def demo_manual_runs() -> None:
    """Build AblationRun objects manually for maximum control."""
    print("\n" + "=" * 60)
    print("Demo 5: Manual run construction")
    print("=" * 60)

    config = AblationConfig(
        base_components=[Component(name="a"), Component(name="b")],
        metrics_to_track=["f1"],
    )

    # Manually construct runs (e.g., for multi-component ablation)
    baseline = AblationRun(
        run_id="baseline-manual",
        disabled_component=None,
        components=[Component(name="a"), Component(name="b")],
        metrics={"f1": 0.88},
    )
    ablate_both = AblationRun(
        run_id="ablate-both",
        disabled_component="a",  # record which one was the primary ablation
        components=[
            Component(name="a", enabled=False),
            Component(name="b", enabled=False),
        ],
        metrics={"f1": 0.30},
    )

    result = AblationResult(config=config, runs=[baseline, ablate_both])
    study = AblationStudy()
    importance = study.compute_importance(result)
    print(f"Importance when both a and b are removed: {importance}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos."""
    demo_basic_study()
    demo_harmful_component()
    demo_json_roundtrip()
    demo_repetitions()
    demo_manual_runs()
    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
