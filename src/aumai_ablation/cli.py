"""CLI entry point for aumai-ablation.

Commands:
  configure   Set up an ablation study from a components YAML file.
  analyze     Analyze ablation results from a JSONL file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from aumai_ablation.core import AblationStudy
from aumai_ablation.models import AblationConfig, AblationResult, AblationRun, Component

__all__ = ["cli"]


def _load_yaml_components(path: str) -> list[Component]:
    """Load components from a YAML file.

    Expected YAML format::

        - name: retriever
          enabled: true
          config:
            top_k: 5
        - name: reranker
          enabled: true
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        click.echo("Error: PyYAML is required. Install with: pip install pyyaml", err=True)
        sys.exit(1)

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        click.echo("Error: components YAML must be a list of component objects.", err=True)
        sys.exit(1)
    return [Component.model_validate(item) for item in raw]


@click.group()
@click.version_option()
def cli() -> None:
    """AumAI Ablation — automated ablation studies for agent components."""


@cli.command("configure")
@click.option(
    "--components",
    "components_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a YAML file listing components.",
)
@click.option(
    "--metrics",
    "metrics_str",
    required=True,
    help="Comma-separated list of metric names, e.g. accuracy,latency_ms.",
)
@click.option(
    "--repetitions",
    default=1,
    type=int,
    show_default=True,
    help="Number of repetitions per ablation run.",
)
@click.option(
    "--output",
    "output_path",
    default="ablation_config.json",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to write the config JSON.",
)
def configure_command(
    components_path: str,
    metrics_str: str,
    repetitions: int,
    output_path: str,
) -> None:
    """Configure an ablation study and generate run templates.

    Example:

    \b
        aumai-ablation configure --components comp.yaml --metrics accuracy,latency
    """
    components = _load_yaml_components(components_path)
    metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]

    if not metrics:
        click.echo("Error: --metrics must contain at least one metric name.", err=True)
        sys.exit(1)

    study = AblationStudy()
    config = study.configure(components=components, metrics=metrics)
    config.repetitions = repetitions

    runs = study.generate_runs(config)

    output = {
        "config": config.model_dump(),
        "runs": [r.model_dump() for r in runs],
        "instructions": (
            "Fill in 'metrics' dict for each run, then use 'analyze' to compute importance."
        ),
    }

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    click.echo(
        f"Ablation config with {len(runs)} runs written to {output_path}"
    )


@cli.command("analyze")
@click.option(
    "--results",
    "results_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a JSONL file (one AblationRun JSON per line) or a JSON file.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to ablation_config.json (if separate from results).",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Optional path to write JSON analysis output.",
)
def analyze_command(
    results_path: str,
    config_path: str | None,
    output_path: str | None,
) -> None:
    """Analyze ablation run results and print component importance ranking.

    The results file may be:

    \b
      - JSONL: one AblationRun JSON object per line
      - JSON: the full output from the 'configure' command

    Example:

    \b
        aumai-ablation analyze --results results.jsonl
    """
    raw_text = Path(results_path).read_text(encoding="utf-8").strip()

    # Try to parse as JSON first (configure output format)
    runs: list[AblationRun] = []
    config: AblationConfig | None = None

    parsed_as_json = False
    if raw_text.startswith("{"):
        try:
            data: Any = json.loads(raw_text)
            if "config" in data and "runs" in data:
                config = AblationConfig.model_validate(data["config"])
                runs = [AblationRun.model_validate(r) for r in data["runs"]]
                parsed_as_json = True
            else:
                click.echo("Error: JSON file must contain 'config' and 'runs' keys.", err=True)
                sys.exit(1)
        except json.JSONDecodeError:
            pass  # Fall through to JSONL parsing

    if not parsed_as_json:
        # JSONL format
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                line_data: Any = json.loads(line)
                runs.append(AblationRun.model_validate(line_data))

    if config is None and config_path is not None:
        config_data: Any = json.loads(Path(config_path).read_text(encoding="utf-8"))
        config = AblationConfig.model_validate(
            config_data.get("config", config_data)
        )

    if config is None:
        # Build a minimal config from the runs
        all_component_names: set[str] = set()
        for run in runs:
            for comp in run.components:
                all_component_names.add(comp.name)
        base_components = [Component(name=n) for n in sorted(all_component_names)]
        all_metrics: set[str] = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())
        config = AblationConfig(
            base_components=base_components,
            metrics_to_track=sorted(all_metrics),
        )

    result = AblationResult(config=config, runs=runs)
    study = AblationStudy()
    ranking = study.rank_components(result)

    output_data = {
        "component_importance": result.component_importance,
        "ranking": [{"component": name, "importance": score} for name, score in ranking],
    }

    json_output = json.dumps(output_data, indent=2)
    if output_path:
        Path(output_path).write_text(json_output, encoding="utf-8")
        click.echo(f"Analysis written to {output_path}")
    else:
        click.echo(json_output)


# Allow both `aumai-ablation` and legacy `main` entry point names
main = cli

if __name__ == "__main__":
    cli()
