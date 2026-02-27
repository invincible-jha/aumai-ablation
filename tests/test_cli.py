"""CLI tests for aumai-ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

# pyyaml is an optional dependency for the configure command
yaml = pytest.importorskip("yaml", reason="pyyaml is required for ablation CLI tests")

from aumai_ablation.cli import cli


SAMPLE_YAML_COMPONENTS = """\
- name: retriever
  enabled: true
  config:
    top_k: 5
- name: reranker
  enabled: true
"""


def _extract_json(text: str) -> dict:
    start = text.index("{")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("No JSON object found")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def components_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "components.yaml"
    path.write_text(SAMPLE_YAML_COMPONENTS, encoding="utf-8")
    return path


@pytest.fixture()
def config_json(tmp_path: Path, components_yaml: Path, runner: CliRunner) -> Path:
    """Create a configure output JSON file for use in analyze tests."""
    output_path = tmp_path / "ablation_config.json"
    result = runner.invoke(
        cli,
        [
            "configure",
            "--components", str(components_yaml),
            "--metrics", "accuracy,latency_ms",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    return output_path


class TestCLIGroup:
    def test_cli_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_cli_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestConfigureCommand:
    def test_configure_basic(
        self, runner: CliRunner, components_yaml: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "config.json"
        result = runner.invoke(
            cli,
            [
                "configure",
                "--components", str(components_yaml),
                "--metrics", "accuracy,latency_ms",
                "--output", str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()

    def test_configure_creates_runs(
        self, runner: CliRunner, components_yaml: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "config.json"
        runner.invoke(
            cli,
            [
                "configure",
                "--components", str(components_yaml),
                "--metrics", "accuracy",
                "--output", str(output_path),
            ],
        )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "config" in data
        assert "runs" in data
        # 1 baseline + 2 components = 3 runs
        assert len(data["runs"]) == 3

    def test_configure_baseline_run_present(
        self, runner: CliRunner, components_yaml: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "config.json"
        runner.invoke(
            cli,
            [
                "configure",
                "--components", str(components_yaml),
                "--metrics", "accuracy",
                "--output", str(output_path),
            ],
        )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        baseline_runs = [r for r in data["runs"] if r["disabled_component"] is None]
        assert len(baseline_runs) == 1

    def test_configure_repetitions_flag(
        self, runner: CliRunner, components_yaml: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "config.json"
        runner.invoke(
            cli,
            [
                "configure",
                "--components", str(components_yaml),
                "--metrics", "accuracy",
                "--repetitions", "3",
                "--output", str(output_path),
            ],
        )
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["config"]["repetitions"] == 3

    def test_configure_missing_components_fails(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "configure",
                "--components", str(tmp_path / "nonexistent.yaml"),
                "--metrics", "accuracy",
            ],
        )
        assert result.exit_code != 0

    def test_configure_empty_metrics_fails(
        self, runner: CliRunner, components_yaml: Path, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "config.json"
        result = runner.invoke(
            cli,
            [
                "configure",
                "--components", str(components_yaml),
                "--metrics", "   ",
                "--output", str(output_path),
            ],
        )
        assert result.exit_code != 0


class TestAnalyzeCommand:
    def test_analyze_json_format(
        self, runner: CliRunner, config_json: Path, tmp_path: Path
    ) -> None:
        """Fill in metrics in config JSON then analyze."""
        data = json.loads(config_json.read_text(encoding="utf-8"))
        for run in data["runs"]:
            if run["disabled_component"] is None:
                run["metrics"] = {"accuracy": 0.90, "latency_ms": 100.0}
            elif run["disabled_component"] == "retriever":
                run["metrics"] = {"accuracy": 0.70, "latency_ms": 90.0}
            else:
                run["metrics"] = {"accuracy": 0.85, "latency_ms": 95.0}
        config_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = runner.invoke(
            cli, ["analyze", "--results", str(config_json)]
        )
        assert result.exit_code == 0, result.output
        output = _extract_json(result.output)
        assert "ranking" in output
        assert "component_importance" in output

    def test_analyze_jsonl_format(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        runs = [
            {"run_id": "baseline-abc", "disabled_component": None, "components": [{"name": "ret", "enabled": True, "config": {}}], "metrics": {"accuracy": 0.9}},
            {"run_id": "ablate-ret-xyz", "disabled_component": "ret", "components": [{"name": "ret", "enabled": False, "config": {}}], "metrics": {"accuracy": 0.7}},
        ]
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in runs), encoding="utf-8"
        )
        result = runner.invoke(
            cli, ["analyze", "--results", str(jsonl_path)]
        )
        assert result.exit_code == 0, result.output
        output = _extract_json(result.output)
        assert "ranking" in output

    def test_analyze_writes_output_file(
        self, runner: CliRunner, config_json: Path, tmp_path: Path
    ) -> None:
        data = json.loads(config_json.read_text(encoding="utf-8"))
        for run in data["runs"]:
            run["metrics"] = {"accuracy": 0.9}
        config_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        output_path = tmp_path / "analysis.json"
        result = runner.invoke(
            cli,
            ["analyze", "--results", str(config_json), "--output", str(output_path)],
        )
        assert result.exit_code == 0, result.output
        assert output_path.exists()

    def test_analyze_missing_results_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["analyze", "--results", "/nonexistent/results.json"]
        )
        assert result.exit_code != 0
