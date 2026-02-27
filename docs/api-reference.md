# API Reference — aumai-ablation

Complete reference for all public classes, methods, and Pydantic models in `aumai-ablation`.

---

## Module: `aumai_ablation.core`

The core computation engine. Import directly or via the package root.

```python
from aumai_ablation import AblationStudy
# or
from aumai_ablation.core import AblationStudy
```

---

### `class AblationStudy`

Ablation study engine for agent component evaluation.

Generates a structured set of runs (one baseline plus one per enabled component) and computes
importance scores from observed metric deltas. This class is stateless — all data lives in the
Pydantic models it operates on.

**Example:**

```python
from aumai_ablation import AblationStudy, Component, AblationResult

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
```

---

#### `AblationStudy.configure`

```python
def configure(
    self,
    components: list[Component],
    metrics: list[str],
) -> AblationConfig
```

Build an `AblationConfig` from a component list and metric names.

**Parameters:**

| Parameter    | Type            | Description                                                   |
|--------------|-----------------|---------------------------------------------------------------|
| `components` | list[Component] | All components in the system. All should have `enabled=True`. |
| `metrics`    | list[str]       | Names of evaluation metrics to record per run.                |

**Returns:** A validated `AblationConfig` instance.

**Example:**

```python
config = study.configure(
    components=[
        Component(name="retriever"),
        Component(name="reranker"),
    ],
    metrics=["accuracy", "f1"],
)
print(config.repetitions)  # 1 (default)
config.repetitions = 3     # update in place; Pydantic validates the new value
```

---

#### `AblationStudy.generate_runs`

```python
def generate_runs(
    self,
    config: AblationConfig,
) -> list[AblationRun]
```

Generate the complete set of ablation runs from a configuration.

Produces one baseline run (all components at their configured `enabled` state) plus one
ablation run per currently-enabled component (that component flipped to `enabled=False`,
all others unchanged).

Components that have `enabled=False` in `config.base_components` are skipped — they are
already ablated in the baseline and generating an additional run for them is redundant.

**Parameters:**

| Parameter | Type           | Description                          |
|-----------|----------------|--------------------------------------|
| `config`  | AblationConfig | The study configuration to expand.   |

**Returns:** A `list[AblationRun]` with empty `metrics` dicts, ready for evaluation.

**Run ID format:**
- Baseline: `baseline-<8 hex chars>`
- Ablation: `ablate-<component_name>-<8 hex chars>`

**Example:**

```python
components = [
    Component(name="a"),
    Component(name="b", enabled=False),  # already disabled — skipped
    Component(name="c"),
]
config = study.configure(components=components, metrics=["score"])
runs = study.generate_runs(config)
len(runs)  # 3: baseline + ablate-a + ablate-c
```

---

#### `AblationStudy.compute_importance`

```python
def compute_importance(
    self,
    result: AblationResult,
) -> dict[str, float]
```

Compute component importance as performance delta versus the baseline.

For each ablation run, computes the arithmetic mean of all metric values, then subtracts that
from the baseline mean. A positive delta means the component helps overall performance;
negative means it hurts.

```
importance(component) = mean(baseline.metrics.values()) - mean(ablated.metrics.values())
```

**Parameters:**

| Parameter | Type            | Description                                                            |
|-----------|-----------------|------------------------------------------------------------------------|
| `result`  | AblationResult  | A result object whose `runs` have populated `metrics` dicts.           |

**Returns:** `dict[str, float]` mapping component name to importance score, rounded to 6
decimal places. Returns an empty dict if no baseline run is found or if the baseline has
no metrics.

**Edge cases:**
- Runs with an empty `metrics` dict receive an importance score of `0.0`.
- The baseline run (where `disabled_component is None`) is excluded from the output dict.

**Example:**

```python
result.runs[0].metrics = {"accuracy": 0.87}  # baseline
result.runs[1].metrics = {"accuracy": 0.71}  # retriever ablated

importance = study.compute_importance(result)
print(importance)  # {"retriever": 0.16}
```

---

#### `AblationStudy.rank_components`

```python
def rank_components(
    self,
    result: AblationResult,
) -> list[tuple[str, float]]
```

Rank components from most to least important.

Calls `compute_importance` internally, stores the result in `result.component_importance`
as a side effect, and returns a sorted list.

**Parameters:**

| Parameter | Type           | Description             |
|-----------|----------------|-------------------------|
| `result`  | AblationResult | A completed study result. |

**Returns:** `list[tuple[str, float]]` sorted descending by importance score.

**Side effects:** Mutates `result.component_importance` with the computed scores.

**Example:**

```python
ranking = study.rank_components(result)
for name, score in ranking:
    print(f"{name}: {score:+.6f}")
```

---

## Module: `aumai_ablation.models`

All Pydantic v2 data models. Import via the package root or the module directly.

```python
from aumai_ablation import Component, AblationConfig, AblationRun, AblationResult
# or
from aumai_ablation.models import Component, AblationConfig, AblationRun, AblationResult
```

All models share these Pydantic settings:
- `str_strip_whitespace=True` — leading/trailing whitespace stripped from strings automatically
- `validate_assignment=True` — field validation re-runs on every attribute assignment

---

### `class Component`

A named, toggle-able component in an agent or ML system.

```python
class Component(BaseModel):
    name: str          # Field(min_length=1, max_length=128)
    enabled: bool      # Field(default=True)
    config: dict[str, Any]  # Field(default_factory=dict)
```

**Fields:**

| Field     | Type              | Required | Default | Constraints        | Description                                         |
|-----------|-------------------|----------|---------|--------------------|-----------------------------------------------------|
| `name`    | str               | yes      | —       | 1–128 characters   | Unique component name; used as the ablation run key |
| `enabled` | bool              | no       | `True`  | —                  | Whether this component participates in the baseline |
| `config`  | dict[str, Any]    | no       | `{}`    | —                  | Arbitrary component configuration payload           |

**Examples:**

```python
# Minimal
c = Component(name="retriever")

# With config
c = Component(name="retriever", config={"top_k": 5, "model": "text-embedding-3-small"})

# Pre-disabled (will be skipped in generate_runs)
c = Component(name="legacy_cache", enabled=False)

# Serialise
c.model_dump()
# {"name": "retriever", "enabled": True, "config": {"top_k": 5, ...}}

# Validate from dict
c = Component.model_validate({"name": "reranker"})
```

---

### `class AblationConfig`

Configuration for an ablation study.

```python
class AblationConfig(BaseModel):
    base_components: list[Component]   # Field(min_length=1)
    metrics_to_track: list[str]        # Field(min_length=1)
    repetitions: int                   # Field(default=1, ge=1)
```

**Fields:**

| Field              | Type            | Required | Default | Constraints   | Description                                          |
|--------------------|-----------------|----------|---------|---------------|------------------------------------------------------|
| `base_components`  | list[Component] | yes      | —       | min 1 item    | The full component set for the baseline run          |
| `metrics_to_track` | list[str]       | yes      | —       | min 1 item    | Names of metrics that runs will record               |
| `repetitions`      | int             | no       | `1`     | >= 1          | Number of repetitions per configuration for variance |

**Examples:**

```python
config = AblationConfig(
    base_components=[Component(name="retriever"), Component(name="reranker")],
    metrics_to_track=["accuracy", "latency_ms"],
    repetitions=3,
)

# Serialise / deserialise
data = config.model_dump()
config2 = AblationConfig.model_validate(data)

# Update repetitions in place (validated)
config.repetitions = 5
```

---

### `class AblationRun`

A single ablation run — either the baseline or a configuration with one component disabled.

```python
class AblationRun(BaseModel):
    run_id: str                          # Field(min_length=1)
    disabled_component: str | None       # Field(default=None)
    components: list[Component]          # Field(default_factory=list)
    metrics: dict[str, float]            # Field(default_factory=dict)
```

**Fields:**

| Field                | Type               | Required | Default  | Description                                                        |
|----------------------|--------------------|----------|----------|--------------------------------------------------------------------|
| `run_id`             | str                | yes      | —        | Unique identifier for this run; generated by `generate_runs`       |
| `disabled_component` | str or None        | no       | `None`   | Name of the component disabled for this run; `None` = baseline     |
| `components`         | list[Component]    | no       | `[]`     | Snapshot of the component list for this run                        |
| `metrics`            | dict[str, float]   | no       | `{}`     | Metric name to value; you fill this in after running evaluation    |

**Examples:**

```python
# Baseline run (auto-generated by generate_runs)
run = AblationRun(
    run_id="baseline-a1b2c3d4",
    disabled_component=None,
    components=[Component(name="retriever"), Component(name="reranker")],
)

# Fill in metrics after evaluation
run.metrics = {"accuracy": 0.87, "latency_ms": 320.0}

# Ablation run
run = AblationRun(
    run_id="ablate-retriever-e5f6a7b8",
    disabled_component="retriever",
    components=[
        Component(name="retriever", enabled=False),
        Component(name="reranker", enabled=True),
    ],
    metrics={"accuracy": 0.71, "latency_ms": 210.0},
)

# Check if baseline
is_baseline = run.disabled_component is None
```

---

### `class AblationResult`

Aggregated results of a complete ablation study.

```python
class AblationResult(BaseModel):
    config: AblationConfig
    runs: list[AblationRun]              # Field(default_factory=list)
    component_importance: dict[str, float]  # Field(default_factory=dict)
```

**Fields:**

| Field                  | Type                 | Required | Default | Description                                               |
|------------------------|----------------------|----------|---------|-----------------------------------------------------------|
| `config`               | AblationConfig       | yes      | —       | The configuration that generated this study               |
| `runs`                 | list[AblationRun]    | no       | `[]`    | All runs, including the baseline and ablation runs        |
| `component_importance` | dict[str, float]     | no       | `{}`    | Populated by `rank_components`; maps name to delta score  |

The `component_importance` field is set as a side effect of calling `study.rank_components(result)`.

**Examples:**

```python
result = AblationResult(config=config, runs=runs)

# Analyze
study = AblationStudy()
ranking = study.rank_components(result)

# Access stored importance after ranking
print(result.component_importance)
# {"retriever": 0.16, "reranker": 0.03}

# Serialise full result
import json
print(json.dumps(result.model_dump(), indent=2))
```

---

## Module: `aumai_ablation.cli`

The Click-based command-line interface. Exposed as the `aumai-ablation` executable.

```
aumai-ablation --help
aumai-ablation configure --help
aumai-ablation analyze --help
```

The `cli` object is a `click.Group`. You can import it and invoke it programmatically if needed:

```python
from aumai_ablation.cli import cli
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(cli, ["configure", "--help"])
print(result.output)
```

---

## Public exports (`aumai_ablation.__init__`)

The following names are available directly from the package root:

```python
from aumai_ablation import (
    Component,        # models.Component
    AblationConfig,   # models.AblationConfig
    AblationRun,      # models.AblationRun
    AblationResult,   # models.AblationResult
    AblationStudy,    # core.AblationStudy
)

print(aumai_ablation.__version__)  # "1.0.0"
```

---

## Error handling

All Pydantic models raise `pydantic.ValidationError` on invalid input:

```python
from pydantic import ValidationError
from aumai_ablation import Component

try:
    c = Component(name="")  # empty name violates min_length=1
except ValidationError as exc:
    print(exc)
```

The CLI exits with code `1` and prints to stderr on invalid input, using `click.echo(..., err=True)`.
