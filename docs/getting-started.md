# Getting Started with aumai-ablation

This guide walks you from a fresh install to a complete, analyzed ablation study in about
15 minutes.

---

## Prerequisites

- Python 3.11 or later
- pip or a compatible package manager (uv, pipx, poetry)
- Basic familiarity with Python and the command line
- An evaluation harness that can produce numeric metrics for different component configurations
  (you supply this; aumai-ablation handles the bookkeeping)

---

## Installation

### Standard install

```bash
pip install aumai-ablation
```

### With YAML support (recommended for the CLI)

```bash
pip install aumai-ablation pyyaml
```

### Development install (editable)

```bash
git clone https://github.com/aumai/aumai-ablation
cd aumai-ablation
pip install -e ".[dev]"
```

### Verify your installation

```bash
aumai-ablation --version
python -c "import aumai_ablation; print(aumai_ablation.__version__)"
```

---

## Step-by-step tutorial

This tutorial walks through a realistic scenario: you have an agent pipeline with three
components and want to know which ones are worth keeping.

### Step 1: Identify your components

Write down every component in your pipeline. For this tutorial we use:

- `retriever` — fetches relevant documents from a vector store
- `reranker` — reorders retrieved documents by predicted relevance
- `query_rewriter` — rephrases the user query before retrieval

### Step 2: Create a components YAML file

```yaml
# components.yaml
- name: retriever
  enabled: true
  config:
    top_k: 5
    model: text-embedding-3-small

- name: reranker
  enabled: true
  config:
    model: cross-encoder/ms-marco-MiniLM-L-6-v2

- name: query_rewriter
  enabled: true
```

### Step 3: Generate the study plan

```bash
aumai-ablation configure \
  --components components.yaml \
  --metrics accuracy,f1,latency_ms \
  --output study.json
```

This writes `study.json` containing:
- The study configuration
- 4 run templates (1 baseline + 1 per component), each with an empty `metrics` dict
- Instructions for the next step

### Step 4: Run your evaluations

Open `study.json` and find the `"runs"` array. For each run, the `"disabled_component"` field
tells you which component to disable (or `null` for the baseline). Run your evaluation
pipeline in that configuration and fill in the `"metrics"` dict.

For example, your `study.json` might look like this after you fill it in:

```json
{
  "config": { ... },
  "runs": [
    {
      "run_id": "baseline-a1b2c3d4",
      "disabled_component": null,
      "components": [...],
      "metrics": {"accuracy": 0.87, "f1": 0.84, "latency_ms": 320}
    },
    {
      "run_id": "ablate-retriever-e5f6a7b8",
      "disabled_component": "retriever",
      "components": [...],
      "metrics": {"accuracy": 0.71, "f1": 0.68, "latency_ms": 210}
    },
    {
      "run_id": "ablate-reranker-c9d0e1f2",
      "disabled_component": "reranker",
      "components": [...],
      "metrics": {"accuracy": 0.84, "f1": 0.81, "latency_ms": 290}
    },
    {
      "run_id": "ablate-query_rewriter-a3b4c5d6",
      "disabled_component": "query_rewriter",
      "components": [...],
      "metrics": {"accuracy": 0.86, "f1": 0.83, "latency_ms": 305}
    }
  ]
}
```

### Step 5: Analyze the results

```bash
aumai-ablation analyze --results study.json
```

Output:

```json
{
  "component_importance": {
    "retriever": 0.1533,
    "reranker": 0.04,
    "query_rewriter": 0.0167
  },
  "ranking": [
    {"component": "retriever", "importance": 0.1533},
    {"component": "reranker", "importance": 0.04},
    {"component": "query_rewriter", "importance": 0.0167}
  ]
}
```

The retriever contributes the most. The query_rewriter contributes very little — you might
consider removing it to reduce latency.

### Step 6 (optional): Do the same in Python

```python
from aumai_ablation import AblationStudy, Component, AblationResult

study = AblationStudy()
config = study.configure(
    components=[
        Component(name="retriever", config={"top_k": 5}),
        Component(name="reranker"),
        Component(name="query_rewriter"),
    ],
    metrics=["accuracy", "f1", "latency_ms"],
)
runs = study.generate_runs(config)

# ... fill runs[i].metrics from your evaluation harness ...

result = AblationResult(config=config, runs=runs)
ranking = study.rank_components(result)
for name, score in ranking:
    print(f"{name}: {score:+.4f}")
```

---

## Common patterns and recipes

### Pattern 1: Tracking latency separately from accuracy

If you want to see which components are expensive in latency terms, track latency as a metric
but be aware that higher latency is worse. The importance score is `baseline - ablated`, so a
component that adds latency will produce a *negative* importance score for the latency metric
when averaged with accuracy metrics. Consider running separate studies for quality metrics and
cost metrics rather than mixing them.

```python
# Separate quality and cost studies
study = AblationStudy()

quality_config = study.configure(components=components, metrics=["accuracy", "f1"])
cost_config = study.configure(components=components, metrics=["latency_ms", "token_cost"])
```

### Pattern 2: Batch processing from JSONL

If your evaluation harness outputs one JSON object per line, use JSONL format directly:

```jsonl
{"run_id": "baseline-a1b2c3d4", "disabled_component": null, "components": [...], "metrics": {"accuracy": 0.87}}
{"run_id": "ablate-retriever-e5f6a7b8", "disabled_component": "retriever", "components": [...], "metrics": {"accuracy": 0.71}}
```

```bash
aumai-ablation analyze --results runs.jsonl --config ablation_config.json
```

### Pattern 3: Variance reduction with repetitions

For stochastic evaluations (LLM-based metrics, sampling-dependent pipelines), run each
configuration multiple times and average the results before computing importance.

```python
from aumai_ablation import AblationStudy, Component

components = [Component(name="retriever"), Component(name="reranker")]
study = AblationStudy()
config = study.configure(components=components, metrics=["accuracy"])
config.repetitions = 5  # Record that each run should be repeated 5 times

runs = study.generate_runs(config)
# Your harness should run each configuration 5 times and store the average in run.metrics
```

### Pattern 4: Detecting and removing harmful components

A negative importance score is a signal that a component is actively degrading performance.

```python
from aumai_ablation import AblationStudy, Component, AblationResult

components = [
    Component(name="experimental_filter"),
    Component(name="core_pipeline"),
]
study = AblationStudy()
config = study.configure(components=components, metrics=["accuracy"])
runs = study.generate_runs(config)

runs[0].metrics = {"accuracy": 0.72}  # baseline
runs[1].metrics = {"accuracy": 0.80}  # filter removed: accuracy improved!
runs[2].metrics = {"accuracy": 0.45}  # core removed: catastrophic

result = AblationResult(config=config, runs=runs)
importance = study.compute_importance(result)

for name, score in importance.items():
    if score < 0:
        print(f"WARNING: {name} has negative importance ({score:+.4f}) — consider removing it")
```

### Pattern 5: Saving importance scores for tracking over time

```python
import json
from datetime import datetime, timezone
from aumai_ablation import AblationStudy, AblationResult

# ... run your study ...

result = AblationResult(config=config, runs=runs)
study = AblationStudy()
study.rank_components(result)

record = {
    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    "component_importance": result.component_importance,
}

with open("importance_history.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
```

---

## Troubleshooting FAQ

**Q: `aumai-ablation configure` fails with "Error: PyYAML is required"**

Install PyYAML: `pip install pyyaml`. The YAML parser is an optional dependency kept separate
to minimise the base install footprint.

**Q: `compute_importance` returns an empty dict**

The baseline run must have at least one metric populated. Check that `runs[0].metrics` (the
run where `disabled_component is None`) is not empty before calling `compute_importance` or
`rank_components`.

**Q: All importance scores are 0.0**

This happens when an ablation run has an empty `metrics` dict. The formula treats missing
metrics as 0.0 effective average. Make sure every run in your `AblationResult.runs` list has
its `metrics` dict filled in.

**Q: I have more than one baseline run in my results file**

`compute_importance` uses the first run with `disabled_component == None` as the baseline.
If you have multiple baseline runs (e.g., from different repetitions), average their metrics
into a single dict before constructing the `AblationResult`.

**Q: The CLI `analyze` command says "JSON file must contain 'config' and 'runs' keys"**

If you're passing a plain JSON file (not the configure output), either add a `config` key or
use the `--config` flag to point to your config file separately, or use JSONL format.

**Q: I want to ablate two components simultaneously**

The current design ablates exactly one component per run. For multi-component ablation
(interaction effects), construct `AblationRun` objects manually with multiple components
set to `enabled=False`, then pass them directly to `AblationResult`.

---

## Next steps

- Read the [API Reference](api-reference.md) for complete class and method documentation
- Explore the [quickstart example](../examples/quickstart.py)
- See the [README](../README.md) for integration patterns with other AumAI tools
- Join the [AumAI Discord](https://discord.gg/aumai)
