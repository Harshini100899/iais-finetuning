# Text2Cypher: fine-tuning SmolLM2-135M

Fine-tune [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) with LoRA to turn a Neo4j graph schema and a natural language question into a Cypher query. Dataset: [`RomanTeucher/text2cypher-curated`](https://huggingface.co/datasets/RomanTeucher/text2cypher-curated). Trains and runs on CPU.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). Works the same on Windows, macOS, and Linux.

```
uv sync
```

This creates a virtual environment and installs the runtime and dev dependencies. Experiment tracking is an optional extra (see below). Run the commands below from the repo root.

## Reproduce

Training and evaluation are driven by the YAML files in `configs/`. Edit a value, then run the script. There are no command line flags.

Train:

```
uv run python train.py
```

Reads `configs/train.yaml` and writes the LoRA checkpoint (`best/` and `final/`) plus a loss history to the `checkpoint_dir` set there.

Evaluate (set `checkpoint` and `prompt_variant` in `configs/eval.yaml` first):

```
uv run python evaluate.py
```

Writes per-sample and summary JSON to `results/`.

Run the tests:

```
uv run pytest
```

To switch prompt setups, set `prompt_variant` to `default` or `fewshot` in both configs and point `eval.yaml` at a checkpoint trained with the same value.

## Experiment tracking (optional)

Training and evaluation can log parameters, metrics, and artifacts to MLflow, but it is optional. The configs ship with `use_mlflow: true`; if MLflow is not installed the run prints a note and continues without tracking.

To enable it, install the extra:

```
uv sync --extra tracking
```

Runs log to a local SQLite store (`mlflow.db`) by default. View them with:

```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To turn tracking off, set `use_mlflow: false` in `configs/train.yaml` and `configs/eval.yaml`. To use a tracking server, set `mlflow_tracking_uri` to its URL; if the server is unreachable the run falls back to the local SQLite store.

## Design decisions and limitations

- LoRA instead of full fine-tuning. With 1,000 training rows, full fine-tuning would memorise. The low-rank update keeps capacity in check. Settings: r=16, alpha=32, dropout=0.05 on all seven projection layers.
- Loss is computed on the answer only. Prompts are built with the chat template and the prompt tokens are masked, so gradients flow only through the Cypher tokens.
- Two prompt setups. `default` is a one line instruction. `fewshot` adds Cypher rules and four worked examples taken from the train split only. Few-shot scores a little higher.
- `cyspider-*` rows are dropped. They are SQL-style schemas repurposed from a text-to-SQL set and do not fit the Cypher task.
- Metrics are text and structure only: exact match (alias-normalised), component F1, syntactic validity, and schema grounding. There is no live Neo4j, so nothing is execution-checked. A query can read as valid and still return the wrong rows.
- It is a 135M model. It gets many queries wrong, does poorly on the synthetic-source styles, and fails on harder patterns like the same label in two roles, `WITH` chaining, `CASE`, and `UNWIND`. The value here is the pipeline and the evaluation, not the accuracy.

Trained adapters and the full result tables are on the Hugging Face model card.
