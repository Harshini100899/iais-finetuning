"""Generation + metrics evaluation + JSON output for Text2Cypher."""

from __future__ import annotations

# Allow running this file directly (IDE "Run" button or path) as well as via the
# repo-root wrapper or `python -m text2cypher.evaluate`.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "text2cypher"

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset

from .config import (
    CONFIGS_DIR,
    CYSPIDER_PREFIX,
    DATASET_NAME,
    MAX_LENGTH,
    load_yaml,
    resolve_path,
)
from .data import _normalize_schema, build_messages
from .metrics import (
    component_f1,
    exact_match_strict,
    normalized_exact_match,
    schema_grounding,
    schema_grounding_score,
    syntactic_validity,
)
from .model import load_model_and_tokenizer
from .utils import tracking


def generate_cypher(
    model,
    tokenizer,
    question: str,
    schema: str,
    device: torch.device,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    prompt_variant: str = "default",
) -> str:
    """Generate a Cypher query for one question + schema pair via chat template."""
    messages = build_messages(question, schema, prompt_variant)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


def evaluate(args: SimpleNamespace) -> None:
    num_threads = args.num_threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(num_threads)

    device = torch.device("cpu")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_base = args.checkpoint == "base"
    checkpoint = "base" if is_base else str(resolve_path(args.checkpoint))
    checkpoint_label = "base" if is_base else Path(checkpoint).name

    tracking.setup(
        args.mlflow_tracking_uri, args.mlflow_experiment, enabled=args.use_mlflow
    )

    run_id = None
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and (checkpoint_path / "mlflow_run_id.txt").exists():
        run_id = (checkpoint_path / "mlflow_run_id.txt").read_text().strip()

    eval_run_name = f"eval_{checkpoint_label}_{args.split}_{int(time.time())}"
    with tracking.start_run(run_name=eval_run_name, run_id=run_id):
        try:
            tracking.log_param("prompt_variant", args.prompt_variant)
        except Exception:
            # Resuming a run that already logged a different prompt_variant
            # value raises; not worth failing the eval over.
            pass
        model, tokenizer = load_model_and_tokenizer(checkpoint, device)

        print(f"\nLoading dataset split: {args.split}")
        raw = load_dataset(DATASET_NAME)

        split_data = [
            row
            for row in raw[args.split]
            if not row["data_source"].startswith(CYSPIDER_PREFIX)
        ]
        print(f"  Examples after cyspider filter: {len(split_data)}")

        if args.max_samples and len(split_data) > args.max_samples:
            split_data = split_data[: args.max_samples]
            print(f"  Truncated to {args.max_samples} samples (--max-samples).")

        lengths = [len(tokenizer.encode(row["cypher"])) for row in split_data]
        if lengths:
            lengths_sorted = sorted(lengths)
            median = lengths_sorted[len(lengths_sorted) // 2]
            p95 = lengths_sorted[int(0.95 * len(lengths_sorted))]
            print(
                f"  Gold Cypher token length - median: {median}, "
                f"p95: {p95}, max: {max(lengths)}"
            )

        per_sample_results: list[dict] = []

        print(
            f"\nGenerating ({len(split_data)} examples, beams={args.beams}, "
            f"max_new_tokens={args.max_new_tokens}) ...\n"
        )

        for i, row in enumerate(split_data):
            t0 = time.time()
            question = row["question"]
            schema = _normalize_schema(row["schema"], row.get("data_source", ""))
            gold_cypher = row["cypher"].strip()
            data_source = row.get("data_source", "unknown")
            instance_id = row.get("instance_id", "")
            db_alias = row.get("database_reference_alias", "")

            generated = generate_cypher(
                model,
                tokenizer,
                question,
                schema,
                device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beams,
                prompt_variant=args.prompt_variant,
            )

            em = normalized_exact_match(gold_cypher, generated)
            em_strict = exact_match_strict(gold_cypher, generated)
            cf = component_f1(gold_cypher, generated)
            sv = syntactic_validity(generated)
            grounded_props, pred_props = schema_grounding(generated, schema)
            grounding = schema_grounding_score(generated, schema)

            elapsed = time.time() - t0
            grounding_str = f"{grounding:.2f}" if grounding is not None else "n/a"
            print(
                f"  [{i + 1:>3d}/{len(split_data)}] EM={int(em)}  "
                f"F1={cf['f1']:.3f}  Valid={int(sv)}  "
                f"Grounded={grounding_str}  ({elapsed:.1f}s)"
            )
            if args.verbose:
                print(f"    Q : {question[:100]}")
                print(f"    G : {gold_cypher[:120]}")
                print(f"    P : {generated[:120]}")

            per_sample_results.append(
                {
                    "instance_id": instance_id,
                    "data_source": data_source,
                    "database_reference_alias": db_alias,
                    "question": question,
                    "normalized_schema": schema,
                    "gold_cypher": gold_cypher,
                    "generated_cypher": generated,
                    "exact_match": em,
                    "exact_match_strict": em_strict,
                    "component_precision": cf["precision"],
                    "component_recall": cf["recall"],
                    "component_f1": cf["f1"],
                    "component_f1_per_component": cf["per_component"],
                    "syntactic_valid": sv,
                    # ratio (or null if no properties referenced); raw counts
                    # below feed the honest micro-average over the test set.
                    "schema_grounding": grounding,
                    "schema_grounded_props": grounded_props,
                    "schema_pred_props": pred_props,
                }
            )

        def _aggregate(results: list[dict]) -> dict:
            n = len(results)
            if n == 0:
                return {}
            # Schema grounding is micro-averaged: total grounded properties over
            # total predicted properties across all samples. This avoids the trap
            # where a query that references no properties (empty/garbage output)
            # would otherwise count as "perfectly grounded".
            total_grounded = sum(r["schema_grounded_props"] for r in results)
            total_pred = sum(r["schema_pred_props"] for r in results)
            samples_with_props = sum(1 for r in results if r["schema_pred_props"] > 0)
            return {
                "n": n,
                "exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
                "exact_match_strict": round(
                    sum(r["exact_match_strict"] for r in results) / n, 4
                ),
                "component_precision": round(
                    sum(r["component_precision"] for r in results) / n, 4
                ),
                "component_recall": round(
                    sum(r["component_recall"] for r in results) / n, 4
                ),
                "component_f1": round(sum(r["component_f1"] for r in results) / n, 4),
                "syntactic_valid": round(
                    sum(r["syntactic_valid"] for r in results) / n, 4
                ),
                "schema_grounding": round(total_grounded / total_pred, 4)
                if total_pred
                else None,
                "schema_grounding_coverage": round(samples_with_props / n, 4),
            }

        overall = _aggregate(per_sample_results)

        by_source: dict[str, list[dict]] = defaultdict(list)
        for r in per_sample_results:
            by_source[r["data_source"]].append(r)
        per_source_agg = {src: _aggregate(rows) for src, rows in by_source.items()}

        aggregate_summary = {
            "checkpoint": checkpoint,
            "checkpoint_label": checkpoint_label,
            "split": args.split,
            "prompt_variant": args.prompt_variant,
            "overall": overall,
            "per_data_source": per_source_agg,
        }

        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY [{checkpoint_label.upper()}]  (split={args.split})")
        print(f"{'=' * 60}")
        print(f"  {'Examples evaluated':<24}: {overall['n']}")
        print(f"  {'Exact match (alias-norm)':<24}: {overall['exact_match']:.4f}")
        print(f"  {'Exact match (raw)':<24}: {overall['exact_match_strict']:.4f}")
        print(f"  {'Component F1':<24}: {overall['component_f1']:.4f}")
        print(f"    {'precision':<22}: {overall['component_precision']:.4f}")
        print(f"    {'recall':<22}: {overall['component_recall']:.4f}")
        print(f"  {'Syntactic valid':<24}: {overall['syntactic_valid']:.4f}")
        grounding_val = overall["schema_grounding"]
        grounding_disp = f"{grounding_val:.4f}" if grounding_val is not None else "n/a"
        print(
            f"  {'Schema grounding':<24}: {grounding_disp}  "
            f"(fraction of predicted properties that exist in the schema; "
            f"coverage={overall['schema_grounding_coverage']:.2f})"
        )
        print("\n  Breakdown by data_source:")
        for src, agg in sorted(per_source_agg.items()):
            print(
                f"    {src:40s}  n={agg['n']:>3d}  "
                f"EM={agg['exact_match']:.3f}  "
                f"F1={agg['component_f1']:.3f}  "
                f"Valid={agg['syntactic_valid']:.3f}"
            )
        print(f"{'=' * 60}\n")

        # Log only the headline scalars as MLflow metrics; the per-data-source
        # breakdown lives in the summary JSON artifact instead of as one-bar charts.
        prefix = f"{args.split}_"
        tracking.log_metric(f"{prefix}exact_match", overall["exact_match"])
        tracking.log_metric(
            f"{prefix}component_precision", overall["component_precision"]
        )
        tracking.log_metric(f"{prefix}component_recall", overall["component_recall"])
        tracking.log_metric(f"{prefix}component_f1", overall["component_f1"])
        tracking.log_metric(f"{prefix}syntactic_valid", overall["syntactic_valid"])
        if overall["schema_grounding"] is not None:
            tracking.log_metric(
                f"{prefix}schema_grounding", overall["schema_grounding"]
            )

        # prompt_variant is part of the filename so evaluating the same checkpoint
        # under different prompts (default vs fewshot) doesn't overwrite.
        tag = f"{checkpoint_label}_{args.prompt_variant}_{args.split}"
        per_sample_path = output_dir / f"per_sample_{tag}.json"
        with open(per_sample_path, "w", encoding="utf-8") as f:
            json.dump(per_sample_results, f, indent=2, ensure_ascii=False)
        print(f"Per-sample results  -> {per_sample_path}")
        tracking.log_artifact(str(per_sample_path), artifact_path="evaluation")

        summary_path = output_dir / f"summary_{tag}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)
        print(f"Aggregate summary   -> {summary_path}")
        tracking.log_artifact(str(summary_path), artifact_path="evaluation")

        if args.print_samples > 0:
            print(f"\n{'=' * 60}")
            print(f"QUALITATIVE SAMPLE REVIEW (first {args.print_samples} examples)")
            print(f"{'=' * 60}")
            for idx, r in enumerate(per_sample_results[: args.print_samples]):
                print(f"\n[{idx + 1}] instance_id  : {r['instance_id']}")
                print(f"    data_source  : {r['data_source']}")
                print(f"    question     : {r['question']}")
                print(f"    gold_cypher  : {r['gold_cypher']}")
                print(f"    generated    : {r['generated_cypher']}")
                print(f"    exact_match  : {r['exact_match']}")
                print(f"    component_f1 : {r['component_f1']}")
                print(f"    syntactic_ok : {r['syntactic_valid']}")


def main() -> None:
    config = load_yaml(CONFIGS_DIR / "eval.yaml")
    evaluate(SimpleNamespace(**config))


if __name__ == "__main__":
    main()
