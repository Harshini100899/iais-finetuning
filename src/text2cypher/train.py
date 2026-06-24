"""Core training procedure for Text2Cypher LoRA fine-tuning."""

from __future__ import annotations

# Allow running this file directly (IDE "Run" button or path) as well as via the
# repo-root wrapper or `python -m text2cypher.train`.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "text2cypher"

import json
import math
import os
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from .config import CONFIGS_DIR, MODEL_NAME, load_yaml, resolve_path
from .data import collate_fn, get_datasets, get_tokenizer
from .model import get_optimizer_and_scheduler, get_peft_lora_model
from .utils import set_seeds, tracking


def compute_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    grad_accum: int,
    device: torch.device,
    train: bool = True,
) -> float:
    """Run one epoch of training or evaluation; return average loss."""
    model.train(train)
    total_loss = 0.0
    total_steps = 0

    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        if train:
            loss = compute_loss(model, batch)
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                loss = compute_loss(model, batch)

        total_loss += loss.item()
        total_steps += 1

        if train and (step + 1) % max(1, len(loader) // 5) == 0:
            print(
                f"    step {step + 1:>4d}/{len(loader)}  "
                f"loss={loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / total_steps if total_steps > 0 else float("inf")


def train(args: SimpleNamespace) -> None:
    num_threads = args.num_threads or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(num_threads)
    print(f"[config] torch threads: {num_threads}")

    set_seeds(args.seed)
    device = torch.device("cpu")
    checkpoint_dir = resolve_path(args.checkpoint_dir)

    config_summary = {
        "model": MODEL_NAME,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "physical_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lora_rank": args.rank,
        "lora_alpha": args.rank * 2,
        "lora_dropout": args.dropout,
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "num_threads": num_threads,
        "overfit_check": args.overfit_check,
        "checkpoint_dir": str(checkpoint_dir),
        "prompt_variant": args.prompt_variant,
    }
    print("\n" + "=" * 60)
    print("TRAINING CONFIG")
    print("=" * 60)
    for k, v in config_summary.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60 + "\n")

    tracking.setup(
        args.mlflow_tracking_uri, args.mlflow_experiment, enabled=args.use_mlflow
    )

    run_name = (
        f"lora_rank{args.rank}_lr{args.lr}_epochs{args.epochs}_{int(time.time())}"
    )
    if args.overfit_check:
        run_name = f"overfit_{run_name}"

    with tracking.start_run(run_name=run_name) as run:
        run_id = run.run_id
        tracking.log_params(config_summary)

        tokenizer = get_tokenizer()
        overfit_n = 20 if args.overfit_check else None
        datasets, _ = get_datasets(
            tokenizer=tokenizer,
            max_length=args.max_length,
            overfit_n=overfit_n,
            verbose=True,
            prompt_variant=args.prompt_variant,
        )

        pad_id = tokenizer.pad_token_id

        def _collate(batch):
            return collate_fn(batch, pad_id)

        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        print("Loading base model …")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )
        base_model.to(device)

        model = get_peft_lora_model(base_model, rank=args.rank, dropout=args.dropout)
        model.print_trainable_parameters()

        total_update_steps = (
            math.ceil(len(train_loader) / args.grad_accum) * args.epochs
        )
        warmup_steps = max(1, int(0.05 * total_update_steps))

        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            total_update_steps=total_update_steps,
        )
        print(
            f"\nScheduler: warmup {warmup_steps} / {total_update_steps} "
            f"total update steps\n"
        )

        best_dir = checkpoint_dir / "best"
        final_dir = checkpoint_dir / "final"
        best_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        history: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            print(f"=== Epoch {epoch}/{args.epochs} {'=' * 51}")

            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                args.grad_accum,
                device,
                train=True,
            )
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                scheduler,
                args.grad_accum,
                device,
                train=False,
            )
            elapsed = time.time() - epoch_start

            print(
                f"  Epoch {epoch:>3d}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  ({elapsed:.0f}s)"
            )

            history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            tracking.log_metric("train_loss", train_loss, step=epoch)
            tracking.log_metric("val_loss", val_loss, step=epoch)
            tracking.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
                if run_id:
                    (best_dir / "mlflow_run_id.txt").write_text(run_id)
                print(
                    f"  [BEST] New best checkpoint saved to {best_dir}  "
                    f"(val_loss={best_val_loss:.4f})"
                )

        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        if run_id:
            (final_dir / "mlflow_run_id.txt").write_text(run_id)
        print(f"\nFinal checkpoint saved to {final_dir}")

        print("\n" + "=" * 60)
        print("LOSS CURVE SUMMARY")
        print("=" * 60)
        print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
        print(f"  {'-' * 6}  {'-' * 12}  {'-' * 10}")
        for h in history:
            marker = " <- best" if abs(h["val_loss"] - best_val_loss) < 1e-9 else ""
            print(
                f"  {h['epoch']:>6}  {h['train_loss']:>12.4f}  "
                f"{h['val_loss']:>10.4f}{marker}"
            )
        print("=" * 60)

        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({"config": config_summary, "history": history}, f, indent=2)
        print(f"\nTraining history saved to {history_path}")

        tracking.log_artifact(str(history_path))
        if best_dir.exists():
            tracking.log_artifacts(str(best_dir), artifact_path="best_checkpoint")


def main() -> None:
    config = load_yaml(CONFIGS_DIR / "train.yaml")
    train(SimpleNamespace(**config))


if __name__ == "__main__":
    main()
