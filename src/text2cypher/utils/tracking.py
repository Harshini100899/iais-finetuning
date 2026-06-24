"""Optional experiment tracking via MLflow.

When MLflow is installed and tracking is enabled, runs are logged. Otherwise
every call here is a no-op, so training and evaluation work without MLflow.
Install the optional dependency with: uv sync --extra tracking
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from ..config import resolve_tracking_uri

try:
    import mlflow

    _MLFLOW_INSTALLED = True
except ImportError:  # pragma: no cover - exercised only without the extra
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_INSTALLED = False

_active = False


def is_available() -> bool:
    """Return True if MLflow is importable."""
    return _MLFLOW_INSTALLED


def _server_reachable(uri: str) -> bool:
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=1.5) as conn:
            return conn.getcode() == 200
    except Exception:
        return False


def setup(tracking_uri: str, experiment: str, enabled: bool = True) -> bool:
    """Configure tracking and return True if it is active for this run.

    Relative ``sqlite:///`` URIs are anchored to the repo root. An HTTP URI is
    used if the server answers, otherwise it falls back to local SQLite.
    """
    global _active
    if not enabled:
        _active = False
        print("[tracking] disabled via config (use_mlflow: false).")
        return False
    if not _MLFLOW_INSTALLED:
        _active = False
        print(
            "[tracking] MLflow is not installed; running without tracking. "
            "Install it with: uv sync --extra tracking"
        )
        return False

    if tracking_uri.startswith("http"):
        if _server_reachable(tracking_uri):
            uri = tracking_uri
        else:
            uri = resolve_tracking_uri("sqlite:///mlflow.db")
            print(f"[tracking] server {tracking_uri} unreachable; using {uri}")
    else:
        uri = resolve_tracking_uri(tracking_uri)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    _active = True
    print(f"[tracking] MLflow active at {uri} (experiment: {experiment})")
    return True


class _Run:
    """Handle so callers can read run_id without importing MLflow themselves."""

    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id


@contextmanager
def start_run(run_name: str | None = None, run_id: str | None = None) -> Iterator[_Run]:
    """Start or resume a run. Yields a handle with run_id=None when inactive."""
    if not _active:
        yield _Run(None)
        return

    if run_id:
        try:
            run = mlflow.start_run(run_id=run_id)
            print(f"[tracking] resumed run {run_id}")
        except Exception as e:
            print(f"[tracking] could not resume {run_id}: {e}; starting a new run")
            run = mlflow.start_run(run_name=run_name)
    else:
        run = mlflow.start_run(run_name=run_name)

    try:
        yield _Run(run.info.run_id)
    finally:
        mlflow.end_run()


def log_params(params: Mapping[str, Any]) -> None:
    if _active:
        mlflow.log_params(dict(params))


def log_param(key: str, value: Any) -> None:
    if _active:
        mlflow.log_param(key, value)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    if _active:
        mlflow.log_metric(key, value, step=step)


def log_artifact(path: str, artifact_path: str | None = None) -> None:
    if _active:
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_artifacts(path: str, artifact_path: str | None = None) -> None:
    if _active:
        mlflow.log_artifacts(path, artifact_path=artifact_path)
