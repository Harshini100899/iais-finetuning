"""Unit tests for helper and utility functions in utils/."""

from unittest.mock import MagicMock, patch

import pytest

from text2cypher.config import resolve_tracking_uri
from text2cypher.utils import tracking
from text2cypher.utils.reproducibility import set_seeds

mlflow_required = pytest.mark.skipif(
    not tracking.is_available(), reason="MLflow not installed (optional extra)"
)


@pytest.fixture(autouse=True)
def _reset_tracking():
    """Each test starts and ends with tracking inactive."""
    tracking._active = False
    yield
    tracking._active = False


def test_set_seeds():
    set_seeds(42)


def test_setup_disabled_is_noop():
    with patch.object(tracking, "mlflow") as mock_mlflow:
        active = tracking.setup("sqlite:///mlflow.db", "exp", enabled=False)
    assert active is False
    mock_mlflow.set_tracking_uri.assert_not_called()


def test_setup_without_mlflow_installed():
    with patch.object(tracking, "_MLFLOW_INSTALLED", False):
        active = tracking.setup("sqlite:///mlflow.db", "exp", enabled=True)
    assert active is False


def test_logging_when_inactive_is_noop():
    # No setup called, so tracking is inactive; these must not raise.
    with patch.object(tracking, "mlflow") as mock_mlflow:
        tracking.log_params({"a": 1})
        tracking.log_metric("m", 0.5, step=1)
        tracking.log_artifact("x.json")
        with tracking.start_run(run_name="r") as run:
            assert run.run_id is None
    mock_mlflow.log_params.assert_not_called()
    mock_mlflow.start_run.assert_not_called()


@mlflow_required
def test_setup_sqlite_active():
    expected = resolve_tracking_uri("sqlite:///mlflow.db")
    with (
        patch.object(tracking.mlflow, "set_tracking_uri") as set_uri,
        patch.object(tracking.mlflow, "set_experiment") as set_exp,
    ):
        active = tracking.setup("sqlite:///mlflow.db", "exp", enabled=True)
    assert active is True
    set_uri.assert_called_once_with(expected)
    set_exp.assert_called_once_with("exp")


@mlflow_required
def test_setup_http_unreachable_falls_back_to_sqlite():
    expected = resolve_tracking_uri("sqlite:///mlflow.db")
    with (
        patch.object(tracking, "_server_reachable", return_value=False),
        patch.object(tracking.mlflow, "set_tracking_uri") as set_uri,
        patch.object(tracking.mlflow, "set_experiment"),
    ):
        active = tracking.setup("http://127.0.0.1:5000", "exp", enabled=True)
    assert active is True
    set_uri.assert_called_once_with(expected)


@mlflow_required
def test_setup_http_reachable_used_directly():
    with (
        patch.object(tracking, "_server_reachable", return_value=True),
        patch.object(tracking.mlflow, "set_tracking_uri") as set_uri,
        patch.object(tracking.mlflow, "set_experiment"),
    ):
        active = tracking.setup("http://127.0.0.1:5000", "exp", enabled=True)
    assert active is True
    set_uri.assert_called_once_with("http://127.0.0.1:5000")


@mlflow_required
def test_start_run_logs_when_active():
    run_obj = MagicMock()
    run_obj.info.run_id = "abc123"
    with (
        patch.object(tracking.mlflow, "start_run", return_value=run_obj) as start,
        patch.object(tracking.mlflow, "end_run") as end,
    ):
        tracking._active = True
        with tracking.start_run(run_name="r") as run:
            assert run.run_id == "abc123"
    start.assert_called_once()
    end.assert_called_once()
