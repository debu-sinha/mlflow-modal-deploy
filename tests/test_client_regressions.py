"""Regression tests for failures found during real Modal deployments."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import modal
import pytest
from mlflow.exceptions import MlflowException

from mlflow_modal.deployment import ModalDeploymentClient


@pytest.fixture
def modal_stub(monkeypatch):
    stub = SimpleNamespace(
        Volume=SimpleNamespace(objects=SimpleNamespace(delete=MagicMock())), Cls=MagicMock(), exception=modal.exception
    )
    stub.Cls.from_name.return_value.return_value.predict.get_web_url.return_value = "https://predict.modal.run"
    stub.Cls.from_name.return_value.return_value.predict_stream.get_web_url.return_value = "https://stream.modal.run"
    monkeypatch.setattr("mlflow_modal.deployment._import_modal", lambda: stub)
    return stub


@pytest.fixture
def client(modal_stub):
    return ModalDeploymentClient("modal")


def test_current_app_listing(client, monkeypatch):
    run = MagicMock(
        return_value=SimpleNamespace(
            returncode=0, stdout='[{"app_id":"ap-test","description":"model","state":"deployed"}]'
        )
    )
    monkeypatch.setattr("mlflow_modal.deployment.subprocess.run", run)
    assert client.get_deployment("model") == {
        "name": "model",
        "app_id": "ap-test",
        "state": "deployed",
        "endpoint_url": "https://predict.modal.run",
        "streaming_url": "https://stream.modal.run",
    }
    assert run.call_args.args[0][:3] == [sys.executable, "-m", "modal"]


def test_delete_stops_before_removing_volume(client, monkeypatch, modal_stub):
    operations = []

    def stop(command, **kwargs):
        assert command[-2:] == ["model", "--yes"]
        operations.append("stop")
        return SimpleNamespace(returncode=0, stderr="")

    def delete(name, **kwargs):
        assert name == "model-model-volume"
        assert kwargs == {"environment_name": None, "allow_missing": True}
        operations.append("volume")

    monkeypatch.setattr("mlflow_modal.deployment.subprocess.run", stop)
    monkeypatch.setattr(modal_stub.Volume.objects, "delete", delete)
    assert client.delete_deployment("model")["deleted"]
    assert operations == ["stop", "volume"]


def test_failed_stop_preserves_volume(client, monkeypatch, modal_stub):
    monkeypatch.setattr(
        "mlflow_modal.deployment.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="Permission denied"),
    )
    delete = MagicMock()
    monkeypatch.setattr(modal_stub.Volume.objects, "delete", delete)
    with pytest.raises(MlflowException, match="Permission denied"):
        client.delete_deployment("model")
    delete.assert_not_called()


def test_failed_volume_cleanup_is_reported(client, monkeypatch, modal_stub):
    monkeypatch.setattr(
        "mlflow_modal.deployment.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stderr="")
    )
    monkeypatch.setattr(modal_stub.Volume.objects, "delete", MagicMock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(MlflowException, match="failed to delete volume"):
        client.delete_deployment("model")


@pytest.mark.parametrize(
    "error", ["No App with name 'model' found in the 'main' environment.", "App not found", "App is already stopped."]
)
def test_absent_or_stopped_app_still_cleans_volume(client, monkeypatch, modal_stub, error):
    monkeypatch.setattr(
        "mlflow_modal.deployment.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stderr=error)
    )
    delete = MagicMock()
    monkeypatch.setattr(modal_stub.Volume.objects, "delete", delete)
    assert client.delete_deployment("model")["deleted"]
    delete.assert_called_once()


def test_proxy_headers_without_app_metadata(client, monkeypatch):
    monkeypatch.setenv("PROXY_AUTH_TOKEN_ID", "test-id")
    monkeypatch.setenv("PROXY_AUTH_TOKEN_SECRET", "test-secret")
    assert client._build_proxy_auth_headers("model") == {"Modal-Key": "test-id", "Modal-Secret": "test-secret"}


def test_partial_proxy_credentials_rejected(client, monkeypatch):
    monkeypatch.setenv("PROXY_AUTH_TOKEN_ID", "test-id")
    monkeypatch.delenv("PROXY_AUTH_TOKEN_SECRET", raising=False)
    with pytest.raises(MlflowException, match="PROXY_AUTH_TOKEN_SECRET"):
        client._build_proxy_auth_headers("model")


def test_prediction_response_is_not_nested(client, monkeypatch):
    monkeypatch.setattr(client, "get_deployment", lambda name: {"endpoint_url": "https://example.modal.run"})
    monkeypatch.setattr(client, "_build_proxy_auth_headers", lambda name: {})
    response = MagicMock()
    response.json.return_value = {"predictions": [1, 2]}
    monkeypatch.setattr("requests.post", lambda *a, **k: response)
    assert client.predict("model", {"value": [1, 2]})["predictions"] == [1, 2]
