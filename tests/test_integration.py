"""Live Modal integration tests; enable with TEST_MODAL_INTEGRATION=1."""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_MODAL_INTEGRATION") != "1",
    reason="Integration tests require TEST_MODAL_INTEGRATION=1 and Modal auth",
)


@pytest.fixture(scope="module")
def simple_sklearn_model(tmp_path_factory):
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    data = load_iris(as_frame=True)
    model = LogisticRegression(max_iter=200).fit(data.data, data.target)
    model_path = tmp_path_factory.mktemp("modal-model") / "model"
    mlflow.sklearn.save_model(
        model,
        str(model_path),
        signature=infer_signature(data.data, model.predict(data.data)),
    )
    inputs = data.data.iloc[:3].to_dict(orient="list")
    return str(model_path), inputs, model.predict(data.data.iloc[:3]).tolist()


class TestModalIntegration:
    def test_client_creation(self):
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("modal")
        assert client.workspace is None
        assert client.target_uri == "modal"

    def test_list_deployments(self):
        from mlflow.deployments import get_deploy_client

        assert isinstance(get_deploy_client("modal").list_deployments(), list)

    @pytest.mark.parametrize(
        "scenario", ["basic", "pip-index", "pip-extra-index", "secret", "streaming", "proxy-auth", "batching"]
    )
    def test_create_and_delete_deployment(self, simple_sklearn_model, scenario):
        import modal
        import requests
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("modal")
        name = f"mlflow-test-{scenario}-{uuid.uuid4().hex[:10]}"
        secret_name = f"{name}-secret"
        config = {"memory": 1024, "timeout": 300, "max_containers": 1, "scaledown_window": 60}
        if scenario == "basic":
            config["extra_pip_packages"] = ["structlog>=24.0"]
        elif scenario == "pip-index":
            config["pip_index_url"] = "https://pypi.org/simple/"
        elif scenario == "pip-extra-index":
            config["pip_extra_index_url"] = "https://pypi.org/simple/"
        elif scenario == "secret":
            modal.Secret.objects.create(secret_name, {"PIP_EXTRA_INDEX_URL": "https://pypi.org/simple/"})
            config["modal_secret"] = secret_name
        elif scenario == "proxy-auth":
            assert os.environ.get("PROXY_AUTH_TOKEN_ID"), "PROXY_AUTH_TOKEN_ID is required"
            assert os.environ.get("PROXY_AUTH_TOKEN_SECRET"), "PROXY_AUTH_TOKEN_SECRET is required"
            config["proxy_auth"] = True
        elif scenario == "batching":
            config["enable_batching"] = True

        model_path, inputs, expected = simple_sklearn_model
        print(f"Temporary Modal deployment: {name}", flush=True)
        try:
            result = client.create_deployment(name=name, model_uri=model_path, config=config)
            assert result["name"] == name
            assert result["flavor"] == "python_function"
            assert result["endpoint_url"].startswith("https://")
            assert client.get_deployment(name)["name"] == name
            if scenario == "proxy-auth":
                response = requests.post(result["endpoint_url"], json=inputs, timeout=180)
                assert response.status_code == 401
            prediction = client.predict(name, inputs)
            assert prediction["predictions"] == expected
            if scenario == "streaming":
                assert list(client.predict_stream(name, inputs)) == [{"predictions": expected}]
        finally:
            try:
                assert client.delete_deployment(name)["deleted"] is True
                with pytest.raises(modal.exception.NotFoundError):
                    modal.Volume.from_name(f"{name}-model-volume").hydrate()
            finally:
                if scenario == "secret":
                    modal.Secret.objects.delete(secret_name)
