"""Integration tests for Modal deployment - requires Modal authentication."""

import contextlib
import os

import pytest

# Skip all tests in this module if TEST_MODAL_INTEGRATION is not set
pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_MODAL_INTEGRATION") != "1",
    reason="Integration tests require TEST_MODAL_INTEGRATION=1 and Modal auth",
)


@pytest.fixture
def simple_sklearn_model(tmp_path):
    """Create a simple sklearn model for testing."""
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    model_path = tmp_path / "model"
    mlflow.sklearn.save_model(model, str(model_path))

    return str(model_path)


class TestModalIntegration:
    def test_client_creation(self):
        from mlflow_modal import ModalDeploymentClient

        client = ModalDeploymentClient("modal")
        assert client.workspace is None
        assert client.target_uri == "modal"

    def test_list_deployments(self):
        from mlflow_modal import ModalDeploymentClient

        client = ModalDeploymentClient("modal")
        deployments = client.list_deployments()

        assert isinstance(deployments, list)

    def test_create_and_delete_deployment(self, simple_sklearn_model):
        """Full integration test: create, verify, and delete deployment."""
        from mlflow_modal import ModalDeploymentClient

        client = ModalDeploymentClient("modal")
        deployment_name = "mlflow-modal-test-deployment"

        try:
            # Create deployment
            result = client.create_deployment(
                name=deployment_name,
                model_uri=simple_sklearn_model,
                config={
                    "memory": 512,
                    "timeout": 300,
                },
            )

            assert result["name"] == deployment_name
            assert result["flavor"] == "python_function"
            assert "endpoint_url" in result

            # Verify deployment exists
            deployment = client.get_deployment(deployment_name)
            assert deployment["name"] == deployment_name

        finally:
            # Cleanup: delete deployment
            with contextlib.suppress(Exception):
                client.delete_deployment(deployment_name)
