"""
Example: Deploy a model with extra pip packages at deployment time

This example demonstrates the `extra_pip_packages` configuration option,
which allows specifying additional Python packages at deployment time that
aren't in the model's requirements.txt.

Use cases for extra_pip_packages:
1. Missing transitive dependencies that MLflow didn't auto-detect
2. Production monitoring packages (prometheus_client, structlog)
3. Inference-only packages not needed during training
4. Version overrides for production compatibility

Prerequisites:
    pip install mlflow-modal-deploy scikit-learn
    modal setup  # Configure Modal authentication

Usage:
    python deploy_with_extra_packages.py
"""

import mlflow
from mlflow.deployments import get_deploy_client
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_log_model():
    """Train a model and log it to MLflow."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")
        return run.info.run_id


def deploy_with_extra_packages(run_id: str):
    """
    Deploy the model with additional packages specified at deployment time.

    This is useful when:
    - The model's requirements.txt is missing dependencies
    - You need production-only packages (monitoring, logging)
    - You want to override versions for compatibility
    """
    client = get_deploy_client("modal")

    deployment = client.create_deployment(
        name="iris-classifier-extra-deps",
        model_uri=f"runs:/{run_id}/model",
        config={
            "memory": 1024,
            "cpu": 1.0,
            "timeout": 60,
            "min_containers": 0,
            # Add extra packages at deployment time
            # These are installed alongside the model's auto-detected requirements
            "extra_pip_packages": [
                "structlog>=24.0",  # Production logging
                "numpy>=1.24",  # Ensure compatible numpy version
            ],
        },
    )

    print("\nDeployment successful!")
    print(f"Endpoint URL: {deployment.get('endpoint_url')}")
    return deployment


def deploy_transformers_model_example():
    """
    Example showing how to add inference-time dependencies for transformer models.

    Transformers models often need additional packages for efficient inference
    that weren't captured during training.
    """
    get_deploy_client("modal")  # Verify plugin is installed

    # Note: This is a hypothetical deployment showing the pattern
    # In real usage, you'd have an actual transformers model URI
    print("""
    Example configuration for deploying a Transformers model:

    config = {
        "gpu": "A100",
        "memory": 16384,
        "extra_pip_packages": [
            "accelerate>=0.24",     # Required for efficient GPU inference
            "bitsandbytes>=0.41",   # For quantization
            "safetensors>=0.4",     # Fast model loading
            "sentencepiece",        # Tokenization (often missed)
            "protobuf>=3.20",       # Version compatibility fix
        ],
    }
    """)


def print_config_examples():
    """Print various configuration examples for extra_pip_packages."""
    examples = [
        {
            "use_case": "Production monitoring",
            "config": {
                "extra_pip_packages": [
                    "prometheus_client>=0.19",
                    "structlog>=24.0",
                    "opentelemetry-api>=1.20",
                ],
            },
        },
        {
            "use_case": "ML model with GPU inference optimizations",
            "config": {
                "gpu": "A100",
                "extra_pip_packages": [
                    "onnxruntime-gpu>=1.16",
                    "tensorrt>=8.6",
                ],
            },
        },
        {
            "use_case": "NLP model with missing tokenizer dependencies",
            "config": {
                "extra_pip_packages": [
                    "sentencepiece",
                    "tiktoken>=0.5",
                    "tokenizers>=0.15",
                ],
            },
        },
        {
            "use_case": "Data processing at inference time",
            "config": {
                "extra_pip_packages": [
                    "pyarrow>=14.0",
                    "polars>=0.19",
                ],
            },
        },
        {
            "use_case": "Private PyPI server (unauthenticated)",
            "config": {
                "pip_index_url": "https://pypi.my-company.com/simple/",
                "pip_extra_index_url": "https://pypi.org/simple/",
                "extra_pip_packages": ["my-internal-package>=1.0"],
            },
        },
        {
            "use_case": "Private PyPI server (authenticated via Modal secret)",
            "config": {
                "modal_secret": "pypi-auth",
                "extra_pip_packages": ["my-private-package>=2.0"],
            },
        },
    ]

    print("\n=== Extra Pip Packages Configuration Examples ===\n")
    for example in examples:
        print(f"Use case: {example['use_case']}")
        print(f"Config: {example['config']}")
        print()


def list_deployments():
    """List all Modal deployments."""
    client = get_deploy_client("modal")
    deployments = client.list_deployments()
    print("\nCurrent deployments:")
    for d in deployments:
        print(f"  - {d.get('name')}")
    return deployments


def cleanup(deployment_name: str):
    """Delete a deployment."""
    client = get_deploy_client("modal")
    client.delete_deployment(deployment_name)
    print(f"\nDeleted deployment: {deployment_name}")


if __name__ == "__main__":
    # Show configuration examples
    print_config_examples()

    # Show transformers example pattern
    deploy_transformers_model_example()

    # Train and log model
    print("\nTraining model...")
    run_id = train_and_log_model()

    # Deploy with extra packages
    print("\nDeploying to Modal with extra packages...")
    deployment = deploy_with_extra_packages(run_id)

    # List deployments
    list_deployments()

    # Cleanup (uncomment to delete)
    # cleanup("iris-classifier-extra-deps")
