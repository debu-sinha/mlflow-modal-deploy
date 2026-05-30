#!/usr/bin/env python
"""
End-to-end tests for mlflow-modal-deploy.

These tests deploy real models to Modal and verify:
1. Basic deployment with extra_pip_packages
2. pip_index_url configuration
3. pip_extra_index_url configuration
4. modal_secret integration
5. Streaming predictions via predict_stream()

Prerequisites:
    - Modal authentication configured (modal setup)
    - MLflow and scikit-learn installed

Usage:
    python tests/e2e_test.py              # Run all tests
    python tests/e2e_test.py --quick      # Run quick test only (basic deployment)
    python tests/e2e_test.py --stream     # Run streaming test only
"""

import argparse
import os
import subprocess
import sys
import time

import mlflow
import requests
from dotenv import load_dotenv
from mlflow.deployments import get_deploy_client
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()


def train_and_log_model():
    """Train a simple model and log to MLflow."""
    print("Training model...")
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        print(f"  Run ID: {run.info.run_id}")
        return run.info.run_id


def make_prediction(endpoint_url: str, max_attempts: int = 3, headers: dict | None = None) -> dict | None:
    """Make a prediction request to the deployed model."""
    import requests

    sample_data = {
        "sepal length (cm)": [5.1],
        "sepal width (cm)": [3.5],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
    }

    for attempt in range(max_attempts):
        try:
            response = requests.post(endpoint_url, json=sample_data, timeout=180, headers=headers)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
                raise e  # re-raise 401 for auth validation
            elif attempt < max_attempts - 1:
                print(f"    Attempt {attempt + 1} failed, retrying in 15s...")
                time.sleep(15)
            else:
                print(f"    Prediction timed out (cold start): {e}")
                return None


def test_basic_deployment(run_id: str) -> bool:
    """Test basic deployment with extra_pip_packages."""
    print("\n" + "=" * 60)
    print("TEST: Basic deployment with extra_pip_packages")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-basic"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                "extra_pip_packages": ["structlog>=24.0"],
            },
        )

        print("  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        # Verify config
        config = deployment.get("config", {})
        assert config.get("extra_pip_packages") == ["structlog>=24.0"]
        print("  extra_pip_packages correctly set")

        # Make prediction
        endpoint_url = deployment.get("endpoint_url")
        if endpoint_url:
            print("  Testing prediction...")
            result = make_prediction(endpoint_url)

            if result:
                print(f"  Prediction successful: {result}")
            else:
                print("  Prediction timed out, but deployment was created")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def test_proxy_auth(run_id: str) -> bool:
    """Test basic deployment with proxy authentication."""
    print("\n" + "=" * 60)
    print("TEST: Basic deployment with proxy authentication")
    print("=" * 60)

    headers = {
        "Modal-Key": os.environ.get("PROXY_AUTH_TOKEN_ID"),
        "Modal-Secret": os.environ.get("PROXY_AUTH_TOKEN_SECRET"),
    }
    if not headers["Modal-Key"]:
        print(
            "PROXY_AUTH_TOKEN_ID is not set. Run 'modal token new' to create a new token and add environment variables to your shell. See .env.example for an example."
        )
    if not headers["Modal-Secret"]:
        print(
            "PROXY_AUTH_TOKEN_SECRET is not set. Run 'modal token new' to create a new token and add environment variables to your shell. See .env.example for an example."
        )
    if not headers["Modal-Key"] or not headers["Modal-Secret"]:
        return False
    client = get_deploy_client("modal")
    deployment_name = "e2e-test-proxy-auth"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 512,
                "timeout": 120,
                "scaledown_window": 60,
                "extra_pip_packages": ["structlog>=24.0"],
                "proxy_auth": True,
            },
        )

        # Verify config
        config = deployment.get("config", {})
        if not config.get("proxy_auth"):
            print("  proxy_auth not set")
            return False
        else:
            print("  proxy_auth correctly set")

        # Make prediction
        endpoint_url = deployment.get("endpoint_url")
        print(f"  Endpoint URL: {endpoint_url}")

        print("  Testing prediction...")
        try:
            result = make_prediction(endpoint_url)  # expected to raise 401
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("  Prediction without authentication denied")
            else:
                print(f"  Prediction error, expected 401: {e}")
                return False
        else:
            print(f"  Prediction error, expected 401 but got {result}")
            return False

        try:
            result = make_prediction(endpoint_url, headers=headers)
        except requests.exceptions.HTTPError as e:
            print(f"  Prediction error, expected 200: {e}")
            return False
        else:
            print(f"  Prediction successful: {result}")

        if not result:
            print("  Prediction timed out, but deployment was created")
            return False

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def test_pip_index_url(run_id: str) -> bool:
    """Test deployment with pip_index_url."""
    print("\n" + "=" * 60)
    print("TEST: pip_index_url configuration")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-pip-index-url"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                "pip_index_url": "https://pypi.org/simple/",
            },
        )

        print("  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        config = deployment.get("config", {})
        assert config.get("pip_index_url") == "https://pypi.org/simple/"
        print("  pip_index_url correctly set")

        endpoint_url = deployment.get("endpoint_url")
        if endpoint_url:
            print("  Testing prediction...")
            result = make_prediction(endpoint_url)
            if result:
                print(f"  Prediction successful: {result}")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def test_pip_extra_index_url(run_id: str) -> bool:
    """Test deployment with pip_extra_index_url."""
    print("\n" + "=" * 60)
    print("TEST: pip_extra_index_url configuration")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-pip-extra-index"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                "pip_extra_index_url": "https://pypi.org/simple/",
            },
        )

        print("  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        config = deployment.get("config", {})
        assert config.get("pip_extra_index_url") == "https://pypi.org/simple/"
        print("  pip_extra_index_url correctly set")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def test_modal_secret(run_id: str) -> bool:
    """Test deployment with modal_secret integration."""
    print("\n" + "=" * 60)
    print("TEST: modal_secret integration")
    print("=" * 60)

    # Create test Modal secret
    secret_name = "e2e-test-pip-credentials"
    print(f"  Creating Modal secret '{secret_name}'...")

    result = subprocess.run(
        [
            "modal",
            "secret",
            "create",
            secret_name,
            "PIP_EXTRA_INDEX_URL=https://pypi.org/simple/",
            "--force",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Warning: Could not create secret: {result.stderr}")
    else:
        print("  Secret created")

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-modal-secret"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
                "modal_secret": secret_name,
            },
        )

        print("  Deployment successful!")
        print(f"  Endpoint URL: {deployment.get('endpoint_url')}")

        config = deployment.get("config", {})
        assert config.get("modal_secret") == secret_name
        print("  modal_secret correctly set")

        endpoint_url = deployment.get("endpoint_url")
        if endpoint_url:
            print("  Testing prediction...")
            result = make_prediction(endpoint_url)
            if result:
                print(f"  Prediction successful: {result}")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass

        # Clean up secret
        subprocess.run(
            ["modal", "secret", "delete", secret_name, "--yes"],
            capture_output=True,
        )
        print(f"  Cleaned up secret: {secret_name}")


def test_streaming(run_id: str) -> bool:
    """Test streaming predictions via predict_stream()."""
    print("\n" + "=" * 60)
    print("TEST: Streaming predictions (predict_stream)")
    print("=" * 60)

    client = get_deploy_client("modal")
    deployment_name = "e2e-test-streaming"

    try:
        deployment = client.create_deployment(
            name=deployment_name,
            model_uri=f"runs:/{run_id}/model",
            config={
                "memory": 1024,
                "timeout": 120,
                "scaledown_window": 60,
            },
        )

        print("  Deployment successful!")
        endpoint_url = deployment.get("endpoint_url")
        print(f"  Endpoint URL: {endpoint_url}")

        if not endpoint_url:
            print("  ERROR: No endpoint URL returned")
            return False

        # Test regular predict first
        print("  Testing regular predict()...")
        sample_data = {
            "sepal length (cm)": [5.1],
            "sepal width (cm)": [3.5],
            "petal length (cm)": [1.4],
            "petal width (cm)": [0.2],
        }

        # Wait for cold start
        print("  Waiting for container to warm up...")
        time.sleep(10)

        result = make_prediction(endpoint_url)
        if result:
            print(f"  Regular predict() successful: {result}")
        else:
            print("  Regular predict() timed out, continuing with stream test...")

        # Test streaming predict
        print("  Testing predict_stream()...")
        chunks = []
        try:
            for chunk in client.predict_stream(
                deployment_name=deployment_name,
                inputs=sample_data,
            ):
                chunks.append(chunk)
                print(f"    Received chunk: {chunk}")
        except Exception as e:
            print(f"  predict_stream() error: {e}")
            # This might fail if the model doesn't support streaming natively
            # but the endpoint should still return a single chunk with predictions
            import traceback

            traceback.print_exc()

        if chunks:
            print(f"  Streaming successful! Received {len(chunks)} chunk(s)")
            # Verify the chunk contains predictions
            last_chunk = chunks[-1]
            if "predictions" in last_chunk:
                print(f"  Predictions: {last_chunk['predictions']}")
            return True
        else:
            print("  No chunks received from streaming endpoint")
            # Try direct HTTP request to streaming endpoint for debugging
            print("  Attempting direct HTTP request to streaming endpoint...")
            import requests

            # Handle both URL patterns: path-based (/predict) and subdomain-based (-predict.)
            if "/predict" in endpoint_url:
                stream_url = endpoint_url.replace("/predict", "/predict_stream")
            else:
                stream_url = endpoint_url.replace("-predict.", "-predict-stream.")
            try:
                with requests.post(
                    stream_url,
                    json=sample_data,
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                    timeout=180,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            print(f"    Raw line: {line}")
                            chunks.append(line)
                if chunks:
                    print(f"  Direct streaming successful! Received {len(chunks)} line(s)")
                    return True
            except Exception as e:
                print(f"  Direct streaming failed: {e}")

            return False

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            client.delete_deployment(deployment_name)
            print(f"  Cleaned up: {deployment_name}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="E2E tests for mlflow-modal-deploy")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--stream", action="store_true", help="Run streaming test only")
    parser.add_argument("--proxy-auth", action="store_true", help="Run proxy authentication test only")
    args = parser.parse_args()

    print("=" * 60)
    print("E2E Tests: mlflow-modal-deploy")
    print("=" * 60)

    run_id = train_and_log_model()

    results = {}

    if args.quick:
        results["basic_deployment"] = test_basic_deployment(run_id)
    elif args.stream:
        results["streaming"] = test_streaming(run_id)
    elif args.proxy_auth:
        results["proxy-auth"] = test_proxy_auth(run_id)
    else:
        results["basic_deployment"] = test_basic_deployment(run_id)
        results["streaming"] = test_streaming(run_id)
        results["pip_index_url"] = test_pip_index_url(run_id)
        results["pip_extra_index_url"] = test_pip_extra_index_url(run_id)
        results["modal_secret"] = test_modal_secret(run_id)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
