"""
Modal Deployment Client for MLflow models.

This module provides the ModalDeploymentClient class for deploying
MLflow models to Modal's serverless infrastructure.
"""

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
from collections.abc import Iterator
from typing import Any

import requests
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

from .codegen import ModalAppCodeConfig, generate_modal_app_code

_logger = logging.getLogger(__name__)

# Default configuration values
_DEFAULT_GPU = None
_DEFAULT_MEMORY = 512  # MB
_DEFAULT_CPU = 1.0
_DEFAULT_TIMEOUT = 300  # seconds
_DEFAULT_STARTUP_TIMEOUT = None  # seconds, overrides timeout during startup
_DEFAULT_SCALEDOWN_WINDOW = 60  # seconds
_DEFAULT_CONCURRENT_INPUTS = 1
_DEFAULT_TARGET_INPUTS = None  # autoscaler target concurrency
_DEFAULT_MIN_CONTAINERS = 0
_DEFAULT_MAX_CONTAINERS = None
_DEFAULT_BUFFER_CONTAINERS = None  # extra idle containers under load

# Supported GPU types (see https://modal.com/docs/guide/gpu)
SUPPORTED_GPUS = [
    "T4",
    "L4",
    "L40S",
    "A10G",
    "A10",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "H100",
    "H200",
    "B200",
    "RTX-PRO-6000",
]


def _get_model_requirements(model_path: str) -> tuple[list[str], list[str]]:
    """
    Extract Python requirements from an MLflow model.

    Returns:
        Tuple of (pip_requirements, wheel_files) where wheel_files are paths
        to .whl files in the model's code/ directory.
    """
    requirements = []
    wheel_files = []

    # Check for wheel files in code/ directory
    code_dir = os.path.join(model_path, "code")
    if os.path.exists(code_dir):
        for filename in os.listdir(code_dir):
            if filename.endswith(".whl"):
                wheel_files.append(os.path.join(code_dir, filename))
                _logger.info(f"Found wheel dependency: {filename}")

    req_file = os.path.join(model_path, "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                # Skip wheel references - we handle them separately
                if (
                    line
                    and not line.startswith("#")
                    and not line.lower().startswith("mlflow")
                    and not line.endswith(".whl")
                    and "code/" not in line
                ):
                    requirements.append(line)
        return requirements, wheel_files

    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, dict) and "pip" in dep:
                    requirements.extend(
                        pip_dep
                        for pip_dep in dep["pip"]
                        if not pip_dep.lower().startswith("mlflow") and not pip_dep.endswith(".whl")
                    )
                elif isinstance(dep, str) and not dep.startswith("python") and not dep.lower().startswith("mlflow"):
                    requirements.append(dep)
        except Exception as e:
            _logger.warning(f"Failed to parse conda.yaml: {e}")
    return requirements, wheel_files


def _get_model_python_version(model_path: str) -> str | None:
    """Extract Python version from an MLflow model."""
    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, str) and dep.startswith("python"):
                    version = dep.split("=")[-1].split(">")[-1].split("<")[-1]
                    parts = version.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
        except Exception as e:
            _logger.warning(f"Failed to parse Python version: {e}")
    return None


def _clear_volume(modal, volume_name: str) -> None:
    """Clear Modal volume to allow redeployment."""
    try:
        volume = modal.Volume.from_name(volume_name)
        for entry in volume.listdir("/"):
            with contextlib.suppress(Exception):
                volume.remove_file(f"/{entry.path}")
        _logger.info(f"Cleared volume: {volume_name}")
    except Exception as e:
        _logger.debug(f"Could not clear volume {volume_name}: {e}")


def _get_preferred_deployment_flavor(model_config):
    """Obtain the flavor MLflow prefers for deployment."""
    if PYFUNC_FLAVOR_NAME in model_config.flavors:
        return PYFUNC_FLAVOR_NAME
    raise MlflowException(
        message=(
            "The specified model does not contain the python_function flavor "
            "which is required for Modal deployment. "
            f"The model contains the following flavors: {list(model_config.flavors.keys())}."
        ),
        error_code=RESOURCE_DOES_NOT_EXIST,
    )


def _validate_deployment_flavor(model_config, flavor):
    """Validate that the specified flavor is supported and present in the model."""
    if flavor != PYFUNC_FLAVOR_NAME:
        raise MlflowException(
            message=f"Flavor '{flavor}' is not supported for Modal deployment. "
            f"Only '{PYFUNC_FLAVOR_NAME}' is supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if flavor not in model_config.flavors:
        raise MlflowException(
            message=f"The specified model does not contain the '{flavor}' flavor. "
            f"Available flavors: {list(model_config.flavors.keys())}.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )


def _import_modal():
    """Import modal and raise helpful error if not installed."""
    try:
        import modal

        return modal
    except ImportError as e:
        raise MlflowException(
            "The `modal` package is required for Modal deployments. Please install it with: pip install modal"
        ) from e


_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def _sanitize_deployment_name(name: str) -> str:
    """Validate and sanitize deployment name to prevent code injection."""
    if not name:
        raise MlflowException("Deployment name cannot be empty", error_code=INVALID_PARAMETER_VALUE)
    if len(name) > 63:
        raise MlflowException(
            f"Deployment name '{name}' exceeds 63 character limit", error_code=INVALID_PARAMETER_VALUE
        )
    if not _VALID_NAME_PATTERN.match(name):
        raise MlflowException(
            f"Invalid deployment name '{name}'. Must start with a letter and contain only "
            "alphanumeric characters, underscores, and hyphens.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return name


def _build_modal_app_codegen_config(
    app_name: str,
    config: dict[str, Any],
    model_requirements: list[str] | None,
    wheel_filenames: list[str] | None,
) -> ModalAppCodeConfig:
    """Build a ModalAppCodeConfig from the raw deployment config."""

    gpu_config = config.get("gpu")
    memory = config.get("memory", _DEFAULT_MEMORY)
    cpu = config.get("cpu", _DEFAULT_CPU)
    timeout = config.get("timeout", _DEFAULT_TIMEOUT)
    startup_timeout = config.get("startup_timeout", _DEFAULT_STARTUP_TIMEOUT)
    scaledown_window = config.get("scaledown_window", _DEFAULT_SCALEDOWN_WINDOW)
    enable_batching = config.get("enable_batching", False)
    max_batch_size = config.get("max_batch_size", 8)
    batch_wait_ms = config.get("batch_wait_ms", 100)
    python_version = config.get("python_version", "3.10")
    concurrent_inputs = config.get("concurrent_inputs", _DEFAULT_CONCURRENT_INPUTS)
    target_inputs = config.get("target_inputs", _DEFAULT_TARGET_INPUTS)
    buffer_containers = config.get("buffer_containers", _DEFAULT_BUFFER_CONTAINERS)
    extra_pip_packages = config.get("extra_pip_packages", [])
    pip_index_url = config.get("pip_index_url")
    pip_extra_index_url = config.get("pip_extra_index_url")
    modal_secret = config.get("modal_secret")
    min_containers = config.get("min_containers", _DEFAULT_MIN_CONTAINERS)
    max_containers = config.get("max_containers")
    proxy_auth = config.get("proxy_auth", False)

    return ModalAppCodeConfig(
        app_name=app_name,
        python_version=python_version,
        model_requirements=model_requirements,
        extra_pip_packages=extra_pip_packages,
        pip_index_url=pip_index_url,
        pip_extra_index_url=pip_extra_index_url,
        gpu_config=gpu_config,
        concurrent_inputs=concurrent_inputs,
        target_inputs=target_inputs,
        memory=memory,
        cpu=cpu,
        timeout=timeout,
        min_containers=min_containers,
        max_containers=max_containers,
        buffer_containers=buffer_containers,
        scaledown_window=scaledown_window,
        startup_timeout=startup_timeout,
        modal_secret=modal_secret,
        wheel_filenames=wheel_filenames,
        enable_batching=enable_batching,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
        proxy_auth=proxy_auth,
    )


def _generate_modal_app_code(
    app_name: str,
    config: dict[str, Any],
    model_requirements: list[str] | None = None,
    wheel_filenames: list[str] | None = None,
) -> str:
    """
    Generate Modal app Python code for serving an MLflow model.

    Args:
        app_name: Name of the Modal app
        config: Deployment configuration
        model_requirements: List of pip requirements
        wheel_filenames: List of wheel filenames (just names, not paths) to install from volume
    """
    codegen_config = _build_modal_app_codegen_config(
        app_name=app_name,
        config=config,
        model_requirements=model_requirements,
        wheel_filenames=wheel_filenames,
    )
    return generate_modal_app_code(codegen_config)


class ModalDeploymentClient(BaseDeploymentClient):
    """
    Client for deploying MLflow models to Modal.

    Modal is a serverless platform for running Python code in the cloud.
    This client enables deploying MLflow models as Modal web endpoints.

    Args:
        target_uri: A URI that follows one of the following formats:
            - ``modal``: Uses default Modal workspace from environment
            - ``modal:/workspace-name``: Uses the specified workspace

    Example:
        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("modal")
            client.create_deployment(
                name="my-model",
                model_uri="runs:/abc123/model",
                config={"gpu": "T4", "enable_batching": True},
            )
    """

    def __init__(self, target_uri: str):
        super().__init__(target_uri=target_uri)
        self.workspace = self._parse_workspace_from_uri(target_uri)
        self._validate_modal_auth()

    def _parse_workspace_from_uri(self, target_uri: str) -> str | None:
        """Parse workspace name from target URI."""
        parsed = urllib.parse.urlparse(target_uri)
        if parsed.scheme == "modal":
            path = parsed.path.strip("/")
            return path or None
        return None

    def _validate_modal_auth(self):
        """Validate that Modal authentication is configured."""
        _import_modal()

    def _get_modal_workspace(self) -> str | None:
        """Get the current Modal workspace/username."""
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _construct_endpoint_url(self, deployment_name: str, method: str = "predict") -> str | None:
        """Construct endpoint URL from Modal naming convention."""
        workspace = self.workspace or self._get_modal_workspace()
        if not workspace:
            return None
        # Modal URL pattern: https://{workspace}--{app-name}-mlflowmodel-{method}.modal.run
        # Note: underscores in method names become hyphens
        method_slug = method.replace("_", "-")
        return f"https://{workspace}--{deployment_name}-mlflowmodel-{method_slug}.modal.run"

    def _default_deployment_config(self) -> dict[str, Any]:
        """Return default deployment configuration."""
        return {
            "gpu": _DEFAULT_GPU,
            "memory": _DEFAULT_MEMORY,
            "cpu": _DEFAULT_CPU,
            "timeout": _DEFAULT_TIMEOUT,
            "startup_timeout": _DEFAULT_STARTUP_TIMEOUT,
            "scaledown_window": _DEFAULT_SCALEDOWN_WINDOW,
            "enable_batching": False,
            "max_batch_size": 8,
            "batch_wait_ms": 100,
            "concurrent_inputs": _DEFAULT_CONCURRENT_INPUTS,
            "target_inputs": _DEFAULT_TARGET_INPUTS,
            "min_containers": _DEFAULT_MIN_CONTAINERS,
            "max_containers": _DEFAULT_MAX_CONTAINERS,
            "buffer_containers": _DEFAULT_BUFFER_CONTAINERS,
            "python_version": None,
            "extra_pip_packages": [],
            "pip_index_url": None,
            "pip_extra_index_url": None,
            "modal_secret": None,
            "proxy_auth": False,
        }

    def _validate_gpu_config(self, gpu_config: str | list[str]) -> None:
        """Validate GPU configuration supports Modal's GPU specification formats."""

        def validate_single_gpu(gpu: str) -> None:
            # Extract base GPU type from multi-GPU syntax (e.g., "H100:8" -> "H100")
            base_gpu = gpu.split(":")[0].rstrip("!+")
            if base_gpu not in SUPPORTED_GPUS:
                raise MlflowException(
                    f"Unsupported GPU type: {gpu}. Supported: {SUPPORTED_GPUS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if isinstance(gpu_config, list):
            for gpu in gpu_config:
                validate_single_gpu(gpu)
        else:
            validate_single_gpu(gpu_config)

    def _apply_custom_config(self, config: dict[str, Any], custom_config: dict[str, Any] | None) -> dict[str, Any]:
        """Apply custom configuration over defaults."""
        if not custom_config:
            return config

        # Handle deprecated parameter names for backward compatibility
        deprecated_mappings = {
            "container_idle_timeout": "scaledown_window",
            "allow_concurrent_inputs": "concurrent_inputs",
        }
        for old_key, new_key in deprecated_mappings.items():
            if old_key in custom_config and new_key not in custom_config:
                custom_config[new_key] = custom_config.pop(old_key)
                _logger.warning(f"Config '{old_key}' is deprecated, use '{new_key}' instead")

        int_fields = {
            "memory",
            "timeout",
            "startup_timeout",
            "scaledown_window",
            "max_batch_size",
            "batch_wait_ms",
            "min_containers",
            "max_containers",
            "buffer_containers",
            "concurrent_inputs",
            "target_inputs",
        }
        float_fields = {"cpu"}
        bool_fields = {
            "enable_batching",
            "proxy_auth",
        }

        for key, value in custom_config.items():
            if key not in config:
                config[key] = value
                continue

            if value is None:
                config[key] = value
            elif key in int_fields and not isinstance(value, int):
                config[key] = int(value)
            elif key in float_fields and not isinstance(value, float):
                config[key] = float(value)
            elif key in bool_fields and not isinstance(value, bool):
                config[key] = str(value).lower() == "true"
            else:
                config[key] = value

        if config.get("gpu"):
            self._validate_gpu_config(config["gpu"])

        return config

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Deploy an MLflow model to Modal.

        Args:
            name: Name of the deployment (will be used as Modal app name)
            model_uri: URI of the MLflow model to deploy
            flavor: Model flavor (only python_function supported)
            config: Deployment configuration dict with keys:
                - gpu: GPU type, multi-GPU ("H100:8"), or fallback list (["H100", "A100"])
                - memory: Memory in MB (default: 512)
                - cpu: CPU cores (default: 1.0)
                - timeout: Request timeout in seconds (default: 300)
                - scaledown_window: Seconds before idle container scales down (default: 60)
                - enable_batching: Enable dynamic batching (default: False)
                - max_batch_size: Max batch size (default: 8)
                - batch_wait_ms: Batch wait time in ms (default: 100)
                - concurrent_inputs: Max concurrent inputs per container (default: 1)
                - min_containers: Minimum containers (default: 0)
                - max_containers: Maximum containers (default: None)
                - proxy_auth: Enable proxy authentication in modal's ENDPOINT URL (default: False)
                - python_version: Python version (default: auto-detect)
            endpoint: Unused, kept for API compatibility

        Returns:
            Dictionary with deployment information including endpoint URL
        """
        name = _sanitize_deployment_name(name)
        modal = _import_modal()

        with TempDir() as tmp_dir:
            local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())
            model_config = Model.load(local_model_path)

            if flavor is None:
                flavor = _get_preferred_deployment_flavor(model_config)
            else:
                _validate_deployment_flavor(model_config, flavor)

            deployment_config = self._default_deployment_config()
            deployment_config = self._apply_custom_config(deployment_config, config)

            if deployment_config.get("python_version") is None:
                detected_version = _get_model_python_version(local_model_path)
                deployment_config["python_version"] = detected_version or "3.10"

            model_requirements, wheel_files = _get_model_requirements(local_model_path)
            if model_requirements:
                _logger.info(f"Detected pip requirements: {model_requirements}")
            if wheel_files:
                _logger.info(f"Detected wheel files: {[os.path.basename(w) for w in wheel_files]}")

            wheel_filenames = [os.path.basename(w) for w in wheel_files] if wheel_files else None
            app_code = _generate_modal_app_code(name, deployment_config, model_requirements, wheel_filenames)

            app_file = os.path.join(tmp_dir.path(), "modal_app.py")
            with open(app_file, "w") as f:
                f.write(app_code)

            volume_name = f"{name}-model-volume"
            _clear_volume(modal, volume_name)

            # Verify model files exist before upload
            mlmodel_path = os.path.join(local_model_path, "MLmodel")
            if not os.path.exists(mlmodel_path):
                raise MlflowException(
                    f"MLmodel file not found at {mlmodel_path}. "
                    f"Contents of {local_model_path}: {os.listdir(local_model_path)}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            _logger.info(f"Uploading model to Modal volume: {volume_name}")
            _logger.info(f"Model files: {os.listdir(local_model_path)}")
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)

            # Upload model files using Modal 1.0 batch_upload context manager
            with volume.batch_upload(force=True) as batch:
                # Upload all files and directories from model path to volume root
                for item in os.listdir(local_model_path):
                    item_path = os.path.join(local_model_path, item)
                    if os.path.isfile(item_path):
                        batch.put_file(item_path, f"/{item}")
                        _logger.debug(f"Uploaded file: /{item}")
                    elif os.path.isdir(item_path):
                        batch.put_directory(item_path, f"/{item}")
                        _logger.debug(f"Uploaded directory: /{item}")

                # Upload wheel files to a separate wheels/ directory
                if wheel_files:
                    wheels_dir = os.path.join(tmp_dir.path(), "wheels")
                    os.makedirs(wheels_dir, exist_ok=True)
                    for whl in wheel_files:
                        shutil.copy(whl, wheels_dir)
                    _logger.info(f"Uploading {len(wheel_files)} wheel file(s) to volume")
                    batch.put_directory(wheels_dir, "/wheels")

            _logger.info(f"Deploying Modal app: {name}")
            deploy_cmd = ["modal", "deploy", app_file]
            if self.workspace:
                deploy_cmd.extend(["--env", self.workspace])

            result = subprocess.run(deploy_cmd, capture_output=True, text=True, cwd=tmp_dir.path())

            # Log deployment output for debugging
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    _logger.info(f"[modal deploy] {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if result.returncode != 0:
                        _logger.error(f"[modal deploy] {line}")
                    else:
                        _logger.warning(f"[modal deploy] {line}")

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise MlflowException(
                    f"Failed to deploy Modal app. Build logs:\n{error_msg}\n\n"
                    "Tip: Run with MLFLOW_LOGGING_LEVEL=DEBUG for more details.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            endpoint_url = None
            streaming_url = None
            for line in result.stdout.split("\n"):
                if "https://" in line and ".modal.run" in line:
                    match = re.search(r"(https://[^\s]+\.modal\.run[^\s]*)", line)
                    if match:
                        url = match.group(1)
                        # Prefer the regular predict endpoint over streaming
                        if "-predict-stream." in url or "/predict_stream" in url:
                            streaming_url = url
                        elif "-predict." in url or "/predict" in url or endpoint_url is None:
                            endpoint_url = url
            # Fall back to streaming URL if no regular predict URL found
            if endpoint_url is None:
                endpoint_url = streaming_url
            if endpoint_url is None:
                endpoint_url = self._construct_endpoint_url(name, "predict")

            return {
                "name": name,
                "flavor": flavor,
                "model_uri": model_uri,
                "endpoint_url": endpoint_url,
                "config": deployment_config,
            }

    def update_deployment(
        self,
        name: str,
        model_uri: str | None = None,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing Modal deployment."""
        if model_uri is None:
            raise MlflowException(
                "model_uri is required for updating a Modal deployment",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return self.create_deployment(name, model_uri, flavor, config, endpoint)

    def delete_deployment(self, name: str, endpoint: str | None = None) -> dict[str, Any]:  # type: ignore[override]
        """Delete a Modal deployment."""
        modal = _import_modal()

        stop_cmd = ["modal", "app", "stop", name]
        if self.workspace:
            stop_cmd.extend(["--env", self.workspace])

        result = subprocess.run(stop_cmd, capture_output=True, text=True)

        volume_name = f"{name}-model-volume"
        try:
            volume = modal.Volume.from_name(volume_name)
            volume.delete()
            _logger.info(f"Deleted volume: {volume_name}")
        except Exception as e:
            _logger.debug(f"Could not delete volume {volume_name}: {e}")

        if result.returncode != 0 and "not found" not in (result.stderr or "").lower():
            raise MlflowException(
                f"Failed to delete Modal app: {result.stderr}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return {"name": name, "deleted": True}

    def list_deployments(self, endpoint: str | None = None) -> list[dict[str, Any]]:
        """List all Modal deployments."""
        list_cmd = ["modal", "app", "list", "--json"]
        if self.workspace:
            list_cmd.extend(["--env", self.workspace])

        result = subprocess.run(list_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise MlflowException(
                f"Failed to list Modal apps: {result.stderr}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        try:
            apps = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            apps = []

        # Modal JSON uses "Description" for app name and "App ID" for ID
        return [
            {
                "name": app.get("Description", app.get("name", app.get("App ID", "unknown"))),
                "app_id": app.get("App ID", app.get("app_id")),
                "state": app.get("State", app.get("state")),
            }
            for app in apps
        ]

    def get_deployment(self, name: str, endpoint: str | None = None) -> dict[str, Any]:
        """Get information about a specific Modal deployment."""
        deployments = self.list_deployments()
        for deployment in deployments:
            if deployment.get("name") == name:
                return deployment

        raise MlflowException(
            f"Deployment '{name}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def predict(
        self,
        deployment_name: str | None = None,
        inputs: Any = None,
        endpoint: str | None = None,
    ) -> PredictionsResponse:
        """Make predictions using a deployed Modal model."""
        if deployment_name is None:
            raise MlflowException(
                "deployment_name is required",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if inputs is None:
            raise MlflowException(
                "inputs is required for prediction",
                error_code=INVALID_PARAMETER_VALUE,
            )

        deployment = self.get_deployment(deployment_name)
        endpoint_url = deployment.get("endpoint_url")

        if not endpoint_url:
            # Construct URL from Modal naming convention
            endpoint_url = self._construct_endpoint_url(deployment_name, "predict")

        if not endpoint_url:
            raise MlflowException(
                f"Could not find endpoint URL for deployment '{deployment_name}'. "
                "The deployment may not have a web endpoint configured.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        proxy_auth_headers = self._build_proxy_auth_headers(deployment_name)
        response = requests.post(endpoint_url, json=inputs, timeout=300, headers=proxy_auth_headers)

        response.raise_for_status()

        return PredictionsResponse(predictions=response.json())

    def _build_proxy_auth_headers(self, deployment_name: str) -> dict[str, str]:
        proxy_auth_enabled = self.get_deployment(deployment_name).get("config", {}).get("proxy_auth", False)
        if not proxy_auth_enabled:
            return {}
        token_id = os.environ.get("PROXY_AUTH_TOKEN_ID")
        token_secret = os.environ.get("PROXY_AUTH_TOKEN_SECRET")
        if not token_id or not token_secret:
            missing = []
            if not token_id:
                missing.append("PROXY_AUTH_TOKEN_ID")
            if not token_secret:
                missing.append("PROXY_AUTH_TOKEN_SECRET")
            raise MlflowException(
                f"Proxy auth is enabled but the following environment variables are not set: {', '.join(missing)}. "
                "Run 'modal token new' to create a new token.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return {"Modal-Key": token_id, "Modal-Secret": token_secret}

    def predict_stream(
        self,
        deployment_name: str | None = None,
        inputs: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream predictions from a deployed Modal model.

        This method is compatible with the MLflow Databricks client API,
        enabling consistent streaming across deployment targets.

        Args:
            deployment_name: Name of the deployment to query.
            inputs: The inputs to the query, as a dictionary.
            endpoint: Unused, kept for API compatibility.

        Returns:
            An iterator of dictionaries containing streamed response chunks.

        Example:
            .. code-block:: python

                from mlflow.deployments import get_deploy_client

                client = get_deploy_client("modal")
                for chunk in client.predict_stream(
                    deployment_name="my-llm-model",
                    inputs={
                        "messages": [{"role": "user", "content": "Hello!"}],
                        "temperature": 0.7,
                        "max_tokens": 100,
                    },
                ):
                    print(chunk)
                    # Example chunk:
                    # {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}
        """
        if deployment_name is None:
            raise MlflowException(
                "deployment_name is required",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if inputs is None:
            raise MlflowException(
                "inputs is required for prediction",
                error_code=INVALID_PARAMETER_VALUE,
            )

        deployment = self.get_deployment(deployment_name)
        endpoint_url = deployment.get("endpoint_url")

        # Construct streaming endpoint URL
        stream_url = None
        if endpoint_url:
            # Handle both URL patterns:
            # 1. Path-based: https://host/predict -> https://host/predict_stream
            # 2. Subdomain-based: https://x--y-predict.modal.run -> https://x--y-predict-stream.modal.run
            if "/predict" in endpoint_url:
                stream_url = endpoint_url.replace("/predict", "/predict_stream")
            else:
                stream_url = endpoint_url.replace("-predict.", "-predict-stream.")
        else:
            # Construct URL from Modal naming convention
            stream_url = self._construct_endpoint_url(deployment_name, "predict_stream")

        if not stream_url:
            raise MlflowException(
                f"Could not find streaming endpoint URL for deployment '{deployment_name}'. "
                "The deployment may not have a streaming endpoint configured.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Make streaming request
        proxy_auth_headers = self._build_proxy_auth_headers(deployment_name)
        with requests.post(
            stream_url,
            json=inputs,
            stream=True,
            headers={"Accept": "text/event-stream", **proxy_auth_headers},
            timeout=300,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                # Parse SSE format: "data: <json>"
                if not line_str.startswith("data:"):
                    continue

                data = line_str[5:].strip()  # Remove "data:" prefix

                # Check for end of stream
                if data == "[DONE]":
                    return

                yield json.loads(data)


def run_local(
    target_uri: str,
    name: str,
    model_uri: str,
    flavor: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Run a Modal deployment locally using `modal serve`.

    Args:
        target_uri: Target URI (e.g., "modal")
        name: Name for the local deployment
        model_uri: URI of the MLflow model to deploy
        flavor: Model flavor (only python_function supported)
        config: Deployment configuration
    """
    _import_modal()

    with TempDir() as tmp_dir:
        local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())

        deployment_config = ModalDeploymentClient("modal")._default_deployment_config()
        if config:
            deployment_config.update(config)

        if deployment_config.get("python_version") is None:
            detected_version = _get_model_python_version(local_model_path)
            deployment_config["python_version"] = detected_version or "3.10"

        model_requirements, wheel_files = _get_model_requirements(local_model_path)
        if model_requirements:
            _logger.info(f"Detected pip requirements: {model_requirements}")
        if wheel_files:
            _logger.info(f"Detected wheel files: {[os.path.basename(w) for w in wheel_files]}")

        wheel_filenames = [os.path.basename(w) for w in wheel_files] if wheel_files else None
        app_code = _generate_modal_app_code(name, deployment_config, model_requirements, wheel_filenames)

        app_file = os.path.join(tmp_dir.path(), "modal_app.py")
        with open(app_file, "w") as f:
            f.write(app_code)

        _logger.info(f"Starting local Modal server for {name}...")
        subprocess.run(["modal", "serve", app_file], cwd=tmp_dir.path())


def target_help() -> str:
    """Return help text for the Modal deployment target."""
    return """
    MLflow Modal Deployment Plugin
    ==============================

    Deploy MLflow models to Modal's serverless platform (https://modal.com).

    Installation
    ------------
    pip install mlflow-modal-deploy

    Target URI Format
    -----------------
    - ``modal``: Use default workspace from Modal authentication
    - ``modal:/workspace-name``: Use a specific Modal environment/workspace

    Authentication
    --------------
    Modal authentication is handled via the Modal CLI or environment variables:
    - Run ``modal setup`` to configure authentication interactively
    - Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables

    Configuration Options
    ---------------------
    Pass these options via the ``config`` parameter in create_deployment():

    Resource Configuration:
    - ``gpu``: GPU type (T4, L4, L40S, A10, A100, A100-40GB, A100-80GB, H100, H200, B200, RTX-PRO-6000)
              Supports multi-GPU ("H100:8"), fallback lists (["H100", "A100"]),
              dedicated suffix ("H100!"), and upgrade suffix ("B200+")
    - ``memory``: Memory allocation in MB (default: 512)
    - ``cpu``: CPU cores (default: 1.0)
    - ``timeout``: Request timeout in seconds (default: 300)
    - ``proxy_auth``: Enable proxy authentication in modal's ENDPOINT URL (default: False)

    Scaling Configuration:
    - ``min_containers``: Minimum containers to keep warm (default: 0)
    - ``max_containers``: Maximum containers to scale to (default: None)
    - ``scaledown_window``: Seconds before idle container scales down (default: 60)
    - ``concurrent_inputs``: Max concurrent inputs per container (default: 1)

    Batching Configuration:
    - ``enable_batching``: Enable dynamic batching (default: False)
    - ``max_batch_size``: Maximum batch size when batching enabled (default: 8)
    - ``batch_wait_ms``: Batch wait time in milliseconds (default: 100)

    Private PyPI Configuration:
    - ``extra_pip_packages``: Additional pip packages to install
    - ``pip_index_url``: Custom PyPI index URL for private packages
    - ``pip_extra_index_url``: Additional PyPI index URL
    - ``modal_secret``: Modal secret name containing PyPI credentials

    Example
    -------
    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("modal")

        deployment = client.create_deployment(
            name="my-classifier",
            model_uri="runs:/abc123/model",
            config={
                "gpu": "T4",
                "memory": 2048,
                "min_containers": 1,
                "enable_batching": True,
            }
        )

        predictions = client.predict(
            deployment_name="my-classifier",
            inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
        )

    Streaming (for LLM models)
    --------------------------
    .. code-block:: python

        # Stream responses for LLM deployments (compatible with Databricks API)
        for chunk in client.predict_stream(
            deployment_name="my-llm",
            inputs={"messages": [{"role": "user", "content": "Hello!"}]}
        ):
            print(chunk.get("text", ""), end="", flush=True)

    CLI Usage
    ---------
    .. code-block:: bash

        mlflow deployments create -t modal -m runs:/abc123/model --name my-model
        mlflow deployments list -t modal
        mlflow deployments get -t modal --name my-model
        mlflow deployments delete -t modal --name my-model
    """
