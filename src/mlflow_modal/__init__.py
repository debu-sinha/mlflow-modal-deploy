"""
MLflow Modal Deployment Plugin

Deploy MLflow models to Modal's serverless infrastructure.
See https://modal.com for more information.
"""

from mlflow_modal.deployment import (
    SUPPORTED_GPUS,
    ModalDeploymentClient,
    run_local,
    target_help,
)

__version__ = "0.6.2"
__all__ = [
    "ModalDeploymentClient",
    "run_local",
    "target_help",
    "SUPPORTED_GPUS",
]
