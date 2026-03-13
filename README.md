# mlflow-modal-deploy

[![CI](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/ci.yml/badge.svg)](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/ci.yml)
[![CodeQL](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/codeql.yml/badge.svg)](https://github.com/debu-sinha/mlflow-modal-deploy/actions/workflows/codeql.yml)
[![PyPI version](https://img.shields.io/pypi/v/mlflow-modal-deploy)](https://pypi.org/project/mlflow-modal-deploy/)
[![Downloads](https://static.pepy.tech/badge/mlflow-modal-deploy)](https://pepy.tech/project/mlflow-modal-deploy)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

Deploy MLflow models to [Modal](https://modal.com)'s serverless GPU infrastructure with a single command.

> **If you find this project useful, please consider giving it a star!** It helps others discover the project and motivates continued development. Using it in production? [Share your experience](https://github.com/debu-sinha/mlflow-modal-deploy/issues) - we'd love to hear from you!

## Installation

```bash
pip install mlflow-modal-deploy
```

## Features

- **One-command deployment**: Deploy any MLflow model to Modal's serverless infrastructure
- **GPU support**: T4, L4, L40S, A10, A10G, A100, A100-40GB, A100-80GB, H100, H200, B200, RTX-PRO-6000
- **Streaming predictions**: `predict_stream()` API compatible with MLflow Databricks client
- **Auto-scaling**: Configure min/max containers, scale-down windows
- **Dynamic batching**: Built-in request batching for high-throughput workloads
- **Automatic dependency detection**: Extracts requirements from model artifacts
- **Wheel file support**: Handles private dependencies packaged as wheel files
- **Private PyPI support**: Deploy with private packages via `pip_index_url` or Modal secrets
- **MLflow CLI integration**: Use familiar `mlflow deployments` commands

## How it Works

```
MLflow Model -> Extract Dependencies -> Modal Volume -> Generate Modal App -> HTTPS Endpoint
```

1. **Extract**: MLflow model artifacts and dependencies are extracted from the model URI
2. **Upload**: Model files are uploaded to a Modal Volume for persistent storage
3. **Generate**: A Modal app is generated with FastAPI endpoints (`/invocations`, `/predict_stream`)
4. **Deploy**: Modal builds a container with all dependencies and deploys to serverless infrastructure
5. **Serve**: An HTTPS endpoint URL is returned, ready to handle prediction requests

The generated container mirrors your training environment, ensuring consistent behavior between development and production.

## Quick Start

### Python API

```python
from mlflow.deployments import get_deploy_client

# Get the Modal deployment client
client = get_deploy_client("modal")

# Deploy a model
deployment = client.create_deployment(
    name="my-classifier",
    model_uri="runs:/abc123/model",
    config={
        "gpu": "T4",
        "memory": 2048,
        "min_containers": 1,
    }
)

print(f"Deployed to: {deployment['endpoint_url']}")

# Make predictions
predictions = client.predict(
    deployment_name="my-classifier",
    inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
)
```

### CLI

```bash
# Deploy a model
mlflow deployments create -t modal -m runs:/abc123/model --name my-model

# Deploy with GPU
mlflow deployments create -t modal -m runs:/abc123/model --name gpu-model \
    -C gpu=T4 -C memory=4096

# List deployments
mlflow deployments list -t modal

# Get deployment info
mlflow deployments get -t modal --name my-model

# Delete deployment
mlflow deployments delete -t modal --name my-model
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu` | str/list | None | GPU type (T4, L4, L40S, A10, A10G, A100, A100-40GB, A100-80GB, H100, H200, B200, RTX-PRO-6000), multi-GPU (`H100:8`), dedicated (`H100!`), upgrade fallback (`B200+`), or fallback list (`["H100", "A100"]`) |
| `memory` | int | 512 | Memory allocation in MB |
| `cpu` | float | 1.0 | CPU cores |
| `timeout` | int | 300 | Request timeout in seconds |
| `startup_timeout` | int | None | Container startup timeout (overrides timeout during model loading) |
| `scaledown_window` | int | 60 | Seconds before idle container scales down |
| `concurrent_inputs` | int | 1 | Max concurrent requests per container |
| `target_inputs` | int | None | Target concurrency for autoscaler (enables smarter scaling) |
| `min_containers` | int | 0 | Minimum warm containers |
| `max_containers` | int | None | Maximum containers |
| `buffer_containers` | int | None | Extra idle containers to maintain under load |
| `enable_batching` | bool | False | Enable dynamic batching |
| `max_batch_size` | int | 8 | Max batch size when batching enabled |
| `batch_wait_ms` | int | 100 | Batch wait time in milliseconds |
| `python_version` | str | auto | Python version (auto-detected from model) |
| `extra_pip_packages` | list | [] | Additional pip packages to install at deployment time |
| `pip_index_url` | str | None | Custom PyPI index URL for private packages |
| `pip_extra_index_url` | str | None | Additional PyPI index URL (fallback) |
| `modal_secret` | str | None | Modal secret name containing pip credentials |
| `proxy_auth` | bool | False | Enable proxy auth protection for modal endpoint |

## Authentication

Configure Modal authentication before deploying:

```bash
# Interactive setup
modal setup

# Or use environment variables
export MODAL_TOKEN_ID=your-token-id
export MODAL_TOKEN_SECRET=your-token-secret
```

## Local Testing (Recommended)

Before deploying to Modal's cloud infrastructure, test your deployment locally to catch issues early:

```python
from mlflow_modal import run_local

run_local(
    target_uri="modal",
    name="test-model",
    model_uri="runs:/abc123/model",
    config={"gpu": "T4"}
)
```

This runs `modal serve` locally, allowing you to verify:
- Model loads correctly with all dependencies
- Inference endpoint responds as expected
- GPU configuration is valid

Once local testing passes, deploy to production with `create_deployment()`.

## Advanced Usage

### Streaming Predictions

For LLM and generative models, use `predict_stream()` for token-by-token streaming responses. This API is compatible with MLflow's Databricks client, enabling consistent code across deployment targets.

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("modal")

# Stream predictions (for LLM models)
for chunk in client.predict_stream(
    deployment_name="my-llm",
    inputs={
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "max_tokens": 100,
    },
):
    print(chunk, end="", flush=True)
```

**How it works:**
- Models with native `predict_stream()` support (LLMs) stream token-by-token
- Non-streaming models (sklearn, XGBoost, etc.) return predictions in a single chunk
- Uses Server-Sent Events (SSE) format for efficient streaming over HTTP

### Deploy to Specific Workspace

```python
# Use workspace-specific URI
client = get_deploy_client("modal:/production")
```

Or via CLI:

```bash
mlflow deployments create -t modal:/production -m runs:/abc123/model --name my-model
```

### High-Throughput Deployment with Batching

```python
client.create_deployment(
    name="batch-classifier",
    model_uri="runs:/abc123/model",
    config={
        "gpu": "A100",
        "enable_batching": True,
        "max_batch_size": 32,
        "batch_wait_ms": 50,
        "min_containers": 2,
        "max_containers": 20,
    }
)
```

### Adding Extra Packages at Deployment Time

Use `extra_pip_packages` when the model's auto-detected requirements are incomplete or you need production-specific packages:

```python
client.create_deployment(
    name="my-model",
    model_uri="runs:/abc123/model",
    config={
        "gpu": "A100",
        "extra_pip_packages": [
            "accelerate>=0.24",      # GPU inference optimization
            "prometheus_client",     # Monitoring
            "structlog",             # Production logging
        ],
    }
)
```

Common use cases:
- **Missing transitive dependencies**: Packages MLflow didn't auto-detect
- **Inference optimizations**: `accelerate`, `bitsandbytes`, `onnxruntime-gpu`
- **Production monitoring**: `prometheus_client`, `opentelemetry-api`
- **Version overrides**: Pin specific versions for compatibility

### Deploying with Private Packages

For private PyPI servers or authenticated package repositories:

**Step 1**: Create a Modal secret with your credentials:

```bash
# Create a secret with your private PyPI credentials
modal secret create pypi-auth \
    PIP_INDEX_URL="https://user:token@pypi.my-company.com/simple/" \
    PIP_EXTRA_INDEX_URL="https://pypi.org/simple/"
```

**Step 2**: Reference the secret in your deployment:

```python
client.create_deployment(
    name="my-model",
    model_uri="runs:/abc123/model",
    config={
        # Option 1: Use Modal secret for authenticated access
        "modal_secret": "pypi-auth",
        "extra_pip_packages": ["my-private-package>=1.0"],

        # Option 2: Direct URL (for unauthenticated private repos)
        # "pip_index_url": "https://pypi.my-company.com/simple/",
        # "pip_extra_index_url": "https://pypi.org/simple/",
    }
)
```

Supported private package sources:
- **Private PyPI servers**: Artifactory, CodeArtifact, DevPI, Nexus
- **Authenticated indexes**: Any pip-compatible index with auth tokens
- **Wheel files**: Already supported via the `code/` directory in model artifacts

### Models with Private Dependencies

If your model includes wheel files in the `code/` directory, they are automatically detected and installed:

```
model/
├── MLmodel
├── requirements.txt
├── code/
│   └── my_private_package-1.0.0-py3-none-any.whl  # Auto-detected
└── ...
```

### Deploying with Proxy Authentication Enabled
Enables [proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth#proxy-auth-tokens) in modal's ENDPOINT URL.

Apps deployed without proxy authentication enabled are public to anyone with knowledge of the endpoint to make api requests, it can be hit by any client over the Internet. With proxy authentication enabled, Modal's authentication feature only allows users with access to make requests.

```python
# Deploy model
client.create_deployment(
    name="my-classifier",
    model_uri="runs:/abc123/model",
    config={
        "proxy_auth": True,
    }
)
```

```python
import os

# Set an environment variable (if are not set)
os.environ['PROXY_AUTH_TOKEN_ID'] = 'your_api_key_here'
os.environ['PROXY_AUTH_TOKEN_SECRET'] = 'your_secret_here'

# Make predictions
predictions = client.predict(
    deployment_name="my-classifier",
    inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
    proxy_auth_enabled=True
)
```

or

```sh
export PROXY_AUTH_TOKEN_ID=your_api_key_here
export PROXY_AUTH_TOKEN_SECRET=your_secret_here
curl -H "Modal-Key: $PROXY_AUTH_TOKEN_ID" \
     -H "Modal-Secret: $PROXY_AUTH_TOKEN_SECRET" \
     https://private-url--goes-here.modal.run
```

## Troubleshooting

### Modal Authentication Fails

```bash
# Re-authenticate with Modal
modal setup

# Verify authentication
modal profile list
```

### "MLmodel not found" Error

- Ensure model was logged with `mlflow.pyfunc.log_model()` or similar MLflow logging function
- Verify the model URI is correct: `runs:/<run_id>/model` or `models:/<name>/<version>`
- Check that the model directory contains an `MLmodel` file

### Deployment Times Out

For large models that take longer to load:

```python
client.create_deployment(
    name="large-model",
    model_uri="runs:/abc123/model",
    config={
        "startup_timeout": 600,  # 10 minutes for model loading
        "timeout": 300,          # 5 minutes for inference requests
    }
)
```

### Missing Dependencies at Runtime

If the model fails with import errors:

```python
client.create_deployment(
    name="my-model",
    model_uri="runs:/abc123/model",
    config={
        "extra_pip_packages": ["missing-package>=1.0"],
    }
)
```

### View Build Logs

Check the [Modal Dashboard](https://modal.com/apps) for detailed build and runtime logs.

## Requirements

- Python 3.10+
- MLflow 2.10.0+
- Modal 1.0.0+

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/debu-sinha/mlflow-modal-deploy/blob/main/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/debu-sinha/mlflow-modal-deploy.git
cd mlflow-modal-deploy

# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v
```

## License

Apache License 2.0

## Acknowledgments

- [MLflow](https://mlflow.org/) - Open source platform for the ML lifecycle
- [Modal](https://modal.com/) - Serverless cloud for AI/ML

## Useful Links

- [Modal Documentation](https://modal.com/docs) - Modal platform docs and tutorials
- [MLflow Deployment Guide](https://mlflow.org/docs/latest/deployment/index.html) - MLflow deployment concepts
- [MLflow Model Format](https://mlflow.org/docs/latest/models.html) - Understanding MLflow models
- [Modal GPU Guide](https://modal.com/docs/guide/gpu) - GPU types and configuration

## Support

- [GitHub Issues](https://github.com/debu-sinha/mlflow-modal-deploy/issues) - Bug reports and feature requests
- [MLflow Slack](https://mlflow.org/slack) - Community discussion
- [Modal Community](https://modal.com/slack) - Modal-specific questions
