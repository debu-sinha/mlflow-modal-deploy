# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-06-13

### Fixed

- Generated Modal predict endpoints now coerce input DataFrame column dtypes to match the model's MLflow input schema before calling `model.predict()`. Fixes schema enforcement errors when JSON payloads serialize whole numbers as int64 for double-typed columns ([#20](https://github.com/debu-sinha/mlflow-modal-deploy/issues/20)). Applied to all three generated predict paths: non-batching predict, predict_batch, and the predict_stream fallback. Contributed by [@GuilhermePFM](https://github.com/GuilhermePFM) in [#21](https://github.com/debu-sinha/mlflow-modal-deploy/pull/21).

### Changed

- Bumped `actions/github-script` from 8 to 9 in `api-compatibility.yml`.
- Bumped `softprops/action-gh-release` from 2 to 3 in `release.yml`.
- Bumped `codecov/codecov-action` from 5 to 7 in `ci.yml`.

## [0.6.2] - 2026-03-09

### Fixed
- Fixed GPU validation to handle Modal's `+` upgrade suffix (e.g., `B200+` for fallback to B300)
- Fixed potential `AttributeError` when `result.stderr` is `None` in `delete_deployment()`
- Fixed `.tolist()` calls in generated code to handle predictions that return plain lists instead of numpy arrays

### Added
- Added `RTX-PRO-6000` to supported GPU types
- Added Python 3.13 classifier

### Changed
- Updated `SECURITY.md` supported versions from 0.2.x to 0.6.x

## [0.6.1] - 2026-02-20

### Fixed
- Fixed IndentationError when deploying with `concurrent_inputs > 1` (the `@modal.concurrent` decorator was adding trailing whitespace that broke class indentation in the generated app)
- Fixed SyntaxError when deploying models with many pip dependencies (>10 packages generated a single line too long for the Python parser)

### Added
- Added `A10G` to supported GPU types (Modal's primary GPU variant name alongside existing `A10`)

## [0.6.0] - 2026-01-16

### Added
- `predict_stream()` method for streaming predictions (compatible with MLflow Databricks client API)
- Streaming endpoint (`/predict_stream`) in generated Modal apps with Server-Sent Events (SSE) support
- Automatic model capability detection for streaming (falls back to regular prediction for non-streaming models)
- Helper methods `_get_modal_workspace()` and `_construct_endpoint_url()` for robust endpoint URL resolution
- 13 new tests for streaming functionality (105 total)

### Fixed
- Fixed endpoint URL parsing to prefer regular `predict` endpoint over `predict_stream` in deployment output
- Fixed Modal JSON key parsing (`"Description"` and `"App ID"` instead of `"name"` and `"app_id"`)
- Fixed streaming URL construction to handle both path-based and subdomain-based Modal URL patterns

## [0.5.1] - 2026-01-16

### Security
- Add deployment name validation to prevent code injection attacks
- Add URL/string escaping in generated code to prevent injection via `pip_index_url`, `pip_extra_index_url`, or `modal_secret`
- Add input validation for `predict()` method

### Added
- Weekly API compatibility monitoring workflow (tests against latest Modal/MLflow)
- 9 new security tests (92 total)

## [0.5.0] - 2026-01-16

### Added
- `extra_pip_packages` config option for specifying additional pip packages at deployment time
- Private PyPI server support via `pip_index_url` and `pip_extra_index_url` config options
- Modal secret integration via `modal_secret` config for authenticated private package repositories
- New example `deploy_with_extra_packages.py` demonstrating the feature
- Model file verification before upload to catch issues early
- End-to-end test script for real Modal deployment testing

### Fixed
- Fixed volume upload to use Modal 1.0 `batch_upload` context manager API
- Model files are now correctly uploaded to volume root, fixing "MLmodel not found" errors
- Fixed code generation indentation issues when using `modal_secret` and other optional configs

## [0.4.0] - 2026-01-16

### Added
- `startup_timeout` parameter for separate container startup timeout (useful for large models)
- `target_inputs` parameter for smarter autoscaler targeting in `@modal.concurrent`
- `buffer_containers` parameter for extra idle containers under load
- Dedicated GPU syntax support (`H100!` to prevent auto-upgrade)
- Tests for all new Modal 1.0 parameters

### Changed
- Added `pyyaml` and `requests` as explicit dependencies

## [0.3.1] - 2026-01-16

### Fixed
- Updated README to reflect Modal 1.0 API changes
- Fixed version test to not hardcode version string

## [0.3.0] - 2026-01-16

### Added
- Support for all Modal GPU types: T4, L4, L40S, A10, A100, A100-40GB, A100-80GB, H100, H200, B200
- Multi-GPU syntax support (e.g., `"H100:8"` for 8x H100)
- GPU fallback list support (e.g., `["H100", "A100"]`)
- `concurrent_inputs` parameter for concurrent request handling per container

### Changed
- **BREAKING**: Minimum Modal version is now 1.0.0
- Renamed `container_idle_timeout` to `scaledown_window` (backward compatible)
- Renamed `allow_concurrent_inputs` to `concurrent_inputs` (backward compatible)
- Updated generated code to use `@modal.fastapi_endpoint` instead of deprecated `@modal.web_endpoint`
- Updated generated code to use `.uv_pip_install()` instead of `.pip_install()` for faster builds
- Moved `@modal.concurrent` decorator to class level per Modal 1.0 best practices

### Deprecated
- `container_idle_timeout` parameter (use `scaledown_window` instead)
- `allow_concurrent_inputs` parameter (use `concurrent_inputs` instead)

## [0.2.5] - 2025-01-15

### Fixed
- Switch PyPI badge to shields.io for faster updates

## [0.2.4] - 2025-01-15

### Added
- Added working example in `examples/deploy_sklearn_model.py`
- Added CodeQL badge to README

### Changed
- Updated GitHub Actions to latest versions (checkout v6, setup-uv v7, codecov v5, codeql v4)

## [0.2.3] - 2025-01-15

### Added
- Added `py.typed` marker for PEP 561 type hints support
- Added `SECURITY.md` with vulnerability reporting guidelines
- Added `CHANGELOG.md` following Keep a Changelog format
- Added Dependabot configuration for automated dependency updates
- Added CodeQL workflow for security scanning
- Added automated release workflow for PyPI publishing

### Fixed
- Fixed version inconsistency between `__init__.py` and `pyproject.toml`
- Fixed package name in `target_help()` to correctly reference `mlflow-modal-deploy`

### Removed
- Removed legacy `setup.py` in favor of `pyproject.toml`

## [0.2.2] - 2025-01-15

### Fixed
- Fixed MLflow plugin entry point to use module instead of class reference
- Fixed `run_local` and `target_help` interface requirements

## [0.2.1] - 2025-01-15

### Fixed
- Fixed CONTRIBUTING.md link in README for PyPI rendering

## [0.2.0] - 2025-01-14

### Changed
- Renamed package from `mlflow-modal` to `mlflow-modal-deploy` to avoid naming conflicts

## [0.1.0] - 2025-01-14

### Added
- Initial release
- `ModalDeploymentClient` for deploying MLflow models to Modal
- GPU support: T4, L4, A10G, A100, A100-80GB, H100
- Auto-scaling configuration (min/max containers, idle timeout)
- Dynamic batching support for high-throughput workloads
- Automatic dependency detection from model artifacts
- Wheel file support for private dependencies
- `run_local` function for local testing with `modal serve`
- Full MLflow CLI integration (`mlflow deployments` commands)
- Workspace targeting via URI (`modal:/workspace-name`)

[Unreleased]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/debu-sinha/mlflow-modal-deploy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/debu-sinha/mlflow-modal-deploy/releases/tag/v0.1.0
