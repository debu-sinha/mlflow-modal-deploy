"""Tests for MLflow Modal deployment client."""

import pytest

import mlflow_modal
from mlflow_modal.deployment import (
    SUPPORTED_GPUS,
    ModalDeploymentClient,
    _escape_string_for_codegen,
    _generate_modal_app_code,
    _get_model_python_version,
    _get_model_requirements,
    _get_preferred_deployment_flavor,
    _sanitize_deployment_name,
    _validate_deployment_flavor,
    target_help,
)


class TestModuleExports:
    def test_version_exported(self):
        assert hasattr(mlflow_modal, "__version__")
        assert mlflow_modal.__version__  # Just verify it's set

    def test_client_exported(self):
        assert hasattr(mlflow_modal, "ModalDeploymentClient")
        assert mlflow_modal.ModalDeploymentClient is ModalDeploymentClient

    def test_supported_gpus_exported(self):
        assert mlflow_modal.SUPPORTED_GPUS == SUPPORTED_GPUS


class TestSupportedGPUs:
    @pytest.mark.parametrize(
        "gpu",
        ["T4", "L4", "L40S", "A10", "A100", "A100-40GB", "A100-80GB", "H100", "H200", "B200", "RTX-PRO-6000"],
    )
    def test_gpu_in_supported_list(self, gpu):
        assert gpu in SUPPORTED_GPUS


class TestClientURIParsing:
    @pytest.mark.parametrize(
        "uri,expected_workspace",
        [
            ("modal", None),
            ("modal:/", None),
            ("modal:/my-workspace", "my-workspace"),
            ("modal:/prod-env", "prod-env"),
        ],
    )
    def test_workspace_parsing(self, uri, expected_workspace):
        client = ModalDeploymentClient(uri)
        assert client.workspace == expected_workspace


class TestDefaultConfig:
    def test_default_config_values(self):
        client = ModalDeploymentClient("modal")
        config = client._default_deployment_config()

        assert config["gpu"] is None
        assert config["memory"] == 512
        assert config["cpu"] == 1.0
        assert config["timeout"] == 300
        assert config["scaledown_window"] == 60
        assert config["concurrent_inputs"] == 1
        assert config["enable_batching"] is False
        assert config["max_batch_size"] == 8
        assert config["min_containers"] == 0


class TestConfigValidation:
    def test_apply_custom_config_type_conversion(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        custom = {"memory": "2048", "cpu": "2.0", "enable_batching": "true"}
        result = client._apply_custom_config(base_config, custom)

        assert result["memory"] == 2048
        assert isinstance(result["memory"], int)
        assert result["cpu"] == 2.0
        assert isinstance(result["cpu"], float)
        assert result["enable_batching"] is True
        assert isinstance(result["enable_batching"], bool)

    def test_invalid_gpu_raises_error(self):
        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        with pytest.raises(MlflowException, match="Unsupported GPU type"):
            client._apply_custom_config(base_config, {"gpu": "INVALID_GPU"})

    @pytest.mark.parametrize("gpu", SUPPORTED_GPUS)
    def test_valid_gpus_accepted(self, gpu):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": gpu})
        assert result["gpu"] == gpu

    def test_multi_gpu_syntax_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": "H100:8"})
        assert result["gpu"] == "H100:8"

    def test_gpu_fallback_list_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": ["H100", "A100"]})
        assert result["gpu"] == ["H100", "A100"]

    def test_backward_compat_container_idle_timeout(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"container_idle_timeout": 120})
        assert result["scaledown_window"] == 120

    def test_backward_compat_allow_concurrent_inputs(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"allow_concurrent_inputs": 5})
        assert result["concurrent_inputs"] == 5

    def test_new_modal_1_0_params_in_config(self):
        client = ModalDeploymentClient("modal")
        config = client._default_deployment_config()

        assert "startup_timeout" in config
        assert "target_inputs" in config
        assert "buffer_containers" in config
        assert config["startup_timeout"] is None
        assert config["target_inputs"] is None
        assert config["buffer_containers"] is None

    def test_dedicated_gpu_syntax_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": "H100!"})
        assert result["gpu"] == "H100!"

    def test_upgrade_gpu_suffix_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": "B200+"})
        assert result["gpu"] == "B200+"

    def test_upgrade_gpu_suffix_with_count_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": "B200+:4"})
        assert result["gpu"] == "B200+:4"

    def test_rtx_pro_6000_accepted(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        result = client._apply_custom_config(base_config, {"gpu": "RTX-PRO-6000"})
        assert result["gpu"] == "RTX-PRO-6000"


class TestModelRequirements:
    def test_empty_model_path_returns_empty_tuple(self, tmp_path):
        requirements, wheel_files = _get_model_requirements(str(tmp_path))
        assert requirements == []
        assert wheel_files == []

    def test_requirements_txt_parsing(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.24.0\npandas>=2.0\n# comment\nmlflow==2.10.0\nscikit-learn")

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas>=2.0" in requirements
        assert "scikit-learn" in requirements
        assert not any("mlflow" in r.lower() for r in requirements)
        assert wheel_files == []

    def test_conda_yaml_parsing(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - pip:
    - numpy==1.24.0
    - mlflow==2.10.0
    - pandas
"""
        )

        requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas" in requirements
        assert not any("mlflow" in r.lower() for r in requirements)
        assert wheel_files == []

    def test_wheel_files_detected(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "my_package-1.0.0-py3-none-any.whl").write_text("fake wheel")
        (code_dir / "other_pkg-2.0.0-py3-none-any.whl").write_text("fake wheel")

        _requirements, wheel_files = _get_model_requirements(str(tmp_path))

        assert len(wheel_files) == 2
        assert any("my_package" in w for w in wheel_files)
        assert any("other_pkg" in w for w in wheel_files)


class TestPythonVersionDetection:
    def test_no_conda_yaml_returns_none(self, tmp_path):
        result = _get_model_python_version(str(tmp_path))
        assert result is None

    @pytest.mark.parametrize(
        "python_spec,expected",
        [
            ("python=3.10.0", "3.10"),
            ("python=3.11", "3.11"),
            ("python>=3.9", "3.9"),
        ],
    )
    def test_python_version_extraction(self, tmp_path, python_spec, expected):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(f"dependencies:\n  - {python_spec}\n")

        result = _get_model_python_version(str(tmp_path))
        assert result == expected


class TestAppCodeGeneration:
    def test_basic_app_generation(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("test-app", config)

        assert 'app = modal.App("test-app")' in code
        assert "gpu=None" in code
        assert "memory=512" in code
        assert "@modal.fastapi_endpoint" in code
        assert "def predict" in code
        assert ".uv_pip_install" in code

    def test_gpu_config_in_generated_code(self):
        config = {
            "gpu": "T4",
            "memory": 2048,
            "cpu": 2.0,
            "timeout": 600,
            "scaledown_window": 120,
            "enable_batching": False,
            "python_version": "3.11",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("gpu-app", config)

        assert 'gpu="T4"' in code
        assert "memory=2048" in code
        assert 'python_version="3.11"' in code

    def test_batching_enabled_generates_batch_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": True,
            "max_batch_size": 16,
            "batch_wait_ms": 200,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("batch-app", config)

        assert "@modal.batched" in code
        assert "max_batch_size=16" in code
        assert "wait_ms=200" in code
        assert "def predict_batch" in code

    def test_model_requirements_included(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }
        requirements = ["numpy==1.24.0", "pandas>=2.0"]

        code = _generate_modal_app_code("req-app", config, requirements)

        assert '"mlflow"' in code
        assert '"numpy==1.24.0"' in code
        assert '"pandas>=2.0"' in code

    def test_scaling_config_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 300,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 2,
            "max_containers": 10,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("scale-app", config)

        assert "min_containers=2" in code
        assert "max_containers=10" in code
        assert "scaledown_window=300" in code

    def test_concurrent_inputs_generates_decorator(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 5,
        }

        code = _generate_modal_app_code("concurrent-app", config)

        assert "@modal.concurrent(max_inputs=5)" in code

    def test_concurrent_inputs_default_no_decorator(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("no-concurrent-app", config)

        assert "@modal.concurrent" not in code

    def test_wheel_installation_code_generated(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }
        wheel_filenames = ["my_package-1.0.0-py3-none-any.whl", "other-2.0.0-py3-none-any.whl"]

        code = _generate_modal_app_code("wheel-app", config, None, wheel_filenames)

        assert "Install wheel dependencies from volume" in code
        assert "/model/wheels/my_package-1.0.0-py3-none-any.whl" in code
        assert "/model/wheels/other-2.0.0-py3-none-any.whl" in code

    def test_gpu_fallback_list_in_generated_code(self):
        config = {
            "gpu": ["H100", "A100-80GB"],
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("fallback-gpu-app", config)

        assert 'gpu=["H100", "A100-80GB"]' in code

    def test_target_inputs_generates_decorator(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "target_inputs": 3,
        }

        code = _generate_modal_app_code("target-inputs-app", config)

        assert "@modal.concurrent(target_inputs=3)" in code

    def test_buffer_containers_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 2,
            "max_containers": 10,
            "buffer_containers": 3,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("buffer-app", config)

        assert "buffer_containers=3" in code

    def test_startup_timeout_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "startup_timeout": 600,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("startup-timeout-app", config)

        assert "startup_timeout=600" in code


class TestClientInstance:
    def test_client_is_real_instance(self):
        from mlflow.deployments import BaseDeploymentClient

        client = ModalDeploymentClient("modal")
        assert isinstance(client, BaseDeploymentClient)
        assert isinstance(client, ModalDeploymentClient)

    def test_target_uri_stored(self):
        client = ModalDeploymentClient("modal:/production")
        assert client.target_uri == "modal:/production"
        assert client.workspace == "production"


class TestFlavorValidation:
    def test_get_preferred_flavor_with_pyfunc(self):
        from unittest.mock import MagicMock

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}, "sklearn": {}}

        result = _get_preferred_deployment_flavor(model_config)
        assert result == "python_function"

    def test_get_preferred_flavor_without_pyfunc_raises(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"sklearn": {}}

        with pytest.raises(MlflowException, match="does not contain the python_function flavor"):
            _get_preferred_deployment_flavor(model_config)

    def test_validate_deployment_flavor_valid(self):
        from unittest.mock import MagicMock

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}}

        _validate_deployment_flavor(model_config, "python_function")

    def test_validate_deployment_flavor_unsupported(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"python_function": {}}

        with pytest.raises(MlflowException, match="not supported for Modal deployment"):
            _validate_deployment_flavor(model_config, "sklearn")

    def test_validate_deployment_flavor_missing(self):
        from unittest.mock import MagicMock

        from mlflow.exceptions import MlflowException

        model_config = MagicMock()
        model_config.flavors = {"sklearn": {}}

        with pytest.raises(MlflowException, match="does not contain"):
            _validate_deployment_flavor(model_config, "python_function")


class TestCondaYamlEdgeCases:
    def test_conda_yaml_with_non_pip_dependencies(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - numpy=1.24.0
  - scipy
"""
        )

        requirements, _wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy=1.24.0" in requirements
        assert "scipy" in requirements

    def test_conda_yaml_filters_wheel_references(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text(
            """
name: test-env
dependencies:
  - python=3.10
  - pip:
    - numpy==1.24.0
    - ./code/custom-1.0.0-py3-none-any.whl
"""
        )

        requirements, _wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert not any(".whl" in r for r in requirements)

    def test_requirements_txt_filters_wheel_references(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy==1.24.0\ncode/custom-1.0.0.whl\npandas")

        requirements, _wheel_files = _get_model_requirements(str(tmp_path))

        assert "numpy==1.24.0" in requirements
        assert "pandas" in requirements
        assert not any(".whl" in r for r in requirements)
        assert not any("code/" in r for r in requirements)


class TestPythonVersionEdgeCases:
    def test_python_version_with_less_than(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text("dependencies:\n  - python<3.12\n")

        result = _get_model_python_version(str(tmp_path))
        assert result == "3.12"

    def test_python_version_missing_in_dependencies(self, tmp_path):
        conda_file = tmp_path / "conda.yaml"
        conda_file.write_text("dependencies:\n  - numpy\n")

        result = _get_model_python_version(str(tmp_path))
        assert result is None


class TestTargetHelp:
    def test_target_help_returns_string(self):
        result = target_help()
        assert isinstance(result, str)

    def test_target_help_contains_usage_info(self):
        result = target_help()
        assert "modal" in result.lower()
        assert "gpu" in result.lower()
        assert "memory" in result.lower()
        assert "pip install mlflow-modal-deploy" in result

    def test_target_help_contains_examples(self):
        result = target_help()
        assert "get_deploy_client" in result
        assert "create_deployment" in result

    def test_target_help_contains_streaming_docs(self):
        result = target_help()
        assert "predict_stream" in result
        assert "Streaming" in result

    def test_target_help_contains_private_pypi_docs(self):
        result = target_help()
        assert "pip_index_url" in result
        assert "pip_extra_index_url" in result
        assert "modal_secret" in result
        assert "extra_pip_packages" in result


class TestAppCodeGenerationEdgeCases:
    def test_no_scaling_config_when_defaults(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("default-app", config)

        assert "min_containers=0" not in code
        assert "max_containers=" not in code or "max_containers=None" not in code

    def test_empty_requirements_list(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("no-req-app", config, [])

        assert ".uv_pip_install" in code
        assert '"mlflow"' in code

    def test_no_wheel_code_when_none(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("no-wheel-app", config, None, None)

        assert "Install wheel dependencies" not in code

    def test_extra_pip_packages_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "extra_pip_packages": ["transformers>=4.30", "torch==2.0.0"],
        }

        code = _generate_modal_app_code("extra-deps-app", config)

        assert '"transformers>=4.30"' in code
        assert '"torch==2.0.0"' in code
        assert '"mlflow"' in code

    def test_extra_pip_packages_combined_with_model_requirements(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "extra_pip_packages": ["transformers>=4.30"],
        }
        model_requirements = ["numpy==1.24.0", "pandas>=2.0"]

        code = _generate_modal_app_code("combined-deps-app", config, model_requirements)

        assert '"mlflow"' in code
        assert '"numpy==1.24.0"' in code
        assert '"pandas>=2.0"' in code
        assert '"transformers>=4.30"' in code


class TestGeneratedCodeSyntax:
    """Validate that all generated code is syntactically valid Python.

    This catches indentation errors, unclosed brackets, and other issues
    that string-presence assertions miss.
    """

    @pytest.mark.parametrize(
        "config_overrides",
        [
            {},
            {"enable_batching": True, "max_batch_size": 16},
            {"gpu": "T4"},
            {"gpu": ["H100", "A100-80GB"]},
            {"modal_secret": "pip-credentials"},
            {"concurrent_inputs": 5},
            {"wheel_filenames": ["pkg.whl"]},
            {"pip_index_url": "https://pypi.example.com/simple/"},
            {"min_containers": 2, "max_containers": 10, "scaledown_window": 120},
        ],
        ids=[
            "defaults",
            "batching",
            "single_gpu",
            "multi_gpu",
            "secret",
            "concurrent",
            "wheels",
            "pip_index",
            "scaling",
        ],
    )
    def test_generated_code_is_valid_python(self, config_overrides):
        import ast

        base_config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }
        base_config.update(config_overrides)

        code = _generate_modal_app_code("syntax-test", base_config)

        # This will raise SyntaxError if the generated code has
        # indentation issues, unclosed brackets, or other syntax problems
        ast.parse(code)


class TestExtraPipPackagesConfig:
    def test_extra_pip_packages_in_default_config(self):
        client = ModalDeploymentClient("modal")
        config = client._default_deployment_config()

        assert "extra_pip_packages" in config
        assert config["extra_pip_packages"] == []

    def test_extra_pip_packages_custom_config(self):
        client = ModalDeploymentClient("modal")
        base_config = client._default_deployment_config()

        custom = {"extra_pip_packages": ["torch", "transformers"]}
        result = client._apply_custom_config(base_config, custom)

        assert result["extra_pip_packages"] == ["torch", "transformers"]


class TestPrivateRepoConfig:
    def test_private_repo_options_in_default_config(self):
        client = ModalDeploymentClient("modal")
        config = client._default_deployment_config()

        assert "pip_index_url" in config
        assert "pip_extra_index_url" in config
        assert "modal_secret" in config
        assert config["pip_index_url"] is None
        assert config["pip_extra_index_url"] is None
        assert config["modal_secret"] is None

    def test_pip_index_url_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "pip_index_url": "https://pypi.my-company.com/simple/",
        }

        code = _generate_modal_app_code("private-repo-app", config)

        assert 'index_url="https://pypi.my-company.com/simple/"' in code

    def test_pip_extra_index_url_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "pip_extra_index_url": "https://private.pypi.org/simple/",
        }

        code = _generate_modal_app_code("extra-index-app", config)

        assert 'extra_index_url="https://private.pypi.org/simple/"' in code

    def test_modal_secret_in_generated_code(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "modal_secret": "pip-credentials",
        }

        code = _generate_modal_app_code("secret-app", config)

        assert 'modal.Secret.from_name("pip-credentials")' in code
        assert "secrets=[pip_secret]" in code

    def test_combined_private_repo_config(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
            "pip_index_url": "https://pypi.my-company.com/simple/",
            "pip_extra_index_url": "https://pypi.python.org/simple/",
            "modal_secret": "pypi-auth",
            "extra_pip_packages": ["my-private-package>=1.0"],
        }

        code = _generate_modal_app_code("full-private-app", config)

        assert 'index_url="https://pypi.my-company.com/simple/"' in code
        assert 'extra_index_url="https://pypi.python.org/simple/"' in code
        assert 'modal.Secret.from_name("pypi-auth")' in code
        assert '"my-private-package>=1.0"' in code


class TestSecurityValidation:
    """Tests for input validation and security measures."""

    def test_sanitize_valid_name(self):
        assert _sanitize_deployment_name("my-model") == "my-model"
        assert _sanitize_deployment_name("Model123") == "Model123"
        assert _sanitize_deployment_name("test_deployment") == "test_deployment"

    def test_sanitize_empty_name_raises(self):
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="cannot be empty"):
            _sanitize_deployment_name("")

    def test_sanitize_too_long_name_raises(self):
        from mlflow.exceptions import MlflowException

        long_name = "a" * 64
        with pytest.raises(MlflowException, match="exceeds 63 character limit"):
            _sanitize_deployment_name(long_name)

    def test_sanitize_invalid_chars_raises(self):
        from mlflow.exceptions import MlflowException

        invalid_names = [
            "my model",  # space
            "my;model",  # semicolon
            "my'model",  # quote
            'my"model',  # double quote
            "123model",  # starts with number
            "-mymodel",  # starts with hyphen
        ]
        for name in invalid_names:
            with pytest.raises(MlflowException, match="Invalid deployment name"):
                _sanitize_deployment_name(name)

    def test_escape_string_basic(self):
        assert _escape_string_for_codegen("hello") == "hello"
        assert _escape_string_for_codegen("https://pypi.org/") == "https://pypi.org/"

    def test_escape_string_quotes(self):
        assert _escape_string_for_codegen('test"quote') == 'test\\"quote'

    def test_escape_string_backslash(self):
        assert _escape_string_for_codegen("path\\to\\file") == "path\\\\to\\\\file"

    def test_escape_string_newline(self):
        assert _escape_string_for_codegen("line1\nline2") == "line1\\nline2"

    def test_escape_string_injection_attempt(self):
        malicious = 'https://evil.com"); import os; os.system("rm -rf /'
        escaped = _escape_string_for_codegen(malicious)
        # Verify quotes are escaped, preventing code injection
        assert '\\"' in escaped
        # The escaped string should be safe when used in f'index_url="{escaped}"'
        # because the quotes are escaped as \"
        assert escaped == 'https://evil.com\\"); import os; os.system(\\"rm -rf /'


class TestStreamingEndpointGeneration:
    """Tests for streaming endpoint in generated Modal app code."""

    def test_streaming_endpoint_generated(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("stream-app", config)

        assert "def predict_stream" in code
        assert "StreamingResponse" in code
        assert "text/event-stream" in code
        assert "data: [DONE]" in code

    def test_streaming_endpoint_with_batching(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": True,
            "max_batch_size": 8,
            "batch_wait_ms": 100,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("batch-stream-app", config)

        # Both regular and streaming endpoints should be present
        assert "def predict" in code
        assert "def predict_stream" in code
        assert "@modal.batched" in code

    def test_streaming_endpoint_checks_model_capability(self):
        config = {
            "gpu": None,
            "memory": 512,
            "cpu": 1.0,
            "timeout": 300,
            "scaledown_window": 60,
            "enable_batching": False,
            "python_version": "3.10",
            "min_containers": 0,
            "max_containers": None,
            "concurrent_inputs": 1,
        }

        code = _generate_modal_app_code("stream-check-app", config)

        # Should check for model's predict_stream capability
        assert "hasattr(self.model, 'predict_stream')" in code


class TestPredictStreamMethod:
    """Tests for predict_stream() method on ModalDeploymentClient."""

    def test_predict_stream_requires_deployment_name(self):
        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")

        with pytest.raises(MlflowException, match="deployment_name is required"):
            list(client.predict_stream(deployment_name=None, inputs={"test": "data"}))

    def test_predict_stream_requires_inputs(self):
        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")

        with pytest.raises(MlflowException, match="inputs is required"):
            list(client.predict_stream(deployment_name="test-model", inputs=None))

    def test_predict_stream_url_construction_from_predict_url(self):
        from unittest.mock import MagicMock, patch

        client = ModalDeploymentClient("modal")

        # Mock get_deployment to return a URL
        mock_deployment = {"endpoint_url": "https://test--app.modal.run/predict"}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch("requests.post") as mock_post,
        ):
            # Mock streaming response
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [b"data: [DONE]"]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_post.return_value = mock_response

            # Consume the generator
            list(client.predict_stream(deployment_name="test", inputs={"test": "data"}))

            # Verify URL was constructed correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://test--app.modal.run/predict_stream"

    def test_predict_stream_parses_sse_format(self):
        from unittest.mock import MagicMock, patch

        client = ModalDeploymentClient("modal")

        mock_deployment = {"endpoint_url": "https://test--app.modal.run/predict"}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch("requests.post") as mock_post,
        ):
            # Mock streaming response with SSE data
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b'data: {"chunk": 1, "text": "Hello"}',
                b'data: {"chunk": 2, "text": " World"}',
                b"data: [DONE]",
            ]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_post.return_value = mock_response

            # Collect results
            results = list(client.predict_stream(deployment_name="test", inputs={"prompt": "Hi"}))

            assert len(results) == 2
            assert results[0] == {"chunk": 1, "text": "Hello"}
            assert results[1] == {"chunk": 2, "text": " World"}

    def test_predict_stream_skips_empty_lines(self):
        from unittest.mock import MagicMock, patch

        client = ModalDeploymentClient("modal")

        mock_deployment = {"endpoint_url": "https://test--app.modal.run/predict"}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch("requests.post") as mock_post,
        ):
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b"",  # Empty line
                b'data: {"text": "test"}',
                b"",  # Another empty line
                b"data: [DONE]",
            ]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_post.return_value = mock_response

            results = list(client.predict_stream(deployment_name="test", inputs={"test": "data"}))

            assert len(results) == 1
            assert results[0] == {"text": "test"}

    def test_predict_stream_skips_non_data_lines(self):
        from unittest.mock import MagicMock, patch

        client = ModalDeploymentClient("modal")

        mock_deployment = {"endpoint_url": "https://test--app.modal.run/predict"}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch("requests.post") as mock_post,
        ):
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b"event: message",  # Non-data line
                b'data: {"text": "test"}',
                b"id: 123",  # Another non-data line
                b"data: [DONE]",
            ]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_post.return_value = mock_response

            results = list(client.predict_stream(deployment_name="test", inputs={"test": "data"}))

            assert len(results) == 1
            assert results[0] == {"text": "test"}

    def test_predict_stream_sends_correct_headers(self):
        from unittest.mock import MagicMock, patch

        client = ModalDeploymentClient("modal")

        mock_deployment = {"endpoint_url": "https://test--app.modal.run/predict"}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch("requests.post") as mock_post,
        ):
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [b"data: [DONE]"]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_post.return_value = mock_response

            list(client.predict_stream(deployment_name="test", inputs={"test": "data"}))

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["headers"]["Accept"] == "text/event-stream"
            assert call_kwargs["stream"] is True

    def test_predict_stream_raises_on_missing_endpoint(self):
        from unittest.mock import patch

        from mlflow.exceptions import MlflowException

        client = ModalDeploymentClient("modal")

        # Mock get_deployment to return no endpoint URL
        mock_deployment = {"endpoint_url": None}

        with (
            patch.object(client, "get_deployment", return_value=mock_deployment),
            patch.object(client, "_get_modal_workspace", return_value=None),
            pytest.raises(MlflowException, match="Could not find streaming endpoint"),
        ):
            list(client.predict_stream(deployment_name="test", inputs={"test": "data"}))
