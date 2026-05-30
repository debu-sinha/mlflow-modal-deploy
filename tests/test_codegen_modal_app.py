"""Unit tests for the Modal app code generator."""

import pytest

from mlflow_modal.codegen import ModalAppCodeConfig, ModalAppCodeGenerator, generate_modal_app_code


def _make_base_config(**overrides: object) -> ModalAppCodeConfig:
    base: dict[str, object] = {
        "app_name": "test-app",
        "python_version": "3.10",
        "model_requirements": None,
        "extra_pip_packages": None,
        "pip_index_url": None,
        "pip_extra_index_url": None,
        "gpu_config": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "modal_secret": "",
        "wheel_filenames": None,
        "enable_batching": False,
        "max_batch_size": 8,
        "concurrent_inputs": 1,
        "target_inputs": None,
        "batch_wait_ms": 100,
        "min_containers": None,
        "max_containers": None,
        "buffer_containers": None,
        "scaledown_window": None,
        "startup_timeout": None,
        "proxy_auth": False,
    }
    base.update(overrides)
    return ModalAppCodeConfig(**base)  # type: ignore[arg-type]


def _make_generator(**overrides: object) -> ModalAppCodeGenerator:
    return ModalAppCodeGenerator(_make_base_config(**overrides))


# ---------------------------------------------------------------------------
# Overall tests
# ---------------------------------------------------------------------------


class TestModalAppCodeGenerator:
    def test_basic_structure_contains_app_and_class(self) -> None:
        config = _make_base_config()

        code = generate_modal_app_code(config)

        assert 'app = modal.App("test-app")' in code
        assert "class MLflowModel:" in code
        assert "def load_model(self):" in code

    def test_batching_enabled_renders_batched_endpoint(self) -> None:
        config = _make_base_config(enable_batching=True, max_batch_size=16, batch_wait_ms=250)

        code = generate_modal_app_code(config)

        assert "@modal.batched" in code
        assert "max_batch_size=16" in code
        assert "wait_ms=250" in code
        assert "def predict_batch" in code
        assert "return self.predict_batch.local([input_data])[0]" in code

    def test_batching_disabled_renders_simple_predict(self) -> None:
        config = _make_base_config(enable_batching=False)

        code = generate_modal_app_code(config)

        assert "@modal.batched" not in code
        assert "def predict(self, input_data: dict) -> dict" in code
        assert 'return {"predictions": prediction.tolist()' in code

    def test_streaming_endpoint_always_present(self) -> None:
        config = _make_base_config(enable_batching=False)

        code = generate_modal_app_code(config)

        assert "def predict_stream(self, input_data: dict)" in code
        assert "StreamingResponse" in code
        assert "hasattr(self.model, 'predict_stream')" in code

    def test_secrets_and_concurrency_and_wheels_included(self) -> None:
        config = _make_base_config(
            modal_secret="pip-credentials",
            concurrent_inputs=5,
            wheel_filenames=["pkg1.whl"],
        )

        code = generate_modal_app_code(config)

        assert 'modal.Secret.from_name("pip-credentials")' in code
        assert "secrets=[pip_secret]" in code
        assert "@modal.concurrent(max_inputs=5)" in code
        assert "Install wheel dependencies from volume" in code
        assert "/model/wheels/pkg1.whl" in code


# ---------------------------------------------------------------------------
# Per-method unit tests
# ---------------------------------------------------------------------------


class TestRenderGpuStr:
    def test_no_gpu_returns_none_literal(self) -> None:
        gen = _make_generator(gpu_config=None)
        assert gen._render_gpu_str() == "None"

    def test_single_gpu_string(self) -> None:
        gen = _make_generator(gpu_config="T4")
        assert gen._render_gpu_str() == '"T4"'

    def test_multi_gpu_string_treated_as_plain_string(self) -> None:
        gen = _make_generator(gpu_config="H100:8")
        assert gen._render_gpu_str() == '"H100:8"'

    def test_fallback_gpu_list(self) -> None:
        gen = _make_generator(gpu_config=["H100", "A100-80GB"])
        assert gen._render_gpu_str() == '["H100", "A100-80GB"]'

    def test_single_element_list(self) -> None:
        gen = _make_generator(gpu_config=["A10G"])
        assert gen._render_gpu_str() == '["A10G"]'


class TestRenderUvPipInstallStr:
    def test_no_requirements_only_mlflow(self) -> None:
        gen = _make_generator()
        result = gen._render_uv_pip_install_str()
        assert '"mlflow"' in result
        assert result == '"mlflow"'

    def test_model_requirements_appended(self) -> None:
        gen = _make_generator(model_requirements=["numpy==1.24.0", "pandas>=2.0"])
        result = gen._render_uv_pip_install_str()
        assert '"mlflow"' in result
        assert '"numpy==1.24.0"' in result
        assert '"pandas>=2.0"' in result

    def test_extra_pip_packages_appended(self) -> None:
        gen = _make_generator(extra_pip_packages=["torch"])
        result = gen._render_uv_pip_install_str()
        assert '"mlflow"' in result
        assert '"torch"' in result

    def test_both_requirements_and_extras_combined(self) -> None:
        gen = _make_generator(
            model_requirements=["scikit-learn"],
            extra_pip_packages=["torch"],
        )
        result = gen._render_uv_pip_install_str()
        assert '"mlflow"' in result
        assert '"scikit-learn"' in result
        assert '"torch"' in result

    def test_more_than_ten_packages_uses_multiline_format(self) -> None:
        many = [f"pkg{i}" for i in range(11)]
        gen = _make_generator(model_requirements=many)
        result = gen._render_uv_pip_install_str()
        assert "\n" in result
        for pkg in many:
            assert f'"{pkg}"' in result


class TestRenderPipInstallKwargs:
    def test_no_index_urls_returns_empty_string(self) -> None:
        gen = _make_generator()
        assert gen._render_pip_install_kwargs() == ""

    def test_index_url_only(self) -> None:
        gen = _make_generator(pip_index_url="https://pypi.example.com/simple/")
        result = gen._render_pip_install_kwargs()
        assert 'index_url="https://pypi.example.com/simple/"' in result
        assert result.startswith(", ")

    def test_extra_index_url_only(self) -> None:
        gen = _make_generator(pip_extra_index_url="https://extra.pypi.org/simple/")
        result = gen._render_pip_install_kwargs()
        assert 'extra_index_url="https://extra.pypi.org/simple/"' in result
        assert result.startswith(", ")

    def test_both_index_urls_present(self) -> None:
        gen = _make_generator(
            pip_index_url="https://pypi.example.com/simple/",
            pip_extra_index_url="https://extra.pypi.org/simple/",
        )
        result = gen._render_pip_install_kwargs()
        assert 'index_url="https://pypi.example.com/simple/"' in result
        assert 'extra_index_url="https://extra.pypi.org/simple/"' in result


class TestRenderConcurrentDecorator:
    def test_no_concurrency_returns_empty_string(self) -> None:
        gen = _make_generator(concurrent_inputs=1, target_inputs=None)
        assert gen._render_concurrent_decorator() == ""

    def test_max_inputs_only(self) -> None:
        gen = _make_generator(concurrent_inputs=5, target_inputs=None)
        result = gen._render_concurrent_decorator()
        assert "@modal.concurrent(max_inputs=5)" in result

    def test_target_inputs_only(self) -> None:
        gen = _make_generator(concurrent_inputs=None, target_inputs=3)
        result = gen._render_concurrent_decorator()
        assert "@modal.concurrent(target_inputs=3)" in result
        assert "max_inputs" not in result

    def test_both_max_and_target_inputs(self) -> None:
        gen = _make_generator(concurrent_inputs=5, target_inputs=3)
        result = gen._render_concurrent_decorator()
        assert "max_inputs=5" in result
        assert "target_inputs=3" in result


class TestRenderScalingArgs:
    def test_all_none_returns_empty_string(self) -> None:
        gen = _make_generator()
        assert gen._render_scaling_args() == ""

    def test_min_containers_zero_excluded(self) -> None:
        gen = _make_generator(min_containers=0)
        result = gen._render_scaling_args()
        assert "min_containers" not in result

    def test_min_containers_positive_included(self) -> None:
        gen = _make_generator(min_containers=2)
        result = gen._render_scaling_args()
        assert "min_containers=2," in result

    def test_max_containers_included(self) -> None:
        gen = _make_generator(max_containers=10)
        result = gen._render_scaling_args()
        assert "max_containers=10," in result

    def test_buffer_containers_included(self) -> None:
        gen = _make_generator(buffer_containers=3)
        result = gen._render_scaling_args()
        assert "buffer_containers=3," in result

    def test_scaledown_window_included(self) -> None:
        gen = _make_generator(scaledown_window=120)
        result = gen._render_scaling_args()
        assert "scaledown_window=120," in result

    def test_startup_timeout_included(self) -> None:
        gen = _make_generator(startup_timeout=600)
        result = gen._render_scaling_args()
        assert "startup_timeout=600," in result

    def test_multiple_scaling_params_all_present(self) -> None:
        gen = _make_generator(
            min_containers=2,
            max_containers=20,
            buffer_containers=3,
            scaledown_window=120,
            startup_timeout=600,
        )
        result = gen._render_scaling_args()
        assert "min_containers=2," in result
        assert "max_containers=20," in result
        assert "buffer_containers=3," in result
        assert "scaledown_window=120," in result
        assert "startup_timeout=600," in result


class TestRenderWheelInstallCode:
    def test_no_wheels_returns_empty_string(self) -> None:
        gen = _make_generator(wheel_filenames=None)
        assert gen._render_wheel_install_code() == ""

    def test_single_wheel_path_in_code(self) -> None:
        gen = _make_generator(wheel_filenames=["my_pkg-1.0.0-py3-none-any.whl"])
        result = gen._render_wheel_install_code()
        assert "/model/wheels/my_pkg-1.0.0-py3-none-any.whl" in result
        assert "subprocess.check_call" in result

    def test_multiple_wheels_all_paths_present(self) -> None:
        gen = _make_generator(wheel_filenames=["pkg_a-1.0.whl", "pkg_b-2.0.whl"])
        result = gen._render_wheel_install_code()
        assert "/model/wheels/pkg_a-1.0.whl" in result
        assert "/model/wheels/pkg_b-2.0.whl" in result


class TestRenderSecretDecorator:
    def test_empty_secret_returns_empty_strings(self) -> None:
        gen = _make_generator(modal_secret="")
        secret_str, secrets_arg = gen._render_secret_decorator()
        assert secret_str == ""
        assert secrets_arg == ""

    def test_secret_name_generates_reference_and_arg(self) -> None:
        gen = _make_generator(modal_secret="pip-credentials")
        secret_str, secrets_arg = gen._render_secret_decorator()
        assert 'modal.Secret.from_name("pip-credentials")' in secret_str
        assert "pip_secret" in secret_str
        assert secrets_arg == "secrets=[pip_secret],"


class TestRenderFastapiEndpointDecorator:
    def test_no_proxy_auth_standard_decorator(self) -> None:
        gen = _make_generator(proxy_auth=False)
        result = gen._render_fastapi_endpoint_decorator()
        assert result == '@modal.fastapi_endpoint(method="POST")'

    def test_proxy_auth_adds_requires_proxy_auth(self) -> None:
        gen = _make_generator(proxy_auth=True)
        result = gen._render_fastapi_endpoint_decorator()
        assert "requires_proxy_auth=True" in result
        assert 'method="POST"' in result


class TestRenderPredictMethods:
    def test_batching_disabled_simple_predict_no_proxy_auth(self) -> None:
        gen = _make_generator(enable_batching=False, proxy_auth=False)
        result = gen._render_predict_methods()
        assert "@modal.batched" not in result
        assert "def predict" in result
        assert "requires_proxy_auth" not in result

    def test_batching_disabled_simple_predict_proxy_auth(self) -> None:
        gen = _make_generator(enable_batching=False, proxy_auth=True)
        result = gen._render_predict_methods()
        assert "requires_proxy_auth=True" in result
        assert "def predict" in result

    def test_batching_enabled_includes_predict_batch_and_delegation(self) -> None:
        gen = _make_generator(enable_batching=True, max_batch_size=8, batch_wait_ms=100)
        result = gen._render_predict_methods()
        assert "@modal.batched" in result
        assert "def predict_batch" in result
        assert "return self.predict_batch.local([input_data])[0]" in result

    def test_batching_enabled_proxy_auth_on_predict_endpoint(self) -> None:
        gen = _make_generator(enable_batching=True, proxy_auth=True)
        result = gen._render_predict_methods()
        assert "requires_proxy_auth=True" in result

    def test_batching_enabled_modal_batched_has_no_proxy_auth(self) -> None:
        gen = _make_generator(enable_batching=True, proxy_auth=True)
        result = gen._render_predict_methods()
        batched_idx = result.index("@modal.batched")
        proxy_idx = result.index("requires_proxy_auth=True")
        assert proxy_idx > batched_idx


class TestRenderStreamingMethod:
    def test_no_proxy_auth_standard_decorator(self) -> None:
        gen = _make_generator(proxy_auth=False)
        result = gen._render_streaming_method()
        assert "requires_proxy_auth" not in result
        assert '@modal.fastapi_endpoint(method="POST")' in result

    def test_proxy_auth_adds_requires_proxy_auth(self) -> None:
        gen = _make_generator(proxy_auth=True)
        result = gen._render_streaming_method()
        assert "requires_proxy_auth=True" in result

    def test_streaming_fallback_logic_always_present(self) -> None:
        gen = _make_generator()
        result = gen._render_streaming_method()
        assert "hasattr(self.model, 'predict_stream')" in result
        assert "StreamingResponse" in result
        assert "data: [DONE]" in result


# ---------------------------------------------------------------------------
# Integration tests for generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_output_is_deterministic(self) -> None:
        config = _make_base_config(gpu_config="T4", enable_batching=True)
        gen = ModalAppCodeGenerator(config)
        assert gen.generate() == gen.generate()

    def test_section_ordering_header_predict_streaming(self) -> None:
        code = generate_modal_app_code(_make_base_config())
        header_idx = code.index("app = modal.App")
        predict_idx = code.index("def predict(")
        stream_idx = code.index("def predict_stream(")
        assert header_idx < predict_idx < stream_idx

    def test_gpu_none_appears_in_cls_decorator(self) -> None:
        code = generate_modal_app_code(_make_base_config(gpu_config=None))
        assert "gpu=None" in code

    def test_proxy_auth_propagates_to_all_endpoints(self) -> None:
        code = generate_modal_app_code(_make_base_config(proxy_auth=True, enable_batching=False))
        occurrences = code.count("requires_proxy_auth=True")
        # predict and predict_stream both get the flag
        assert occurrences >= 2

    def test_proxy_auth_propagates_to_all_endpoints_with_batching(self) -> None:
        code = generate_modal_app_code(_make_base_config(proxy_auth=True, enable_batching=True))
        occurrences = code.count("requires_proxy_auth=True")
        # predict (batch wrapper) and predict_stream both get the flag
        assert occurrences >= 2

    def test_app_name_used_consistently(self) -> None:
        code = generate_modal_app_code(_make_base_config(app_name="my-inference-app"))
        assert 'modal.App("my-inference-app")' in code
        assert '"my-inference-app-model-volume"' in code

    def test_python_version_in_image(self) -> None:
        code = generate_modal_app_code(_make_base_config(python_version="3.11"))
        assert 'python_version="3.11"' in code

    @pytest.mark.parametrize(
        "gpu_config,expected",
        [
            (None, "gpu=None"),
            ("T4", 'gpu="T4"'),
            ("H100:8", 'gpu="H100:8"'),
            (["H100", "A100-80GB"], 'gpu=["H100", "A100-80GB"]'),
        ],
    )
    def test_gpu_variants_in_generated_code(self, gpu_config: object, expected: str) -> None:
        code = generate_modal_app_code(_make_base_config(gpu_config=gpu_config))
        assert expected in code
