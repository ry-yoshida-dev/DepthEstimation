import pytest
import torch

from depth_estimation import DepthBackend, DepthKind, Device, EstimatorSettings, Precision
from depth_estimation.runtime import TorchRuntime


def build_settings(device: Device, precision: Precision = Precision.FLOAT32) -> EstimatorSettings:
    return EstimatorSettings(
        backend=DepthBackend.DPT, weights_path="stub", depth_kind=DepthKind.METRIC, device=device, precision=precision
    )


def test_cpu_runtime_prepares_models_and_inputs() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.CPU))
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    model: torch.nn.Linear = torch.nn.Linear(2, 2).double()
    runtime.prepare_model(model)
    assert not model.training
    assert model.weight.dtype == torch.float32
    assert runtime.to_model_input(torch.zeros(1, 2, dtype=torch.float64)).dtype == torch.float32


def test_auto_device_resolves_to_an_available_device() -> None:
    assert TorchRuntime(build_settings(Device.AUTO)).device.type in {Device.CPU, Device.CUDA, Device.MPS}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_unavailable_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        TorchRuntime(build_settings(Device.CUDA))


def test_precisions_on_cpu() -> None:
    assert TorchRuntime(build_settings(Device.CPU, Precision.BFLOAT16)).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="float16"):
        TorchRuntime(build_settings(Device.CPU, Precision.FLOAT16))
