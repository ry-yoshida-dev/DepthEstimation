import subprocess
import sys

BACKEND_MODULES: tuple[str, ...] = (
    "torch",
    "transformers",
    "depth_estimation.backends.core",
    "depth_estimation.backends.zoedepth",
    "depth_estimation.backends.depth_pro",
    "depth_estimation.backends.prompt_depth_anything",
)


def test_package_import_loads_no_backend() -> None:
    script: str = (
        "import sys\n"
        + "import depth_estimation\n"
        + f"print(','.join(name for name in {BACKEND_MODULES!r} if name in sys.modules))\n"
    )
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""
