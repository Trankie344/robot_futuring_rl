from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_assets_remain_external_to_repository() -> None:
    prepare = (SCRIPT_ROOT / "prepare_runtime.sh").read_text(encoding="utf-8")
    assert "/mnt/workspace/ys/futuring/openpi_runtime/pico_tele" in prepare
    assert "unsafe or unexpected SDK ZIP member" in prepare
    assert 'linux/64/libPXREARobotSDK.so" "Advanced Micro Devices X86-64"' in prepare
    assert 'linux_aarch64/64/libPXREARobotSDK.so" "AArch64"' in prepare
    assert "SDK.zip" not in {path.name for path in SCRIPT_ROOT.iterdir()}
    assert not any(path.suffix in {".so", ".deb", ".zip"} for path in SCRIPT_ROOT.rglob("*"))


def test_build_uses_only_pico_tele_colcon_base_path() -> None:
    build = (SCRIPT_ROOT / "build.sh").read_text(encoding="utf-8")
    assert 'src/pico_tele"' in build
    assert "PICO_TELE_BUILD_SDK_BRIDGE" in build


def test_launch_defaults_preserve_ros_environment() -> None:
    run = (SCRIPT_ROOT / "run_pico_tele.sh").read_text(encoding="utf-8")
    assert 'ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"' in run
    assert 'RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"' in run
