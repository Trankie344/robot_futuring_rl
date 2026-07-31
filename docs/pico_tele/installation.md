# PICO teleoperation installation

## Prerequisites

The target host is Ubuntu 22.04 with ROS 2 Humble. Install ROS development tools and runtime dependencies, including
`colcon`, `ament_cmake`, `ament_cmake_gtest`, `nlohmann_json`, `rmw_cyclonedds_cpp`, `unzip`, `binutils`, and
`dpkg-deb`.

The PXREA SDK, PC Service, PICO application, APKs, Debian packages, and Unity packages are external assets and are
never committed to Git. Use the [XR-Robotics GitHub organization](https://github.com/XR-Robotics) as the source of
truth. In particular, do not use `XRoboToolkit-Unity-Client-Quest` on a PICO headset.

## Official packages

The versions below were checked against the official GitHub Releases on 2026-07-31.

| Component | Official source | Recommended asset | SHA-256 |
|---|---|---|---|
| Ubuntu 22.04 x86_64 PC Service | [PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0) | [`XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb) | `61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91` |
| Ubuntu 24.04 x86_64 PC Service | [PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0) | [`XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb) | `bce661f0be0b8a246ceecb2e5f1675a81c26b834648dc7fdf23f8c0bfe2a5d19` |
| ARM64 headless PC Service | [PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0) | [`XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb) | `532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f` |
| ARM64 GUI PC Service | [PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0) | [`XRoboToolkit-PC-Service_1.0.0.0_arm64.deb`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb) | `2a6bc1088c77c363fb13030ed0fc8d0f048ec12bbd8ec61a68d2ebd48ed9f942` |
| PICO application | [Unity Client v1.1.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1) | [`XRoboToolkit-PICO-1.1.1.apk`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk) | `6b2bb282405673d24abcb1980e3478b8f1052e90f7207b1f24cc56a59f8d8261` |
| PICO SDK 29 application variant | [Unity Client v1.1.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1) | [`XRoboToolkit-PICO-1.1.1-SDK29.apk`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1-SDK29.apk) | `6089e1ce175cf3e292cea5919c2b97711151186ca39c4a5569786a12fc167eb2` |
| Unity integration package | [Unity Client v1.1.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1) | [`XRoboToolkit-1.1.1.unitypackage`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-1.1.1.unitypackage) | `c40dd6810a2cca94f5ff6701b5f21848eecd0129ab46f4aa07326785ead09a3e` |

The normal PICO deployment only needs one PICO APK. The SDK 29 APK is a separately published compatibility variant;
choose the variant required by the headset deployment environment and do not install both simultaneously. The Unity
package is only required when building a customized headset application.

The robot SDK is maintained in the
[`RoboticsService/SDK`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/tree/85bac4dbc1fd5cef42c74a160d9c30aa3491f122/RoboticsService/SDK)
directory of the PC Service repository. An installed Linux PC Service exposes the same SDK at
`/opt/apps/roboticsservice/SDK`, including `PXREARobotSDK.h` and the x86_64/aarch64 libraries expected by this bridge.

## Install PC Service and build

For the standard Ubuntu 22.04 x86_64 deployment, download and verify the official package:

```bash
curl -fL -o XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb \
  https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
echo "61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91  XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb" \
  | sha256sum --check
sudo apt install ./XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Use the Ubuntu 24.04 asset only on Ubuntu 24.04. On an ARM64 host, use the official ARM64 GUI package or the headless
package from the same release. Never install a package whose Debian architecture does not match
`dpkg --print-architecture`.

After installation, build against the SDK installed by PC Service:

```bash
export PICO_ROBOT_SDK_ROOT=/opt/apps/roboticsservice/SDK
openpi/scripts/pico_tele/build.sh
```

For an external installation without `sudo`, unpack a matching official package into the external runtime:

```bash
export PICO_TELE_RUNTIME=/mnt/workspace/ys/futuring/openpi_runtime/pico_tele
mkdir -p "${PICO_TELE_RUNTIME}/pc_service_root"
dpkg-deb -x <matching-pc-service.deb> "${PICO_TELE_RUNTIME}/pc_service_root"
export PICO_ROBOT_SDK_ROOT="${PICO_TELE_RUNTIME}/pc_service_root/opt/apps/roboticsservice/SDK"
openpi/scripts/pico_tele/build.sh
```

Confirm that `${PICO_ROBOT_SDK_ROOT}/include/PXREARobotSDK.h` exists before building.

## Offline restored assets

The restored files below are retained only as validated offline mirrors, not as the primary download source:

```text
restored/robot/rootfs/home/zme/pico_tele_zme/SDK.zip
restored/robot/rootfs/home/zme/pico_tele_zme/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
```

Their known SHA-256 values are respectively
`d9e4dee0627c68ae1be071700352d347da4a2d3c53e5dd24496c2d23ae1838b2` and
`532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f`. The restored `.deb` is the same official
ARM64 headless v1.0.0 release asset and must not be used on x86_64. Prepare the restored SDK snapshot with:

```bash
openpi/scripts/pico_tele/prepare_runtime.sh \
  --sdk-zip restored/robot/rootfs/home/zme/pico_tele_zme/SDK.zip
```

On an arm64 target, the PC Service can be unpacked into the external runtime without system installation:

```bash
openpi/scripts/pico_tele/prepare_runtime.sh \
  --sdk-zip restored/robot/rootfs/home/zme/pico_tele_zme/SDK.zip \
  --pc-service-deb restored/robot/rootfs/home/zme/pico_tele_zme/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
```

The default runtime is `/mnt/workspace/ys/futuring/openpi_runtime/pico_tele`; override it with `PICO_TELE_RUNTIME`.
The preparation script validates the archive digest, ZIP layout, and both SDK ELF architectures. Then build only the
standalone PICO packages:

```bash
openpi/scripts/pico_tele/build.sh
```

For CI without the proprietary SDK, build the messages, gesture mapper, and command router with:

```bash
PICO_TELE_BUILD_SDK_BRIDGE=OFF openpi/scripts/pico_tele/build.sh
```

## Configure the PICO headset

Install one of the official PICO APKs from
[XRoboToolkit Unity Client v1.1.1](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1),
or build the source at the protocol-reference commit
[`c9326092ff4d11e8b507b041713194b93470a8e1`](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/commit/c9326092ff4d11e8b507b041713194b93470a8e1).
For example, after enabling the headset's normal Android deployment/debugging workflow:

```bash
curl -fL -o XRoboToolkit-PICO-1.1.1.apk \
  https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk
echo "6b2bb282405673d24abcb1980e3478b8f1052e90f7207b1f24cc56a59f8d8261  XRoboToolkit-PICO-1.1.1.apk" \
  | sha256sum --check
adb install -r XRoboToolkit-PICO-1.1.1.apk
```

Then configure the application:

1. Put the PICO and host on the same network.
2. Enable `Head`, `Controller`, and `Data & Control - Send`.
3. Disable `Switch w/ A Button`; otherwise the A button can stop tracking while a dual-primary gesture is held.
4. Enter or select the PC Service host IP.
5. Allow TCP port `63901` through the host firewall.

## Start the stack

Start PC Service in one terminal:

```bash
openpi/scripts/pico_tele/run_pc_service.sh
```

Start the ROS bridge in another terminal:

```bash
openpi/scripts/pico_tele/run_pico_tele.sh
```

Defaults are `ROS_DOMAIN_ID=30` and `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. Set `CYCLONEDDS_URI` externally when a
deployment-specific CycloneDDS configuration is required. Use `PICO_DEVICE_ID=<id>` to select a fixed headset.

The SDK bridge retries `PXREAInit` if PC Service is unavailable. Device disconnects are logged, and an automatically
selected device is released when it goes offline so another device may connect.
