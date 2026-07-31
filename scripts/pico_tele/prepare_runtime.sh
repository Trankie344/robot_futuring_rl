#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SDK_SHA256="d9e4dee0627c68ae1be071700352d347da4a2d3c53e5dd24496c2d23ae1838b2"
EXPECTED_ARM64_DEB_SHA256="532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f"
RUNTIME_ROOT="${PICO_TELE_RUNTIME:-/mnt/workspace/ys/futuring/openpi_runtime/pico_tele}"
SDK_ZIP=""
PC_SERVICE_DEB=""
SDK_SHA256="${EXPECTED_SDK_SHA256}"
DEB_SHA256="${EXPECTED_ARM64_DEB_SHA256}"
ALLOW_UNVERIFIED=0

usage() {
  cat <<'EOF'
Usage: prepare_runtime.sh --sdk-zip PATH [options]

Options:
  --pc-service-deb PATH   Optional architecture-matching PC Service package.
  --runtime-root PATH     External runtime directory.
  --sdk-sha256 SHA256     Expected SDK archive digest.
  --deb-sha256 SHA256     Expected PC Service package digest.
  --allow-unverified      Accept assets with different digests after structural checks.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-zip)
      SDK_ZIP="$2"
      shift 2
      ;;
    --pc-service-deb)
      PC_SERVICE_DEB="$2"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --sdk-sha256)
      SDK_SHA256="$2"
      shift 2
      ;;
    --deb-sha256)
      DEB_SHA256="$2"
      shift 2
      ;;
    --allow-unverified)
      ALLOW_UNVERIFIED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SDK_ZIP}" || ! -f "${SDK_ZIP}" ]]; then
  echo "--sdk-zip must name an existing SDK.zip" >&2
  exit 2
fi

verify_digest() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" && "${ALLOW_UNVERIFIED}" != "1" ]]; then
    echo "${label} SHA-256 mismatch: expected ${expected}, got ${actual}" >&2
    echo "Use the matching asset, pass its expected digest, or explicitly use --allow-unverified." >&2
    exit 1
  fi
  printf '%s' "${actual}"
}

SDK_ACTUAL_SHA256="$(verify_digest "${SDK_ZIP}" "${SDK_SHA256}" "SDK archive")"
DEB_ACTUAL_SHA256=""
if [[ -n "${PC_SERVICE_DEB}" ]]; then
  if [[ ! -f "${PC_SERVICE_DEB}" ]]; then
    echo "--pc-service-deb must name an existing file" >&2
    exit 2
  fi
  DEB_ACTUAL_SHA256="$(verify_digest "${PC_SERVICE_DEB}" "${DEB_SHA256}" "PC Service package")"
fi

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pico-tele-runtime.XXXXXX")"
trap 'rm -rf "${TEMP_ROOT}"' EXIT
python3 - "${SDK_ZIP}" <<'PY'
from pathlib import PurePosixPath
import stat
import sys
import zipfile

archive_path = sys.argv[1]
seen = set()
with zipfile.ZipFile(archive_path) as archive:
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "SDK"
        ):
            raise SystemExit(f"unsafe or unexpected SDK ZIP member: {name!r}")
        normalized = str(path)
        if normalized in seen:
            raise SystemExit(f"duplicate SDK ZIP member: {name!r}")
        seen.add(normalized)
        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise SystemExit(f"SDK ZIP must not contain symbolic links: {name!r}")
PY
unzip -q "${SDK_ZIP}" -d "${TEMP_ROOT}/sdk_archive"
SDK_SOURCE_ROOT="${TEMP_ROOT}/sdk_archive/SDK"
for required in \
  "include/PXREARobotSDK.h" \
  "linux/64/libPXREARobotSDK.so" \
  "linux_aarch64/64/libPXREARobotSDK.so"; do
  if [[ ! -f "${SDK_SOURCE_ROOT}/${required}" ]]; then
    echo "SDK archive is missing SDK/${required}" >&2
    exit 1
  fi
done

verify_elf_machine() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(LC_ALL=C readelf -h "${path}" | awk -F: '/Machine:/ {sub(/^[[:space:]]+/, "", $2); print $2}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "PXREA SDK ELF architecture mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
}

verify_elf_machine "${SDK_SOURCE_ROOT}/linux/64/libPXREARobotSDK.so" "Advanced Micro Devices X86-64"
verify_elf_machine "${SDK_SOURCE_ROOT}/linux_aarch64/64/libPXREARobotSDK.so" "AArch64"

case "$(uname -m)" in
  x86_64|amd64)
    HOST_DEB_ARCH="amd64"
    ;;
  aarch64|arm64)
    HOST_DEB_ARCH="arm64"
    ;;
  *)
    echo "Unsupported host architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

DEB_ARCH=""
if [[ -n "${PC_SERVICE_DEB}" ]]; then
  DEB_ARCH="$(dpkg-deb -f "${PC_SERVICE_DEB}" Architecture)"
  if [[ "${DEB_ARCH}" != "${HOST_DEB_ARCH}" ]]; then
    echo "PC Service architecture ${DEB_ARCH} does not match host ${HOST_DEB_ARCH}" >&2
    exit 1
  fi
  dpkg-deb -x "${PC_SERVICE_DEB}" "${TEMP_ROOT}/pc_service_root"
  SERVICE_BINARY="${TEMP_ROOT}/pc_service_root/opt/apps/roboticsservice/RoboticsServiceProcess"
  if [[ ! -x "${SERVICE_BINARY}" ]]; then
    echo "PC Service package is missing RoboticsServiceProcess" >&2
    exit 1
  fi
fi

mkdir -p "${RUNTIME_ROOT}"
rm -rf "${RUNTIME_ROOT}/sdk"
cp -a "${SDK_SOURCE_ROOT}" "${RUNTIME_ROOT}/sdk"
if [[ -n "${PC_SERVICE_DEB}" ]]; then
  rm -rf "${RUNTIME_ROOT}/pc_service_root"
  cp -a "${TEMP_ROOT}/pc_service_root" "${RUNTIME_ROOT}/pc_service_root"
fi

python3 - \
  "${RUNTIME_ROOT}/asset_manifest.json" \
  "$(realpath "${SDK_ZIP}")" \
  "${SDK_ACTUAL_SHA256}" \
  "${PC_SERVICE_DEB:+$(realpath "${PC_SERVICE_DEB}")}" \
  "${DEB_ACTUAL_SHA256}" \
  "${DEB_ARCH}" <<'PY'
import json
from pathlib import Path
import sys

output, sdk_path, sdk_sha, deb_path, deb_sha, deb_arch = sys.argv[1:]
payload = {
    "schema_version": 1,
    "sdk": {"source": sdk_path, "sha256": sdk_sha},
    "pc_service": None,
}
if deb_path:
    payload["pc_service"] = {
        "source": deb_path,
        "sha256": deb_sha,
        "architecture": deb_arch,
    }
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Prepared PICO runtime: ${RUNTIME_ROOT}"
echo "export PICO_TELE_RUNTIME=${RUNTIME_ROOT}"
echo "export PICO_ROBOT_SDK_ROOT=${RUNTIME_ROOT}/sdk"
