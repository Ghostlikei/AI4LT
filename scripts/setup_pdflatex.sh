#!/usr/bin/env bash

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ai4lt" ]]; then
  echo "Activate the Conda environment 'ai4lt' first." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Activate 'ai4lt' and retry." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TINYTEX_ROOT="${REPO_ROOT}/.tools"
TINYTEX_DIR="${TINYTEX_ROOT}/.TinyTeX"
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
ACTIVATE_HOOK="${ACTIVATE_DIR}/ai4lt_tinytex.sh"
DEACTIVATE_HOOK="${DEACTIVATE_DIR}/ai4lt_tinytex.sh"

find_tinytex_bin() {
  find "${TINYTEX_DIR}/bin" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1
}

TINYTEX_BIN="$(find_tinytex_bin || true)"

if [[ -z "${TINYTEX_BIN}" || ! -x "${TINYTEX_BIN}/pdflatex" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  installer="${tmpdir}/install-bin-unix.sh"

  curl -fsSL "https://yihui.org/tinytex/install-bin-unix.sh" -o "${installer}"
  rm -rf "${TINYTEX_DIR}"
  mkdir -p "${TINYTEX_ROOT}"
  FORCE_REBUILD=1 TINYTEX_DIR="${TINYTEX_ROOT}" sh "${installer}" "${tmpdir}" --no-path

  TINYTEX_BIN="$(find_tinytex_bin || true)"
fi

if [[ -z "${TINYTEX_BIN}" || ! -x "${TINYTEX_BIN}/pdflatex" ]]; then
  echo "TinyTeX installation did not produce a working pdflatex binary." >&2
  exit 1
fi

"${TINYTEX_BIN}/tlmgr" install babel-english listings >/dev/null

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_HOOK}" <<EOF
export AI4LT_TINYTEX_BIN="${TINYTEX_BIN}"
case ":\$PATH:" in
  *":\${AI4LT_TINYTEX_BIN}:"*) ;;
  *) export PATH="\${AI4LT_TINYTEX_BIN}:\$PATH" ;;
esac
EOF

cat > "${DEACTIVATE_HOOK}" <<'EOF'
if [ -n "${AI4LT_TINYTEX_BIN:-}" ]; then
  PATH=":${PATH}:"
  PATH="${PATH//:${AI4LT_TINYTEX_BIN}:/:}"
  PATH="${PATH#:}"
  PATH="${PATH%:}"
  export PATH
  unset AI4LT_TINYTEX_BIN
fi
EOF

export AI4LT_TINYTEX_BIN="${TINYTEX_BIN}"
case ":$PATH:" in
  *":${AI4LT_TINYTEX_BIN}:"*) ;;
  *) export PATH="${AI4LT_TINYTEX_BIN}:$PATH" ;;
esac

echo "TinyTeX path: ${AI4LT_TINYTEX_BIN}"
pdflatex --version | head -n 2
