CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TORCH_HOME="$CODE_ROOT/.cache/torch"
export XDG_CACHE_HOME="$CODE_ROOT/.cache/xdg"
export MPLCONFIGDIR="$CODE_ROOT/.cache/mpl"
export PIP_CACHE_DIR="$CODE_ROOT/.cache/pip"
export TMPDIR="$CODE_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
PYTHON="$CODE_ROOT/.venv/bin/python"
