#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/protein-shape.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT

export SOURCE_DATE_EPOCH=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"

cd "$ROOT"

uv sync --locked
uv lock --check
uv run --frozen --no-sync pytest -q -p no:cacheprovider
uv run --frozen --no-sync python scripts/tables.py
uv run --frozen --no-sync python scripts/make_figures.py
shasum -a 256 -c --quiet results/CHECKSUMS.txt
echo "reproduce: PASS"
