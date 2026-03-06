#!/usr/bin/env bash
set -euo pipefail

# ── FEAX PyPI build script ───────────────────────────────────────────────────
# Usage:
#   ./build.sh          Build sdist + wheel
#   ./build.sh upload   Build and upload to PyPI
#   ./build.sh test     Build and upload to TestPyPI

cd "$(dirname "$0")"

# ── Clean previous builds ────────────────────────────────────────────────────
echo "==> Cleaning previous builds ..."
rm -rf dist/ build/ *.egg-info feax.egg-info

# ── Install build tools (if missing) ─────────────────────────────────────────
echo "==> Checking build tools ..."
pip install --quiet --upgrade build twine

# ── Build sdist + wheel ──────────────────────────────────────────────────────
echo "==> Building sdist and wheel ..."
python -m build

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Build artifacts:"
ls -lh dist/

echo ""
echo "==> Checking with twine ..."
twine check dist/*

# ── Upload (optional) ────────────────────────────────────────────────────────
case "${1:-}" in
  upload)
    echo ""
    echo "==> Uploading to PyPI ..."
    twine upload dist/*
    ;;
  test)
    echo ""
    echo "==> Uploading to TestPyPI ..."
    twine upload --repository testpypi dist/*
    ;;
  *)
    echo ""
    echo "Done. To upload:"
    echo "  ./build.sh upload   # PyPI"
    echo "  ./build.sh test     # TestPyPI"
    ;;
esac
