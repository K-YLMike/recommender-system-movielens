#!/bin/bash
# =============================================================================
# One-time MovieLens download.
#
# Downloads and extracts ml-1m and/or ml-25m into <PROJECT>/raw ONCE. If the
# extracted directory already exists the download is skipped, so it is safe to
# re-run on every job -- later runs cost nothing and never re-fetch.
#
# Data lives under the project directory by default (parent of this script), so
# datasets sit next to the code. Override with RSM_DATA_ROOT if desired.
#
# Usage (from the project root):
#   bash data/download.sh both      # or: ml-1m  /  ml-25m
# =============================================================================
set -euo pipefail

WHICH="${1:-both}"
# Default to the project root (the parent of this data/ directory).
DEFAULT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${RSM_DATA_ROOT:-$DEFAULT_ROOT}"
RAW_DIR="$DATA_ROOT/raw"
mkdir -p "$RAW_DIR"
echo "Downloading into: $RAW_DIR"

download_one() {
  local name="$1"; local url="$2"
  local target="$RAW_DIR/$name"
  if [ -d "$target" ]; then
    echo "[skip] $name already present at $target"
    return 0
  fi
  local zip="$RAW_DIR/$name.zip"
  echo "[download] $name -> $zip"
  wget -c -O "$zip" "$url"          # -c resumes a partial download
  echo "[extract] $zip"
  unzip -q -o "$zip" -d "$RAW_DIR"
  rm -f "$zip"
  echo "[done] $name at $target"
}

case "$WHICH" in
  ml-1m)  download_one "ml-1m"  "https://files.grouplens.org/datasets/movielens/ml-1m.zip" ;;
  ml-25m) download_one "ml-25m" "https://files.grouplens.org/datasets/movielens/ml-25m.zip" ;;
  both)
    download_one "ml-1m"  "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    download_one "ml-25m" "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    ;;
  *) echo "Unknown option '$WHICH' (use ml-1m | ml-25m | both)"; exit 1 ;;
esac
echo "All requested datasets are available under $RAW_DIR"
