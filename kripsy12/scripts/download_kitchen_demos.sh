#!/usr/bin/env bash
# Unduh ulang demo Kitchen MJL resmi (Relay Policy Learning).
# Sumber: https://github.com/google-research/relay-policy-learning
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KITCHEN="$ROOT/FlowPolicy/data/kitchen"
ZIP="$KITCHEN/kitchen_demos_multitask.zip"
URL="https://media.githubusercontent.com/media/google-research/relay-policy-learning/master/kitchen_demos_multitask.zip"
TARGET="$KITCHEN/kitchen_demos_multitask"
LEGACY="$ROOT/FlowPolicy/FlowPolicy/data/kitchen/kitchen_demos_multitask"

mkdir -p "$KITCHEN"
echo ">>> Downloading kitchen_demos_multitask.zip (~663 MB)..."
curl -L --progress-bar "$URL" -o "$ZIP"

echo ">>> Extracting..."
rm -rf "$TARGET"
unzip -q "$ZIP" -d "$KITCHEN"

if [ -d "$LEGACY" ]; then
  echo ">>> Removing legacy duplicate: $LEGACY"
  rm -rf "$(dirname "$LEGACY")"
fi

N="$(find "$TARGET" -mindepth 2 -maxdepth 2 -name '*.mjl' | wc -l)"
echo ">>> Done: $N demo MJL di $TARGET"
