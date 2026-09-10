#!/usr/bin/env bash
# Fetch the GGUF weights into ./models/.
#
#   bash scripts/download_models.sh              # default pair (~1.74 GB)
#   bash scripts/download_models.sh all          # every model (~5.7 GB)
#   bash scripts/download_models.sh qwen3-4b     # one model by name
#
# Nothing here needs huggingface-cli: these are plain HTTPS GETs, resumable
# with curl -C -, so it also works inside a Kaggle/Colab cell.

set -euo pipefail
cd "$(dirname "$0")/.."
MODELS_DIR="${AORUS_RAG_MODELS:-$PWD/models}"
mkdir -p "$MODELS_DIR"

declare -A REPO FILE SIZE
REPO[qwen2.5-3b]="bartowski/Qwen2.5-3B-Instruct-GGUF"; FILE[qwen2.5-3b]="Qwen2.5-3B-Instruct-Q4_K_M.gguf"; SIZE[qwen2.5-3b]="1.93 GB"
REPO[qwen3-1.7b]="unsloth/Qwen3-1.7B-GGUF";            FILE[qwen3-1.7b]="Qwen3-1.7B-Q4_K_M.gguf";          SIZE[qwen3-1.7b]="1.11 GB"
REPO[qwen3-4b]="unsloth/Qwen3-4B-Instruct-2507-GGUF";  FILE[qwen3-4b]="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"; SIZE[qwen3-4b]="2.50 GB"
REPO[e5-small]="cstr/multilingual-e5-small-GGUF";      FILE[e5-small]="multilingual-e5-small-q8_0.gguf";   SIZE[e5-small]="0.13 GB"
REPO[bge-m3]="lm-kit/bge-m3-gguf";                     FILE[bge-m3]="bge-m3-Q8_0.gguf";                   SIZE[bge-m3]="0.63 GB"

DEFAULT=(qwen3-1.7b bge-m3)
ALL=(qwen2.5-3b qwen3-1.7b qwen3-4b e5-small bge-m3)

case "${1:-default}" in
  default) TARGETS=("${DEFAULT[@]}") ;;
  all)     TARGETS=("${ALL[@]}") ;;
  *)       TARGETS=("$@") ;;
esac

for name in "${TARGETS[@]}"; do
  if [[ -z "${REPO[$name]:-}" ]]; then
    echo "unknown model '$name'. Available: ${ALL[*]}" >&2
    exit 1
  fi
  dest="$MODELS_DIR/${FILE[$name]}"
  if [[ -f "$dest" ]]; then
    echo "have  $name  ($(basename "$dest"))"
    continue
  fi
  echo "get   $name  ${SIZE[$name]}  <- ${REPO[$name]}"
  curl -fL -C - --progress-bar \
    "https://huggingface.co/${REPO[$name]}/resolve/main/${FILE[$name]}?download=true" \
    -o "$dest"
done

echo
du -h "$MODELS_DIR"/*.gguf 2>/dev/null || true
