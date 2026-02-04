#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-}"
if [[ -z "${DEVICE}" ]]; then
  echo "Usage: $0 {cpu|gpu|tpu}"
  exit 1
fi

case "${DEVICE}" in
  cpu)
    uv pip install -e ".[cpu,dev]"
    ;;
  gpu)
    uv pip install -e ".[gpu,dev]" \
      --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ;;
  tpu)
    uv pip install -e ".[tpu,dev]" \
      --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ;;
  *)
    echo "Unknown device: ${DEVICE}. Use cpu, gpu, or tpu."
    exit 1
    ;;
esac
