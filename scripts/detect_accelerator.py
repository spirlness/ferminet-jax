"""Detect available JAX accelerators and report suggested extras."""

from __future__ import annotations

import json
from typing import Dict, List

import jax


def summarize_devices() -> Dict[str, object]:
    devices = jax.devices()
    summary: Dict[str, object] = {
        "jax_version": jax.__version__,
        "devices": [],
        "recommendation": "cpu",
    }

    platforms = {device.platform for device in devices}
    if "gpu" in platforms:
        summary["recommendation"] = "cuda"
    elif "tpu" in platforms:
        summary["recommendation"] = "tpu"

    for device in devices:
        summary["devices"].append(
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", "unknown"),
            }
        )

    return summary


def main() -> None:  # pragma: no cover - convenience CLI
    summary = summarize_devices()
    print("Detected accelerator configuration:")
    print(json.dumps(summary, indent=2))
    rec = summary["recommendation"]
    if rec == "cuda":
        print("\nSuggestion: install GPU build via `uv sync --extra cuda --dev`.\n")
    elif rec == "tpu":
        print("\nSuggestion: install TPU build via `uv sync --extra tpu --dev`.\n")
    else:
        print("\nCPU backend detected; default dependencies are sufficient.\n")


if __name__ == "__main__":
    main()
