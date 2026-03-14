#!/bin/bash
echo "Running Baseline"
uv run python scripts/benchmark_train_step.py --timed-steps 100
