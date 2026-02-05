"""Checkpoint save/restore utilities for FermiNet training.

This module provides functionality to save and restore training state including:
- Network parameters
- Optimizer state
- MCMC sampler state
- Training step counter
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class CheckpointData:
    """Container for checkpoint state.

    Attributes:
        step: Training step number.
        params: Network parameters (pytree of arrays).
        opt_state: Optimizer state (e.g., Adam m, v, t).
        mcmc_state: MCMC sampler state (e.g., walker positions).
    """

    step: int
    params: Any
    opt_state: Any
    mcmc_state: Any


def save_checkpoint(
    path: str,
    step: int,
    params: Any,
    opt_state: Any,
    mcmc_state: Any,
) -> Path:
    """Save training checkpoint to disk.

    Saves network parameters, optimizer state, and MCMC state to a pickle file
    with the step number in the filename for easy identification.

    Args:
        path: Directory where checkpoint will be saved.
        step: Training step number (used in filename).
        params: Network parameters (pytree of jax arrays).
        opt_state: Optimizer state (pytree, typically from optax).
        mcmc_state: MCMC sampler state (pytree).

    Returns:
        Path object pointing to the saved checkpoint file.

    Raises:
        OSError: If directory doesn't exist or file can't be written.
    """
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Format filename with zero-padded step number
    filename = f"checkpoint_{step:08d}.pkl"
    checkpoint_path = checkpoint_dir / filename

    # Create checkpoint data
    checkpoint = CheckpointData(
        step=step,
        params=params,
        opt_state=opt_state,
        mcmc_state=mcmc_state,
    )

    # Serialize to pickle
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    return checkpoint_path


def restore_checkpoint(path: str) -> CheckpointData:
    """Restore training checkpoint from disk.

    Loads and deserializes checkpoint data from a pickle file.

    Args:
        path: Path to checkpoint file (e.g., "checkpoint_00001000.pkl").

    Returns:
        CheckpointData object containing step, params, opt_state, mcmc_state.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        pickle.UnpicklingError: If file is corrupted or invalid.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint


def find_latest_checkpoint(directory: str) -> Optional[Tuple[Path, CheckpointData]]:
    """Find and load the most recent checkpoint in a directory.

    Searches for all checkpoint files matching the pattern "checkpoint_*.pkl"
    and returns the one with the highest step number.

    Args:
        directory: Directory containing checkpoint files.

    Returns:
        Tuple of (checkpoint_path, checkpoint_data) for the latest checkpoint,
        or None if no checkpoints found. Returns the highest step number.

    Raises:
        OSError: If directory doesn't exist.
        pickle.UnpicklingError: If latest checkpoint file is corrupted.
    """
    checkpoint_dir = Path(directory)

    if not checkpoint_dir.exists():
        raise OSError(f"Directory not found: {checkpoint_dir}")

    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))

    if not checkpoint_files:
        return None

    # Get the last file (highest step due to numeric sorting)
    latest_path = checkpoint_files[-1]
    checkpoint_data = restore_checkpoint(str(latest_path))

    return latest_path, checkpoint_data


def checkpoint_exists(directory: str, step: Optional[int] = None) -> bool:
    """Check if a checkpoint exists.

    Args:
        directory: Directory containing checkpoints.
        step: Specific step number to check. If None, checks if any checkpoint exists.

    Returns:
        True if checkpoint exists, False otherwise.
    """
    checkpoint_dir = Path(directory)

    if not checkpoint_dir.exists():
        return False

    if step is None:
        # Check if any checkpoint exists
        return bool(list(checkpoint_dir.glob("checkpoint_*.pkl")))
    else:
        # Check specific step
        checkpoint_path = checkpoint_dir / f"checkpoint_{step:08d}.pkl"
        return checkpoint_path.exists()


def list_checkpoints(directory: str) -> List[Tuple[Path, int]]:
    """List all checkpoints in a directory, sorted by step number.

    Args:
        directory: Directory containing checkpoint files.

    Returns:
        List of (checkpoint_path, step_number) tuples, sorted by step ascending.
        Empty list if directory doesn't exist or has no checkpoints.
    """
    checkpoint_dir = Path(directory)

    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for checkpoint_path in sorted(checkpoint_dir.glob("checkpoint_*.pkl")):
        # Extract step from filename: checkpoint_00001000.pkl -> 1000
        step = int(checkpoint_path.stem.split("_")[1])
        checkpoints.append((checkpoint_path, step))

    return checkpoints
