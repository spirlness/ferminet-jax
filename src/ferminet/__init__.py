# Copyright 2024 FermiNet Authors.
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnknownVariableType=false, reportUnusedImport=false
# Licensed under the Apache License, Version 2.0.

"""FermiNet: Fermi Neural Network for quantum chemistry.

A JAX implementation of FermiNet for solving the many-electron Schrödinger
equation using Variational Monte Carlo.

This implementation follows the architecture from:
  D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes,
  "Ab Initio Solution of the Many-Electron Schrödinger Equation with
  Deep Neural Networks", Physical Review Research 2, 033429 (2020).
"""

from ferminet import (
    base_config,
    checkpoint,
    constants,
    envelopes,
    hamiltonian,
    jastrows,
    loss,
    mcmc,
    network_blocks,
)

try:
    from ferminet import networks
except ImportError:  # pragma: no cover - optional module.
    networks = None

from ferminet import pretrain

try:
    from ferminet import train
except ImportError:  # pragma: no cover - optional module.
    train = None
from ferminet import types
from ferminet.base_config import SystemType

# Configuration
from ferminet.base_config import default as default_config
from ferminet.base_config import resolve as resolve_config
from ferminet.checkpoint import CheckpointData, restore_checkpoint, save_checkpoint

# Constants
from ferminet.constants import PMAP_AXIS_NAME, all_gather, pmap, pmean

# Envelopes
from ferminet.envelopes import Envelope, EnvelopeType, make_isotropic_envelope
from ferminet.hamiltonian import local_energy

# Jastrows
from ferminet.jastrows import JastrowType, get_jastrow
from ferminet.loss import AuxiliaryLossData, clip_local_energy, make_loss
from ferminet.mcmc import make_mcmc_step, mh_update

# Network blocks
from ferminet.network_blocks import (
    array_partitions,
    init_linear_layer,
    linear_layer,
    split_into_blocks,
)

# Core types
from ferminet.types import (
    FermiNetData,
    FermiNetLike,
    InitFermiNet,
    LogFermiNetLike,
    OrbitalFnLike,
    Param,
    ParamTree,
)

try:
    from ferminet.networks import make_fermi_net, make_log_psi_apply
except ImportError:  # pragma: no cover - optional module.
    make_fermi_net = None
    make_log_psi_apply = None

try:
    from ferminet.train import train as train_vmc
except ImportError:  # pragma: no cover - optional module.
    train_vmc = None

__version__ = "0.2.0"

__all__ = [
    # Modules
    "base_config",
    "constants",
    "network_blocks",
    "types",
    # Types
    "FermiNetData",
    "ParamTree",
    "Param",
    "InitFermiNet",
    "FermiNetLike",
    "LogFermiNetLike",
    "OrbitalFnLike",
    # Config
    "default_config",
    "resolve_config",
    "SystemType",
    # Constants
    "PMAP_AXIS_NAME",
    "pmap",
    "pmean",
    "all_gather",
    # Network blocks
    "init_linear_layer",
    "linear_layer",
    "array_partitions",
    "split_into_blocks",
]
