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

from ferminet import base_config
from ferminet import checkpoint
from ferminet import constants
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import jastrows
from ferminet import loss
from ferminet import mcmc
from ferminet import network_blocks

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

# Core types
from ferminet.types import FermiNetData
from ferminet.types import ParamTree
from ferminet.types import Param
from ferminet.types import InitFermiNet
from ferminet.types import FermiNetLike
from ferminet.types import LogFermiNetLike
from ferminet.types import OrbitalFnLike

# Configuration
from ferminet.base_config import default as default_config
from ferminet.base_config import resolve as resolve_config
from ferminet.base_config import SystemType

# Constants
from ferminet.constants import PMAP_AXIS_NAME
from ferminet.constants import pmap
from ferminet.constants import pmean
from ferminet.constants import all_gather

# Network blocks
from ferminet.network_blocks import init_linear_layer
from ferminet.network_blocks import linear_layer
from ferminet.network_blocks import array_partitions
from ferminet.network_blocks import split_into_blocks

# Envelopes
from ferminet.envelopes import EnvelopeType
from ferminet.envelopes import Envelope
from ferminet.envelopes import make_isotropic_envelope

# Jastrows
from ferminet.jastrows import JastrowType
from ferminet.jastrows import get_jastrow

from ferminet.loss import AuxiliaryLossData
from ferminet.loss import clip_local_energy
from ferminet.loss import make_loss

from ferminet.hamiltonian import local_energy

from ferminet.mcmc import mh_update
from ferminet.mcmc import make_mcmc_step

from ferminet.checkpoint import CheckpointData
from ferminet.checkpoint import save_checkpoint
from ferminet.checkpoint import restore_checkpoint

try:
    from ferminet.networks import make_fermi_net
    from ferminet.networks import make_log_psi_apply
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
