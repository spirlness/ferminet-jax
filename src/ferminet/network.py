import jax
import jax.numpy as jnp
from typing import Dict

from .multi_determinant import MultiDeterminantOrbitals
from .jastrow import JastrowFactor


class SimpleFermiNet:
    """
    SimpleFermiNet implementation for Stage 1.
    Acts as a base class for ExtendedFermiNet.
    """

    def __init__(self, n_electrons, n_up, nuclei_config, network_config):
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.nuclei_config = nuclei_config
        self.config = network_config
        self.params = {}

    def __call__(self, r_elec):
        raise NotImplementedError("Use ExtendedFermiNet for full functionality.")


class ExtendedFermiNet(SimpleFermiNet):
    """
    Extended FermiNet (Stage 2) with Multi-Determinants, Jastrow Factor, and Residual connections.
    Includes numerical stability improvements for Log-Sum-Exp calculation.
    """

    def __init__(self, n_electrons, n_up, nuclei_config, network_config):
        super().__init__(n_electrons, n_up, nuclei_config, network_config)

        # Multi-determinant orbitals component
        # We reuse the logic for computing orbitals from MultiDeterminantOrbitals
        self.orbitals = MultiDeterminantOrbitals(
            n_electrons,
            n_up,
            n_determinants=network_config.get("determinant_count", 6),
            n_nuclei=nuclei_config["positions"].shape[0],
            config=network_config,
        )

        # Jastrow factor
        self.use_jastrow = network_config.get("use_jastrow", False)
        if self.use_jastrow:
            self.jastrow = JastrowFactor(
                n_electrons,
                hidden_dim=network_config.get("jastrow_hidden_dim", 32),
                n_layers=network_config.get("jastrow_layers", 2),
            )
        else:
            self.jastrow = None

        # Initialize params
        self.params = {}
        # Copy orbital params
        self.params.update(self.orbitals.params)
        # Add Jastrow params if needed
        if self.use_jastrow:
            self.params["jastrow"] = self.jastrow.params

    def multi_determinant_slater(self, orbitals_list, det_weights=None):
        """
        Calculate multi-determinant Slater combination using softmax-normalized weights.

        Weight combination formula (consistent with MultiDeterminantOrbitals):
        - Raw weights -> softmax -> normalized weights
        - log|psi| = log(sum_i w_i * |det_i|)
        - In log space: log|psi| = log(sum_i exp(log_w_i + log|det_i|))
        
        Each determinant is computed as the product of spin-up and spin-down determinants:
        - log|det_i| = log|det_up_i| + log|det_down_i|

        Args:
            orbitals_list: List of orbital matrices for each determinant
                           Each has shape [batch, n_elec, n_elec]
            det_weights: Optional raw weights. If None, uses self.params["det_weights"]

        Returns:
            Combined log|determinant| [batch]
        """
        # Compute log-determinant for each determinant (spin-separated)
        log_abs_dets = []

        for det_idx, orbitals in enumerate(orbitals_list):
            # Extract spin-up block (both electrons and orbitals): [batch, n_up, n_up]
            # This creates a square matrix for spin-up electrons occupying spin-up orbitals
            if self.orbitals.n_up > 0:
                spin_up_orbitals = orbitals[:, :self.orbitals.n_up, :self.orbitals.n_up]
                _, log_det_up = jax.vmap(jnp.linalg.slogdet)(spin_up_orbitals)
            else:
                log_det_up = jnp.zeros(orbitals.shape[0])
            
            # Extract spin-down block (both electrons and orbitals): [batch, n_down, n_down]
            # This creates a square matrix for spin-down electrons occupying spin-down orbitals
            if self.orbitals.n_down > 0:
                spin_down_orbitals = orbitals[:, self.orbitals.n_up:, self.orbitals.n_up:]
                _, log_det_down = jax.vmap(jnp.linalg.slogdet)(spin_down_orbitals)
            else:
                log_det_down = jnp.zeros(orbitals.shape[0])
            
            # Combined log determinant: log|det| = log|det_up| + log|det_down|
            log_abs_det = log_det_up + log_det_down
            log_abs_dets.append(log_abs_det)

        # Stack arrays: [n_determinants, batch] -> transpose to [batch, n_determinants]
        log_abs_det_stack = jnp.stack(log_abs_dets, axis=0).T

        # Get determinant weights and apply softmax normalization
        if det_weights is None:
            det_weights = self.params["det_weights"]
        
        # Softmax normalization of weights (ensures all weights are positive and sum to 1)
        log_weights = jax.nn.log_softmax(det_weights)

        # Add log weights to log determinants
        # log(weighted_det_i) = log_w_i + log|det_i|
        # Broadcasting: [batch, n_determinants] + [n_determinants] -> [batch, n_determinants]
        log_weighted_dets = log_abs_det_stack + log_weights  # [batch, n_determinants]

        # Log-sum-exp for stable combination
        # log(sum_i exp(log_weighted_dets_i))
        max_log = jnp.max(log_weighted_dets, axis=-1, keepdims=True)
        log_sum = max_log + jnp.log(
            jnp.sum(jnp.exp(log_weighted_dets - max_log), axis=-1, keepdims=True)
        )  # [batch, 1]

        return log_sum.squeeze(axis=-1)

    def _jastrow_apply(self, jastrow_params: Dict, r_elec: jnp.ndarray) -> jnp.ndarray:
        """Functional Jastrow forward pass.

        Computes the same value as `self.jastrow(r_elec)` but uses the provided
        `jastrow_params` without mutating `self.jastrow.params`.
        """
        distances = self.jastrow._compute_pairwise_distances(r_elec)
        upper_distances = self.jastrow._extract_upper_triangle(distances)
        x = upper_distances[:, :, None]

        # First layer
        x = jnp.dot(x, jastrow_params["w1"]) + jastrow_params["b1"]
        x = jnp.tanh(x)

        # Hidden layers
        for i in range(1, self.jastrow.n_layers):
            w = jastrow_params[f"w{i + 1}"]
            b = jastrow_params[f"b{i + 1}"]
            x = jnp.dot(x, w) + b
            x = jnp.tanh(x)

        # Output layer
        w_out = jastrow_params[f"w{self.jastrow.n_layers + 1}"]
        b_out = jastrow_params[f"b{self.jastrow.n_layers + 1}"]
        x = jnp.dot(x, w_out) + b_out

        x = x.squeeze(-1)
        return jnp.sum(x, axis=-1)

    def apply(self, params: Dict, r_elec: jnp.ndarray) -> jnp.ndarray:
        """Functional forward pass.

        This is the stateless counterpart to `__call__`: it computes log|psi|
        using the provided `params` without mutating `self.params`.

        Args:
            params: Parameter pytree.
            r_elec: Electron positions [batch, n_elec, 3].

        Returns:
            log|psi| [batch]
        """
        orbitals_list = []
        for i in range(self.orbitals.n_determinants):
            det_key = f"det_{i}"
            if det_key not in params:
                raise KeyError(f"Missing parameters for determinant {i}")
            det_params = params[det_key]
            orb = self.orbitals.compute_orbitals_single_det(
                r_elec, self.nuclei_config["positions"], det_params
            )
            orbitals_list.append(orb)

        log_psi = self.multi_determinant_slater(
            orbitals_list, det_weights=params["det_weights"]
        )

        if self.use_jastrow and self.jastrow:
            if "jastrow" in params:
                log_psi = log_psi + self._jastrow_apply(params["jastrow"], r_elec)

        return log_psi

    def __call__(self, r_elec):
        """
        Forward pass for wave function.

        Args:
            r_elec: Electron positions [batch, n_elec, 3]

        Returns:
            log|psi|: [batch]
        """
        # Generate orbitals list for each determinant
        orbitals_list = []
        for i in range(self.orbitals.n_determinants):
            det_key = f"det_{i}"
            if det_key in self.params:
                det_params = self.params[det_key]
                # Compute orbital matrices using MultiDeterminantOrbitals helper
                orb = self.orbitals.compute_orbitals_single_det(
                    r_elec, self.nuclei_config["positions"], det_params
                )
                orbitals_list.append(orb)
            else:
                raise KeyError(f"Missing parameters for determinant {i}")

        # Compute multi-determinant log psi using the stability-enhanced logic
        log_psi = self.multi_determinant_slater(orbitals_list)

        # Add Jastrow factor if enabled
        if self.use_jastrow and self.jastrow:
            if "jastrow" in self.params:
                # Sync jastrow params
                self.jastrow.params = self.params["jastrow"]
                j_val = self.jastrow(r_elec)
                log_psi = log_psi + j_val

        return log_psi
