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

    def get_network_info(self):
        """
        Returns information about the network.
        """
        total_parameters = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        # Base information always available.
        network_info = {
            'type': 'ExtendedFermiNet',
            'total_parameters': total_parameters,
        }

        # Additional metadata expected by extended debug tooling/tests.
        # Be defensive about attribute names to avoid AttributeError if the
        # underlying MultiDeterminantOrbitals implementation changes.
        determinant_count = getattr(
            self.orbitals,
            'n_determinants',
            getattr(self.orbitals, 'determinant_count', None),
        )
        network_info['determinant_count'] = determinant_count
        network_info['single_layer_width'] = getattr(
            self.orbitals, 'single_layer_width', None
        )
        network_info['pair_layer_width'] = getattr(
            self.orbitals, 'pair_layer_width', None
        )
        network_info['num_interaction_layers'] = getattr(
            self.orbitals, 'num_interaction_layers', None
        )

        return network_info
    def multi_determinant_slater(self, orbitals_list, det_weights=None):
        """
        Calculate multi-determinant Slater combination using Log-Sum-Exp trick.

        Args:
            orbitals_list: List of (orbitals_up, orbitals_down) for each determinant.
                           orbitals_up has shape [batch, n_up, n_up]
                           orbitals_down has shape [batch, n_down, n_down]

        Returns:
            Combined log|determinant| [batch]
        """
        # Compute log-determinant and sign for each determinant
        log_abs_dets = []
        signs = []

        for det_idx, (orb_up, orb_down) in enumerate(orbitals_list):
            # Compute log|det| for up and down spins
            if self.n_up > 0:
                sign_up, log_abs_det_up = jax.vmap(jnp.linalg.slogdet)(orb_up)
            else:
                sign_up, log_abs_det_up = jnp.ones(orb_up.shape[0]), jnp.zeros(orb_up.shape[0])

            if self.n_electrons - self.n_up > 0:
                sign_down, log_abs_det_down = jax.vmap(jnp.linalg.slogdet)(orb_down)
            else:
                sign_down, log_abs_det_down = jnp.ones(orb_up.shape[0]), jnp.zeros(orb_up.shape[0])

            # Total logabsdet = log|det_up| + log|det_down|
            # Total sign = sign_up * sign_down
            log_abs_dets.append(log_abs_det_up + log_abs_det_down)
            signs.append(sign_up * sign_down)

        # Stack arrays: [n_determinants, batch]
        log_abs_det_stack = jnp.stack(log_abs_dets, axis=0)
        sign_stack = jnp.stack(signs, axis=0)

        # Get determinant weights
        if det_weights is None:
            det_weights = self.params["det_weights"]
        # Expand det_weights to [n_determinants, batch]
        batch_size = log_abs_det_stack.shape[1]
        det_weights_expanded = jnp.tile(det_weights[:, None], (1, batch_size))

        # Combined weight sign and magnitude
        weight_signs = jnp.sign(det_weights_expanded)
        log_abs_weights = jnp.log(jnp.abs(det_weights_expanded) + 1e-20)

        total_log_terms = log_abs_weights + log_abs_det_stack
        total_signs = weight_signs * sign_stack

        # Log-Sum-Exp trick for stability
        max_log = jnp.max(total_log_terms, axis=0)  # [batch]
        exp_terms = jnp.exp(total_log_terms - max_log[None, :])
        weighted_sum = jnp.sum(total_signs * exp_terms, axis=0)  # [batch]

        log_psi = max_log + jnp.log(jnp.abs(weighted_sum) + 1e-20)

        return log_psi

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
            # Returns Tuple[orbitals_up, orbitals_down]
            orb_tuple = self.orbitals.compute_orbitals_single_det(
                r_elec, self.nuclei_config["positions"], det_params
            )
            orbitals_list.append(orb_tuple)

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
                # Returns Tuple[orbitals_up, orbitals_down]
                orb_tuple = self.orbitals.compute_orbitals_single_det(
                    r_elec, self.nuclei_config["positions"], det_params
                )
                orbitals_list.append(orb_tuple)
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
