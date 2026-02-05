"""FermiNet neural network architecture (functional).

This module implements a lightweight, production-oriented FermiNet using the
DeepMind functional init/apply pattern. The network follows the standard
two-stream design:

- One-electron stream with electron-nuclear features.
- Two-electron stream with electron-electron features.
- Interaction layers that mix one- and two-electron information.
- Multi-determinant Slater determinants with an envelope placeholder.

The factory function returns `(init, apply, orbitals)` where:
- init(key) -> ParamTree
- apply(params, electrons, spins, atoms, charges) -> (sign, log_psi)
- orbitals(params, electrons, spins, atoms, charges) -> list of orbital tensors

This implementation is intentionally modular and avoids class-based state. It
relies only on local modules and JAX primitives, leaving JIT and batching to the
caller.
"""

# pyright: reportMissingTypeStubs=false, reportUnusedImport=false
# pyright: reportDeprecated=false, reportUnusedVariable=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportUnnecessaryIsInstance=false, reportUnnecessaryCast=false

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import ml_collections

from ferminet import base_config
from ferminet import constants
from ferminet import network_blocks
from ferminet import types


Array = jnp.ndarray
ParamTree = types.ParamTree
ParamMapping = Mapping[str, types.ParamTree]
PMAP_AXIS_NAME = constants.PMAP_AXIS_NAME


def _resolve_config(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
    """Resolve FieldReferences in the configuration."""
    return base_config.resolve(cfg)


def _cfg_section(
    cfg: ml_collections.ConfigDict, name: str
) -> ml_collections.ConfigDict | None:
    """Safely fetch a nested ConfigDict section."""
    if isinstance(cfg, ml_collections.ConfigDict) and name in cfg:
        section = cfg[name]
        if isinstance(section, ml_collections.ConfigDict):
            return section
    return None


def _get_ndim(cfg: ml_collections.ConfigDict, atoms: Array) -> int:
    """Infer spatial dimension from config or atoms array."""
    system_cfg = _cfg_section(cfg, "system")
    if system_cfg is not None:
        try:
            ndim_value = system_cfg["ndim"]
        except KeyError:
            ndim_value = None
        if isinstance(ndim_value, (int, float)):
            return int(cast(float, ndim_value))
    if atoms.ndim == 2:
        return int(atoms.shape[-1])
    return 3


def _normalize_atoms(atoms: Array, ndim: int) -> Array:
    """Ensure atomic positions are shaped as (n_atoms, ndim)."""
    if atoms.ndim == 1:
        n_atoms = atoms.shape[0] // ndim
        return atoms.reshape(n_atoms, ndim)
    if atoms.ndim == 2 and atoms.shape[-1] == ndim:
        return atoms
    raise ValueError("atoms must have shape (n_atoms, ndim) or (n_atoms*ndim,)")


def _normalize_electrons(
    electrons: Array,
    n_electrons: int,
    ndim: int,
) -> Array:
    """Normalize electron positions into shape (n_electrons, ndim) or batched.

    Accepted shapes:
    - (n_electrons*ndim,)
    - (n_electrons, ndim)
    - (batch, n_electrons, ndim)
    - (batch, n_electrons*ndim)
    """
    if electrons.ndim == 1:
        return electrons.reshape(n_electrons, ndim)
    if electrons.ndim == 2:
        if electrons.shape[1] == ndim and electrons.shape[0] == n_electrons:
            return electrons
        if electrons.shape[1] == n_electrons * ndim:
            return electrons.reshape(electrons.shape[0], n_electrons, ndim)
    if electrons.ndim == 3:
        return electrons
    raise ValueError(
        "electrons must have shape (n_electrons*ndim,), "
        "(n_electrons, ndim), (batch, n_electrons, ndim), or "
        "(batch, n_electrons*ndim)"
    )


def _activation_from_name(name: str) -> Callable[[Array], Array]:
    """Return activation function by name."""
    name = name.lower()
    if name == "tanh":
        return jnp.tanh
    if name == "gelu":
        return jax.nn.gelu
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported activation: {name}")


def _pairwise_electron_nuclear_vectors(electrons: Array, atoms: Array) -> Array:
    """Compute electron-nuclear displacement vectors with vmap.

    Args:
        electrons: Array of shape (n_electrons, ndim).
        atoms: Array of shape (n_atoms, ndim).

    Returns:
        Displacements of shape (n_electrons, n_atoms, ndim).
    """
    return jax.vmap(lambda e: e - atoms)(electrons)


def _pairwise_electron_electron_vectors(electrons: Array) -> Array:
    """Compute electron-electron displacement vectors with vmap.

    Args:
        electrons: Array of shape (n_electrons, ndim).

    Returns:
        Displacements of shape (n_electrons, n_electrons, ndim).
    """
    return jax.vmap(lambda e: e - electrons)(electrons)


def _construct_one_electron_features(r_ae: Array, r_ae_norm: Array) -> Array:
    """Construct one-electron features from electron-nuclear vectors."""
    features = jnp.concatenate([r_ae, r_ae_norm[..., None]], axis=-1)
    return features.reshape(r_ae.shape[0], -1)


def _construct_two_electron_features(r_ee: Array, r_ee_norm: Array) -> Array:
    """Construct two-electron features from electron-electron vectors."""
    return jnp.concatenate([r_ee, r_ee_norm[..., None]], axis=-1)


def _electron_electron_mask(n_electrons: int) -> Array:
    """Mask to exclude self-interactions in two-electron features."""
    eye = jnp.eye(n_electrons)
    return 1.0 - eye


def _masked_mean(values: Array, mask: Array) -> Array:
    """Compute masked mean over axis=1.

    Args:
        values: Array of shape (n, n, feat).
        mask: Array of shape (n, n).
    """
    mask_expanded = mask[..., None]
    summed = jnp.sum(values * mask_expanded, axis=1)
    denom = jnp.sum(mask_expanded, axis=1)
    return summed / jnp.maximum(denom, 1.0)


def _init_interaction_layer(
    key: jax.Array,
    in_dim_one: int,
    in_dim_two: int,
    out_dim_one: int,
    out_dim_two: int,
) -> dict[str, dict[str, Array]]:
    """Initialize parameters for a single interaction layer."""
    keys = jax.random.split(key, 3)
    key, key_one, key_two = keys[0], keys[1], keys[2]
    combined_dim = in_dim_one + in_dim_one + in_dim_two
    one_params: dict[str, Array] = dict(
        network_blocks.init_linear_layer(key_one, combined_dim, out_dim_one)
    )
    two_params: dict[str, Array] = dict(
        network_blocks.init_linear_layer(key_two, in_dim_two, out_dim_two)
    )
    return {"one": one_params, "two": two_params}


def _apply_interaction_layer(
    params: Mapping[str, Mapping[str, Array]],
    h_one: Array,
    h_two: Array,
    mask: Array,
    activation: Callable[[Array], Array],
    use_residual: bool,
) -> tuple[Array, Array]:
    """Apply a single interaction layer to one- and two-electron streams."""
    h_one_mean = jnp.mean(h_one, axis=0, keepdims=True)
    h_one_mean = jnp.broadcast_to(h_one_mean, h_one.shape)
    h_two_mean = _masked_mean(h_two, mask)

    h_one_input = jnp.concatenate([h_one, h_one_mean, h_two_mean], axis=-1)
    one_params = params["one"]
    two_params = params["two"]
    one_bias = one_params["b"] if "b" in one_params else None
    two_bias = two_params["b"] if "b" in two_params else None
    h_one_new = activation(
        network_blocks.linear_layer(h_one_input, one_params["w"], one_bias)
    )
    h_two_new = activation(
        network_blocks.linear_layer(h_two, two_params["w"], two_bias)
    )

    if use_residual and h_one_new.shape == h_one.shape:
        h_one_new = h_one_new + h_one
    if use_residual and h_two_new.shape == h_two.shape:
        h_two_new = h_two_new + h_two

    return h_one_new, h_two_new


def _init_orbital_layer(
    key: jax.Array,
    in_dim: int,
    n_spin: int,
    n_determinants: int,
    include_bias: bool,
) -> Mapping[str, Array]:
    """Initialize orbital projection for a single spin channel."""
    if n_spin == 0:
        return {}
    return network_blocks.init_linear_layer(
        key, in_dim, n_determinants * n_spin, include_bias=include_bias
    )


def _apply_orbital_layer(
    params: Mapping[str, Array],
    h_spin: Array,
    n_spin: int,
    n_determinants: int,
) -> Array:
    """Project features into orbital matrices for a spin channel."""
    if n_spin == 0:
        return jnp.zeros((n_determinants, 0, 0))
    bias = params["b"] if "b" in params else None
    projected = network_blocks.linear_layer(h_spin, params["w"], bias)
    projected = projected.reshape(n_spin, n_determinants, n_spin)
    return jnp.transpose(projected, (1, 0, 2))


def _slogdet_safe(orbitals: Array, n_spin: int) -> tuple[Array, Array]:
    """Compute slogdet with graceful handling of empty spins."""
    if n_spin == 0:
        n_det = orbitals.shape[0]
        return jnp.ones((n_det,)), jnp.zeros((n_det,))
    return jnp.linalg.slogdet(orbitals)


def _combine_determinants(
    orbitals_up: Array,
    orbitals_down: Array,
    n_up: int,
    n_down: int,
    eps: float = 1.0e-12,
) -> tuple[Array, Array]:
    """Combine multi-determinant Slater determinants.

    Returns:
        sign: Scalar sign of the wavefunction.
        log_abs: Log absolute value of the wavefunction.
    """
    sign_up, logdet_up = _slogdet_safe(orbitals_up, n_up)
    sign_down, logdet_down = _slogdet_safe(orbitals_down, n_down)

    det_signs = sign_up * sign_down
    det_logs = logdet_up + logdet_down

    max_log = jnp.max(det_logs)
    weighted = jnp.sum(det_signs * jnp.exp(det_logs - max_log))
    total_sign = jnp.sign(weighted)
    log_abs = max_log + jnp.log(jnp.abs(weighted) + eps)

    return total_sign, log_abs


def _apply_envelope(r_ae_norm: Array, sigma: Array, eps: float = 1.0e-12) -> Array:
    """Apply isotropic Gaussian envelope: exp(-sigma * |r_ae|)."""
    decay = jnp.exp(-r_ae_norm * sigma[None, :])
    envelope = jnp.sum(decay, axis=1)
    return jnp.sum(jnp.log(jnp.maximum(envelope, eps)))


def _hidden_dims_from_cfg(
    cfg: ml_collections.ConfigDict,
) -> tuple[tuple[int, int], ...]:
    """Extract hidden dimension pairs from the config."""
    network_cfg = _cfg_section(cfg, "network")
    if network_cfg is not None:
        ferminet_cfg = _cfg_section(network_cfg, "ferminet")
        default_dims = ((256, 32), (256, 32), (256, 32), (256, 32))
        hidden_dims: object = default_dims
        if ferminet_cfg is not None:
            try:
                hidden_dims = ferminet_cfg["hidden_dims"]
            except KeyError:
                hidden_dims = default_dims
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = default_dims
        hidden_dims = cast(Sequence[Sequence[int]], hidden_dims)
        dims: list[tuple[int, int]] = []
        for layer in hidden_dims:
            if not isinstance(layer, Sequence) or len(layer) != 2:
                raise ValueError("Each hidden_dims entry must have length 2")
            first = layer[0]
            second = layer[1]
            if not isinstance(first, (int, float)) or not isinstance(
                second, (int, float)
            ):
                raise ValueError("Hidden dims must be numeric")
            dims.append((int(first), int(second)))
        return tuple(dims)
    return ((256, 32), (256, 32), (256, 32), (256, 32))


def _determinants_from_cfg(cfg: ml_collections.ConfigDict) -> int:
    """Extract number of determinants from the config."""
    network_cfg = _cfg_section(cfg, "network")
    if network_cfg is not None:
        ferminet_cfg = _cfg_section(network_cfg, "ferminet")
        det_value: object = None
        if ferminet_cfg is not None:
            try:
                det_value = ferminet_cfg["determinants"]
            except KeyError:
                det_value = None
        if det_value is None:
            try:
                det_value = network_cfg["determinants"]
            except KeyError:
                det_value = None
        if isinstance(det_value, (int, float)):
            return int(det_value)
    return 16


def _bias_orbitals_from_cfg(cfg: ml_collections.ConfigDict) -> bool:
    """Extract orbital bias flag from the config."""
    network_cfg = _cfg_section(cfg, "network")
    if network_cfg is not None:
        ferminet_cfg = _cfg_section(network_cfg, "ferminet")
        if ferminet_cfg is not None:
            try:
                bias_value = ferminet_cfg["bias_orbitals"]
            except KeyError:
                bias_value = None
            if bias_value is not None:
                return bool(bias_value)
    return True


def _activation_from_cfg(cfg: ml_collections.ConfigDict) -> Callable[[Array], Array]:
    """Get activation function from configuration."""
    name = "tanh"
    network_cfg = _cfg_section(cfg, "network")
    if network_cfg is not None:
        ferminet_cfg = _cfg_section(network_cfg, "ferminet")
        if ferminet_cfg is not None:
            try:
                activation_value = ferminet_cfg["hidden_activation"]
            except KeyError:
                activation_value = None
            if activation_value is not None:
                name = str(activation_value)
    return _activation_from_name(name)


def _envelope_sigma_from_cfg(cfg: ml_collections.ConfigDict) -> float:
    """Extract isotropic envelope sigma from the config."""
    network_cfg = _cfg_section(cfg, "network")
    if network_cfg is not None:
        envelope_cfg = _cfg_section(network_cfg, "envelope")
        if envelope_cfg is not None:
            iso_cfg = _cfg_section(envelope_cfg, "isotropic")
            if iso_cfg is not None:
                try:
                    sigma_value = iso_cfg["sigma"]
                except KeyError:
                    sigma_value = None
                if isinstance(sigma_value, (int, float)):
                    return float(sigma_value)
    return 1.0


def make_fermi_net(
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    spins: tuple[int, int],
    cfg: ml_collections.ConfigDict,
) -> tuple[types.InitFermiNet, types.FermiNetLike, types.OrbitalFnLike]:
    """Create FermiNet init/apply/orbitals functions.

    Args:
        atoms: Atomic positions, shape (n_atoms, ndim) or flattened.
        charges: Atomic charges, shape (n_atoms,).
        spins: Tuple of (n_up, n_down).
        cfg: Configuration dictionary.

    Returns:
        (init, apply, orbitals) tuple of callables.
    """
    del charges
    cfg = _resolve_config(cfg)
    ndim = _get_ndim(cfg, atoms)
    atoms = _normalize_atoms(atoms, ndim)
    n_atoms = atoms.shape[0]
    n_up, n_down = spins
    n_electrons = n_up + n_down

    hidden_dims = _hidden_dims_from_cfg(cfg)
    n_layers = len(hidden_dims)
    n_determinants = _determinants_from_cfg(cfg)
    bias_orbitals = _bias_orbitals_from_cfg(cfg)
    activation = _activation_from_cfg(cfg)
    sigma_init = _envelope_sigma_from_cfg(cfg)

    one_feat_dim = n_atoms * (ndim + 1)
    two_feat_dim = ndim + 1

    def init(key: jax.Array) -> ParamTree:
        """Initialize FermiNet parameters."""
        params: dict[str, types.ParamTree] = {}
        keys = jax.random.split(key)
        key = keys[0]
        layer_key = keys[1]

        layers: list[ParamTree] = []
        in_dim_one = one_feat_dim
        in_dim_two = two_feat_dim
        for out_one, out_two in hidden_dims:
            keys = jax.random.split(layer_key)
            layer_key = keys[0]
            subkey = keys[1]
            layer_params = _init_interaction_layer(
                subkey, in_dim_one, in_dim_two, out_one, out_two
            )
            layers.append(cast(ParamTree, layer_params))
            in_dim_one = out_one
            in_dim_two = out_two

        keys = jax.random.split(key, 3)
        key = keys[0]
        key_up = keys[1]
        key_down = keys[2]
        orbitals: dict[str, ParamTree] = {
            "up": cast(
                ParamTree,
                _init_orbital_layer(
                    key_up, in_dim_one, n_up, n_determinants, bias_orbitals
                ),
            ),
            "down": cast(
                ParamTree,
                _init_orbital_layer(
                    key_down, in_dim_one, n_down, n_determinants, bias_orbitals
                ),
            ),
        }

        params["layers"] = cast(ParamTree, layers)
        params["orbitals"] = cast(ParamTree, orbitals)
        params["envelope_sigma"] = jnp.ones((n_atoms,)) * sigma_init

        return cast(ParamTree, params)

    def _forward_single(
        params: ParamMapping,
        electrons_single: Array,
        atoms_in: Array,
    ) -> tuple[Array, Array, tuple[Array, Array]]:
        """Forward pass for a single configuration."""
        params_map = params
        r_ae = _pairwise_electron_nuclear_vectors(electrons_single, atoms_in)
        r_ee = _pairwise_electron_electron_vectors(electrons_single)
        # Use an epsilon-stabilized norm to avoid NaN gradients at zero distance
        # (notably the r_ee diagonal self-distances).
        eps = jnp.asarray(1.0e-12, dtype=electrons_single.dtype)
        r_ae_norm = jnp.sqrt(jnp.sum(r_ae**2, axis=-1) + eps)
        r_ee_norm = jnp.sqrt(jnp.sum(r_ee**2, axis=-1) + eps)

        h_one = _construct_one_electron_features(r_ae, r_ae_norm)
        h_two = _construct_two_electron_features(r_ee, r_ee_norm)
        mask = _electron_electron_mask(n_electrons)

        layers = cast(Sequence[Mapping[str, Mapping[str, Array]]], params_map["layers"])
        for layer_index, layer_params in enumerate(layers):
            h_one, h_two = _apply_interaction_layer(
                layer_params,
                h_one,
                h_two,
                mask,
                activation,
                use_residual=layer_index > 0,
            )

        h_up = h_one[:n_up]
        h_down = h_one[n_up:]

        orbitals = cast(Mapping[str, Mapping[str, Array]], params_map["orbitals"])
        orb_up = _apply_orbital_layer(orbitals["up"], h_up, n_up, n_determinants)
        orb_down = _apply_orbital_layer(
            orbitals["down"], h_down, n_down, n_determinants
        )

        sign, log_det = _combine_determinants(orb_up, orb_down, n_up, n_down)
        sigma = cast(Array, params_map["envelope_sigma"])
        log_env = _apply_envelope(r_ae_norm, sigma)
        log_psi = log_det + log_env

        return sign, log_psi, (orb_up, orb_down)

    def apply(
        params: ParamTree,
        electrons: Array,
        spins: Array,
        atoms: Array,
        charges: Array,
    ) -> tuple[Array, Array]:
        """Apply FermiNet to input electrons.

        Args:
            params: Parameter tree from init.
            electrons: Electron positions.
            spins: Spin labels (unused, spin counts are fixed).
            atoms: Atomic positions.
            charges: Atomic charges (unused).
        """
        del spins, charges
        params_map: ParamMapping = cast(ParamMapping, params)
        atoms_norm = _normalize_atoms(atoms, ndim)
        electrons_norm = _normalize_electrons(electrons, n_electrons, ndim)

        if electrons_norm.ndim == 2:
            sign, log_psi, _ = _forward_single(params_map, electrons_norm, atoms_norm)
            return sign, log_psi

        if electrons_norm.ndim == 3:
            vmapped = jax.vmap(lambda e: _forward_single(params_map, e, atoms_norm)[:2])
            sign_batch, log_batch = vmapped(electrons_norm)
            return sign_batch, log_batch

        raise ValueError("Invalid electron shape after normalization")

    def orbitals(
        params: ParamTree,
        pos: Array,
        spins: Array,
        atoms: Array,
        charges: Array,
    ) -> Sequence[Array]:
        """Return per-spin orbital matrices for the given positions."""
        del spins, charges
        params_map: ParamMapping = cast(ParamMapping, params)
        atoms_norm = _normalize_atoms(atoms, ndim)
        electrons_norm = _normalize_electrons(pos, n_electrons, ndim)

        if electrons_norm.ndim == 2:
            _, _, (orb_up, orb_down) = _forward_single(
                params_map, electrons_norm, atoms_norm
            )
            return (orb_up, orb_down)

        if electrons_norm.ndim == 3:
            vmapped = jax.vmap(lambda e: _forward_single(params_map, e, atoms_norm)[2])
            orb_up, orb_down = vmapped(electrons_norm)
            return (orb_up, orb_down)

        raise ValueError("Invalid electron shape after normalization")

    return init, apply, orbitals


def make_log_psi_apply(apply_fn: types.FermiNetLike) -> types.LogFermiNetLike:
    """Wrap a FermiNet apply function to return only log|psi|."""

    def log_psi(
        params: ParamTree,
        electrons: Array,
        spins: Array,
        atoms: Array,
        charges: Array,
    ) -> Array:
        _, log_value = apply_fn(params, electrons, spins, atoms, charges)
        return log_value

    return log_psi
