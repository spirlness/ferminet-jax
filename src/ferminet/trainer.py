"""
FermiNet Stage 1 - Variational Monte Carlo Trainer
Implements Energy Loss Function and Adam Optimizer
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Tuple, Dict, Callable

from .physics import make_batched_local_energy
from .scheduler import EnergyBasedScheduler


class VMCTrainer:
    """Variational Monte Carlo Trainer"""

    def __init__(self, network, mcmc, config: Dict):
        """
        Initialize trainer
        """
        self.network = network
        self.mcmc = mcmc
        self.config = config

        # Adam optimizer hyperparameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.beta1 = config.get("beta1", 0.9)
        self.beta2 = config.get("beta2", 0.999)
        self.epsilon = config.get("epsilon", 1e-8)

        # Initialize Adam optimizer state
        self.adam_state = self._init_adam_state(network.params)

        # Batched local energy function (no param mutation)
        def network_forward(params, r_batch):
            return self.network.apply(params, r_batch)

        self._batched_local_energy = make_batched_local_energy(
            network_forward, n_electrons=self.network.n_electrons
        )

        # Pre-compute gradient function for MCMC
        def single_network_forward(params, r_single):
            # network.apply expects [batch, ...]
            # r_single is [n_elec, 3]
            return self.network.apply(params, r_single[None, ...])[0]

        self._grad_single = jax.grad(single_network_forward, argnums=1)
        self._batched_grad_log_psi = jax.vmap(self._grad_single, in_axes=(None, 0))

        # Pre-build grad transform for update step
        self._loss_and_grad = jax.value_and_grad(self.energy_loss, has_aux=True)

        # JIT compiled update function
        self._jit_update = jax.jit(self._update_step)

    def _init_adam_state(self, params: Dict) -> Dict:
        """
        Initialize Adam optimizer state (m and v moments)
        """
        m = jtu.tree_map(jnp.zeros_like, params)
        v = jtu.tree_map(jnp.zeros_like, params)
        return {"m": m, "v": v, "t": jnp.array(0, dtype=jnp.int32)}

    def _make_log_psi_fn(self, params: Dict) -> Callable:
        """
        Create log wave function using given parameters
        """

        def log_psi_fn(r_elec):
            return self.network.apply(params, r_elec)

        return log_psi_fn

    def energy_loss(
        self,
        params: Dict,
        r_elec: jnp.ndarray,
        nuclei_pos: jnp.ndarray,
        nuclei_charge: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Energy loss function
        """
        local_E = self._batched_local_energy(params, r_elec, nuclei_pos, nuclei_charge)

        # Check for NaN/Inf - JAX SAFE
        nan_mask = jnp.isnan(local_E)
        valid_mask = ~nan_mask

        # Compute mean of valid entries (safe division)
        # If all are NaN, mean is 0
        safe_sum = jnp.sum(jnp.where(valid_mask, local_E, 0.0))
        safe_count = jnp.sum(valid_mask)
        mean_valid = jnp.where(safe_count > 0, safe_sum / safe_count, 0.0)

        # Replace NaNs
        local_E = jnp.where(nan_mask, mean_valid, local_E)

        # Handle Infs
        # Just clip everything, it handles Infs naturally
        local_E = jnp.clip(local_E, -1e6, 1e6)

        mean_E = jnp.mean(local_E)
        loss = jnp.mean((local_E - mean_E) ** 2)

        # Ensure loss is finite
        loss = jnp.where(jnp.isnan(loss), jnp.array(0.0), loss)
        loss = jnp.where(jnp.isinf(loss), jnp.array(1e6), loss)

        return loss, mean_E

    def _adam_update(
        self, params: Dict, grads: Dict, state: Dict, learning_rate: float = None
    ) -> Tuple[Dict, Dict]:
        """
        Adam optimizer update - handles nested parameter structures
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        m = state["m"]
        v = state["v"]
        t = state["t"] + jnp.array(1, dtype=state["t"].dtype)

        m_new = jtu.tree_map(
            lambda m_i, g_i: self.beta1 * m_i + (1 - self.beta1) * g_i,
            m,
            grads,
        )
        v_new = jtu.tree_map(
            lambda v_i, g_i: self.beta2 * v_i + (1 - self.beta2) * (g_i**2),
            v,
            grads,
        )

        m_hat = jtu.tree_map(lambda m_i: m_i / (1 - self.beta1**t), m_new)
        v_hat = jtu.tree_map(lambda v_i: v_i / (1 - self.beta2**t), v_new)

        params_new = jtu.tree_map(
            lambda p_i, m_i, v_i: p_i - lr * m_i / (jnp.sqrt(v_i) + self.epsilon),
            params,
            m_hat,
            v_hat,
        )

        state_new = {"m": m_new, "v": v_new, "t": t}
        return params_new, state_new

    def _update_step(self, params, r_elec, nuclei_pos, nuclei_charge, state, lr):
        """
        JIT-compilable update step: gradients + optimizer
        """
        (loss, mean_E), grads = self._loss_and_grad(
            params, r_elec, nuclei_pos, nuclei_charge
        )

        # If ExtendedTrainer features like clipping are needed, they should be applied here.
        # Since this is the base class, we just update.
        # NOTE: For ExtendedTrainer, we need to override this or pass a clip function.

        params_new, state_new = self._adam_update(params, grads, state, lr)
        return params_new, state_new, mean_E, grads

    def warmup(
        self,
        params: Dict,
        r_elec: jnp.ndarray,
        nuclei_pos: jnp.ndarray,
        nuclei_charge: jnp.ndarray,
        learning_rate: float = None,
    ) -> None:
        """Compile the steady-state update path.

        Call this once before timing/benchmarking to pay the first-step compile
        cost up front.
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        out = self._jit_update(
            params, r_elec, nuclei_pos, nuclei_charge, self.adam_state, lr
        )
        # mean_E is always at index 2 for both base and extended update fns.
        jax.block_until_ready(out[2])

    def train_step(
        self,
        params: Dict,
        r_elec: jnp.ndarray,
        key: jnp.ndarray,
        nuclei_pos: jnp.ndarray,
        nuclei_charge: jnp.ndarray,
    ) -> Tuple[Dict, float, float, jnp.ndarray]:
        """
        Execute single training step
        """

        # 1. MCMC Sampling (outside JIT usually, or JITted separately)
        # Creating a functional log_psi for MCMC
        def log_psi_fn(r):
            return self.network.apply(params, r)

        def grad_log_psi_fn(r):
            return self._batched_grad_log_psi(params, r)

        r_elec_new, accept_rate = self.mcmc.sample(
            log_psi_fn, r_elec, key, grad_log_psi_fn=grad_log_psi_fn
        )

        # 2. Update parameters (JITted)
        # Use current learning rate
        lr = self.learning_rate

        params_new, self.adam_state, mean_E, _ = self._jit_update(
            params, r_elec_new, nuclei_pos, nuclei_charge, self.adam_state, lr
        )

        return params_new, mean_E, accept_rate, r_elec_new

    def get_training_info(self) -> Dict:
        """Get training info"""
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "adam_step": self.adam_state["t"],
        }


class ExtendedTrainer(VMCTrainer):
    """
    Extended Trainer with scheduler, gradient clipping, etc.
    """

    def __init__(self, network, mcmc, config: Dict):
        super().__init__(network, mcmc, config)

        self.gradient_clip = config.get("gradient_clip", 1.0)
        self.gradient_clip_norm = config.get("gradient_clip_norm", "inf")

        self.use_scheduler = config.get("use_scheduler", True)
        if self.use_scheduler:
            target_energy = config.get("target_energy", -1.174)
            self.scheduler = EnergyBasedScheduler(
                initial_lr=self.learning_rate,
                target_energy=target_energy,
                patience=config.get("scheduler_patience", 10),
                decay_factor=config.get("decay_factor", 0.5),
                min_lr=config.get("min_lr", 1e-5),
            )
        else:
            self.scheduler = None

        self.energy_history = []
        self.variance_history = []
        self.accept_rate_history = []

        # Override JIT update to include clipping
        self._jit_update = jax.jit(self._extended_update_step)

    def _clip_gradients(
        self, grads: Dict, max_norm: float = 1.0, norm_type: str = "inf"
    ) -> Tuple[Dict, float]:
        """
        Clip gradients
        """
        grad_tensors = [jnp.ravel(g) for g in jtu.tree_leaves(grads)]
        stacked_grads = (
            jnp.concatenate(grad_tensors) if grad_tensors else jnp.array([0.0])
        )

        if norm_type == "inf":
            grad_norm = jnp.max(jnp.abs(stacked_grads))
        elif norm_type == "l2":
            grad_norm = jnp.linalg.norm(stacked_grads, ord=2)
        elif norm_type == "l1":
            grad_norm = jnp.sum(jnp.abs(stacked_grads))
        else:
            # Fallback (should not happen in JIT if validated)
            grad_norm = jnp.max(jnp.abs(stacked_grads))

        # We need a JIT-compatible branch
        # jax.lax.cond would be better, but python if/else works with tracing if condition depends on data
        # Here condition depends on data (grad_norm), so we need jnp.where or lax.cond
        # But for dicts of params, it's easier to multiply by factor.

        scale = jnp.where(grad_norm > max_norm, max_norm / (grad_norm + 1e-10), 1.0)

        clipped_grads = jtu.tree_map(lambda g: g * scale, grads)

        return clipped_grads, grad_norm

    def _extended_update_step(
        self, params, r_elec, nuclei_pos, nuclei_charge, state, lr
    ):
        """
        Extended update step with clipping
        """
        (loss, mean_E), grads = self._loss_and_grad(
            params, r_elec, nuclei_pos, nuclei_charge
        )

        clipped_grads, grad_norm = self._clip_gradients(
            grads, self.gradient_clip, self.gradient_clip_norm
        )

        params_new, state_new = self._adam_update(params, clipped_grads, state, lr)

        return params_new, state_new, mean_E, grad_norm, loss

    def train_step(
        self,
        params: Dict,
        r_elec: jnp.ndarray,
        key: jnp.ndarray,
        nuclei_pos: jnp.ndarray,
        nuclei_charge: jnp.ndarray,
    ) -> Tuple[Dict, float, float, jnp.ndarray, Dict]:
        """
        Execute extended training step
        """

        # 1. MCMC
        def log_psi_fn(r):
            return self.network.apply(params, r)

        def grad_log_psi_fn(r):
            return self._batched_grad_log_psi(params, r)

        r_elec_new, accept_rate = self.mcmc.sample(
            log_psi_fn, r_elec, key, grad_log_psi_fn=grad_log_psi_fn
        )

        # 2. Update with JIT
        current_lr = (
            self.scheduler.get_lr() if self.use_scheduler else self.learning_rate
        )

        params_new, self.adam_state, mean_E, grad_norm, loss = self._jit_update(
            params, r_elec_new, nuclei_pos, nuclei_charge, self.adam_state, current_lr
        )

        # 3. Info
        train_info = {
            "loss": float(loss),
            "energy": float(mean_E),
            "accept_rate": float(accept_rate),
            "grad_norm": float(grad_norm),
            "learning_rate": float(current_lr),
        }

        return params_new, mean_E, accept_rate, r_elec_new, train_info

    def update_scheduler(self, current_energy):
        """Update scheduler"""
        if self.use_scheduler:
            new_lr, decayed, old_lr = self.scheduler.step(current_energy)
            if decayed:
                print(f"  -> Learning rate decayed from {old_lr:.6f} to {new_lr:.6f}")
            return new_lr, decayed, old_lr
        return self.learning_rate, False, None

    def record_training_stats(self, energy, variance, accept_rate):
        """Record stats"""
        self.energy_history.append(energy)
        self.variance_history.append(variance)
        self.accept_rate_history.append(accept_rate)

    def get_training_info(self) -> Dict:
        """Get extended info"""
        info = super().get_training_info()
        info.update(
            {
                "gradient_clip": self.gradient_clip,
                "gradient_clip_norm": self.gradient_clip_norm,
                "use_scheduler": self.use_scheduler,
            }
        )

        if self.use_scheduler:
            info["scheduler"] = self.scheduler.get_info()

        if len(self.energy_history) > 0:
            info["energy_history"] = self.energy_history
            info["variance_history"] = self.variance_history
            info["accept_rate_history"] = self.accept_rate_history
            info["best_energy"] = min(self.energy_history)
            info["last_energy"] = self.energy_history[-1]

        return info


if __name__ == "__main__":
    print("=" * 60)
    print("VMCTrainer Test")
    print("=" * 60)

    # Basic smoke test for syntax errors
    print("Module loaded successfully.")
