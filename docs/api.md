# API 参考

本文只覆盖仓库当前实现的核心 public API（以 `ferminet/` 为准）。

## 配置

- `ferminet/base_config.py:default() -> ml_collections.ConfigDict`
- `ferminet/base_config.py:resolve(cfg) -> ml_collections.ConfigDict`

示例配置：`ferminet/configs/*.py` 中提供 `get_config()`。

## 网络

- `ferminet/networks.py:make_fermi_net(atoms, charges, nspins, cfg)`
  - 输入
    - `atoms: jnp.ndarray`，形状 `(n_atoms, ndim)`
    - `charges: jnp.ndarray`，形状 `(n_atoms,)`
    - `nspins: tuple[int, int]`，(n_up, n_down)
    - `cfg: ConfigDict`
  - 输出 `(init_fn, apply_fn, orbitals_fn)`
    - `init_fn(key) -> ParamTree`
    - `apply_fn(params, electrons, spins, atoms, charges) -> (sign, log_psi)`
      - `electrons` 支持 `(n_electrons*ndim,)` 或 `(batch, n_electrons*ndim)` 等
    - `orbitals_fn(...) -> tuple[orbitals_up, orbitals_down]`

- `ferminet/networks.py:make_log_psi_apply(apply_fn)`
  - 将 `apply_fn` 包装为 `log_psi_fn(params, electrons, spins, atoms, charges) -> log|psi|`

## 局部能量 / 哈密顿量

- `ferminet/hamiltonian.py:local_energy(f, charges, nspins, use_scan=False, complex_output=False)`
  - 输入 `f`：网络函数，签名与 `apply_fn` 一致，返回 `(sign, log|psi|)`
  - 输出 `LocalEnergy`：`(params, key, data) -> (E_local, grad_local_energy|None)`

## MCMC

- `ferminet/mcmc.py:make_mcmc_step(batch_network, batch_per_device, steps=10, atoms=None, ndim=3)`
  - `batch_network`：返回 `log|psi|`，输入 `positions` 为批量
  - 输出 `mcmc_step(params, data, key, width) -> (new_data, pmove)`
  - `pmove`：接受率估计（0~1）

## Loss

- `ferminet/loss.py:make_loss(network, local_energy_fn, clip_local_energy=5.0)`
  - `network`：`(params, positions, spins, atoms, charges) -> log|psi|`（建议用 `make_log_psi_apply` 得到）
  - `local_energy_fn`：`(params, key, data) -> local_energy(batch,)`
  - 返回：`loss_fn(params, key, data) -> (loss, aux)`

## Train

- `ferminet/train.py:train(cfg) -> Mapping[str, Any]`
  - 负责组装网络、MCMC、loss 与优化器；并进行日志与 checkpoint。

## 最小组装示例（伪代码）

```python
import jax
import jax.numpy as jnp

from ferminet.configs import helium
from ferminet import base_config, networks, hamiltonian, loss, mcmc
from ferminet.types import FermiNetData

cfg = base_config.resolve(helium.get_config())
atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([2.0])
nspins = (1, 1)
spins_arr = jnp.array([0, 1])

init_fn, apply_fn, _ = networks.make_fermi_net(atoms, charges, nspins, cfg)
params = init_fn(jax.random.PRNGKey(0))

log_psi = networks.make_log_psi_apply(apply_fn)

# local_energy 单样本函数 -> vmap 到 batch
single_el = hamiltonian.local_energy(apply_fn, charges=charges, nspins=nspins)
def local_energy_fn(params, key, data: FermiNetData):
    def per_config(pos):
        sample = FermiNetData(pos, data.spins, data.atoms, data.charges)
        e, _ = single_el(params, key, sample)
        return e
    return jax.vmap(per_config)(data.positions)

loss_fn = loss.make_loss(log_psi, local_energy_fn)

batch = 16
positions = jax.random.normal(jax.random.PRNGKey(1), (batch, sum(nspins) * 3)) * 0.5
data = FermiNetData(positions, spins_arr, atoms, charges)

# MCMC
mcmc_step = mcmc.make_mcmc_step(log_psi, batch, steps=3, atoms=atoms)
data, pmove = mcmc_step(params, data, jax.random.PRNGKey(2), width=0.02)

value, aux = loss_fn(params, jax.random.PRNGKey(3), data)
```
