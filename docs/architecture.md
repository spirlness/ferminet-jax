# 架构概览

## 设计目标

- 以 JAX 函数式范式实现 FermiNet/VMC：网络与训练逻辑尽量保持纯函数，便于 `jit`/`vmap`/`pmap`。
- 核心 API 采用 DeepMind 风格的 `(init, apply, orbitals)` 工厂函数。

## 顶层目录

- `src/ferminet/`：主要库代码
- `src/ferminet/configs/`：示例配置（`get_config()` 返回 `ml_collections.ConfigDict`）
- `tests/`：基于新 API 的回归测试
- `examples/test_helium.py`：端到端最小训练例子

## `ferminet/` 模块职责

- `ferminet/types.py`
  - `FermiNetData`: 训练/采样的数据容器（positions/spins/atoms/charges）
  - `ParamTree`: 参数树类型别名（JAX pytree）

- `ferminet/base_config.py`
  - `default()`: 生成基础 `ConfigDict`
  - `resolve()`: 解析 `FieldReference`（注意：使用 `cfg.get_ref(...)`/`cfg.network.get_ref(...)` 创建引用）

- `ferminet/networks.py`
  - `make_fermi_net(atoms, charges, nspins, cfg) -> (init, apply, orbitals)`
  - `make_log_psi_apply(apply_fn)`: 将 `(sign, log|psi|)` 的 `apply` 包装为只返回 `log|psi|`
  - 说明：网络内部包含 one-electron / two-electron 两路特征与交互层；使用 `slogdet` 组合自旋通道行列式。

- `ferminet/hamiltonian.py`
  - `local_energy(f, charges, nspins, ...) -> LocalEnergy`: 产生局部能量函数 `E_L = T + V`
  - `local_kinetic_energy(...)`: 基于 `log|psi|` 的梯度与拉普拉斯计算动能项

- `ferminet/mcmc.py`
  - `mh_update(...)`: 单步 Metropolis-Hastings 更新
  - `make_mcmc_step(log_psi_batched, batch_per_device, steps, atoms, ...)`: 多步采样器封装，返回 `(new_data, pmove)`

- `ferminet/loss.py`
  - `make_loss(log_psi, local_energy_fn, clip_local_energy=...)`:
    - 返回 `(loss, aux)`
    - 使用 `custom_jvp` 实现无偏 VMC 梯度（covariance 形式）

- `ferminet/train.py`
  - `train(cfg)`: 训练入口（KFAC/Adam），负责组装 network/local_energy/loss/mcmc 与 checkpoint/logging

## 训练数据流（简化）

1. `configs/*.py:get_config()` 生成 `cfg`（再由 `base_config.resolve(cfg)` 解析引用）
2. `networks.make_fermi_net(...)` 生成 `init/apply/orbitals`
3. `networks.make_log_psi_apply(apply)` 得到 `log_psi(params, positions, spins, atoms, charges)`
4. `hamiltonian.local_energy(apply_sign_log, charges, nspins)` 生成单样本局部能量
5. 对 batch：用 `jax.vmap` 把单样本局部能量提升为 `local_energy_fn(params, key, data) -> (batch,)`
6. `loss.make_loss(log_psi, local_energy_fn)` 生成 VMC loss（`custom_jvp` 无偏梯度）
7. `mcmc.make_mcmc_step(log_psi_batched, batch_per_device, steps, atoms)` 更新采样点
8. `train.train(cfg)` 把上述组件串起来跑训练、日志与 checkpoint

## 数据形状约定（关键）

- `positions`: `(batch, n_electrons * ndim)`（扁平化）
- `spins`: `(n_electrons,)`（示例中用 0/1 标识上下自旋）
- `atoms`: `(n_atoms, ndim)`
- `charges`: `(n_atoms,)`

## 数值稳定性约定

- electron-electron 自距离（对角线）会导致 `||r||` 在 0 处梯度不稳定；网络内部对范数使用了 epsilon-stabilized 形式以避免 NaN 梯度。

## 备注

当前文档以 `ferminet/` 为主实现。
