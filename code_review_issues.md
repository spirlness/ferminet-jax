# 代码审查问题清单

以下问题来自对当前仓库的静态代码审查，建议以 issue 形式跟踪与修复。

## Issue 1: 训练损失未使用传入参数，导致梯度计算与优化失效

**位置**
- `src/ferminet/trainer.py` 中 `VMCTrainer.__init__` 创建的 `network_forward`。 

**问题描述**
- `make_batched_local_energy` 期望的 `log_psi` 形态是 `(params, r_batch) -> log|psi|`，但当前 `network_forward` 忽略了 `params`，直接调用 `self.network(r_batch)`。这会导致优化步骤使用的 `params` 与实际前向计算脱钩，进而使梯度无法正确反映当前参数更新。
- 结果是：优化器更新的 `params` 并不会影响能量估计，训练可能停滞或产生不正确梯度。

**建议修复**
- 将 `network_forward` 改为 `self.network.apply(params, r_batch)`，确保损失与梯度真正依赖传入参数。

**影响范围**
- 训练稳定性与优化有效性；可能导致训练无法收敛。

---

## Issue 2: 多行列式权重组合逻辑不一致

**位置**
- `src/ferminet/multi_determinant.py` 中 `combine_with_weights` 使用 softmax 规范化权重。
- `src/ferminet/network.py` 中 `ExtendedFermiNet.multi_determinant_slater` 直接使用原始 `det_weights` 并引入符号/绝对值对数处理。

**问题描述**
- 同一套参数 `det_weights` 在两个模块中使用方式不同：一个进行 softmax 归一化，另一个保留原始符号并进行 log-abs 组合。两种实现产生的波函数幅度并不一致，导致不同路径下得到的 log|psi| 可能相互矛盾。
- 这会带来训练行为不稳定或难以复现的结果，尤其在权重为负时表现更明显。

**建议修复**
- 统一多行列式权重组合策略：要么全程采用 softmax 权重并保持无符号组合，要么明确支持带符号的组合并在所有路径中一致化实现。

**影响范围**
- 波函数定义一致性与数值稳定性；可能导致训练结果不可重复或发散。

---

## Issue 3: Soft-Coulomb 势能实现存在倒数错误，导致势能数值反向

**位置**
- `src/ferminet/physics.py` 中 `nuclear_potential` 与 `electronic_potential` 的 soft-core 距离使用方式。

**问题描述**
- `soft_coulomb_potential_sq` 已经返回了 `1 / sqrt(r^2 + alpha^2)`，但在 `nuclear_potential` 中又使用 `nuclei_charge / soft_distances`，在 `electronic_potential` 中使用 `1.0 / soft_distances`，等价于将势能变成 `sqrt(r^2 + alpha^2)`，与物理期望的软化库仑势完全相反。
- 结果会导致势能量纲与大小错误，破坏能量估计与训练收敛。

**建议修复**
- 直接使用 `soft_distances` 作为分母的倒数结果，即：`potential = -jnp.sum(nuclei_charge[None, :] * soft_distances)`，以及 `potential = jnp.sum(soft_distances * mask)`。

**影响范围**
- 全部能量计算与优化结果；势能符号与数值可能严重偏离正确结果。

---

## Issue 4: Langevin MCMC 采样未包含 MALA 接受率修正项

**位置**
- `src/ferminet/mcmc.py` 中 `_langevin_step` 与 `_metropolis_accept`。

**问题描述**
- 代码使用了带漂移项的 Langevin 提议 (`r' = r + 0.5 * ∇logψ * dt + N(0, dt)`)，但接受率仅基于 `exp(2*(logψ' - logψ))`，没有包含 MALA 的正向/反向提议密度比。
- 这会导致采样分布偏离目标分布，尤其是步长较大或梯度变化明显时。

**建议修复**
- 使用 MALA 接受率：`accept_prob = exp(2*(logψ' - logψ) + log_q(r|r') - log_q(r'|r))`，其中 `log_q` 为高斯提议密度。
- 或者改为纯随机游走提议以保持对称性。

**影响范围**
- 采样分布正确性与接受率；可能导致估计偏差与训练不稳定。
