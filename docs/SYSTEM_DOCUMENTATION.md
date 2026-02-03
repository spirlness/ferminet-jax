# FermiNet - 基于JAX的量子蒙特卡洛变分波函数实现

## 目录

1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [核心模块详解](#核心模块详解)
   - [3.1 网络层](#31-网络层)
   - [3.2 物理计算层](#32-物理计算层)
   - [3.3 训练器层](#33-训练器层)
   - [3.4 MCMC采样层](#34-mcmc采样层)
4. [性能优化](#性能优化)
5. [使用指南](#使用指南)
6. [API参考](#api参考)
7. [常见问题](#常见问题)

---

## 项目概述

FermiNet是一个使用JAX实现的变分蒙特卡洛(VMC)框架，用于计算分子电子结构的基态能量和性质。

### 核心特性

- **纯JAX实现**：完全基于JAX进行自动微分和JIT编译
- **多行列式支持**：支持1-16个Slater行列式
- **Jastrow因子**：包含Jastrow电子相关因子
- **高效MCMC**：Metropolis-Hastings采样算法
- **Adam优化器**：自适应矩估计优化
- **批量化计算**：支持批样本并行处理

### 技术栈

- Python 3.8+
- JAX 0.8.3+
- NumPy/JAX NumPy
- 可选：GPU加速（CUDA支持）

### 项目结构

```
FermiNet/
├── src/ferminet/          # 核心源代码
│   ├── network.py        # 网络架构
│   ├── physics.py        # 物理计算
│   ├── trainer.py        # 训练器
│   ├── mcmc.py           # MCMC采样
│   ├── multi_determinant.py # 多行列式
│   ├── jastrow.py        # Jastrow因子
│   └── __init__.py      # 包初始化
├── tests/                 # 测试套件
├── examples/              # 示例脚本
├── configs/               # 配置文件
├── docs/                  # 文档
└── README.md
```

---

## 架构设计

### 整体架构

FermiNet采用分层架构，各层职责清晰：

```
┌─────────────────────────────────────────────────────────────┐
│                   应用层                            │
│  (训练脚本、基准测试、示例)                    │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│                  训练器层                         │
│  (VMCTrainer, ExtendedTrainer)                  │
│  - Adam优化器管理                                   │
│  - 学习率调度器                                     │
│  - 梯度裁剪                                         │
└────────────────┬────────────────────────────────────────────┘
                 │
�────────┬─────────┴─────────────────────────────────────
        │         │
┌───────▼──────┐  │  ┌────────▼────────────────────────────────────┐
│  网络层     │  │  │  MCMC采样层                            │
│ (Extended    │  │  │ (FixedStepMCMC)                        │
│  FermiNet)   │  │  │ - Metropolis接受/拒绝                 │
│              │  │  │ - 梯度计算                               │
└──────┬───────┘  │  └────────┬────────────────────────────────────┘
       │           │           │
┌───────▼─────────▼──────┐  │  ┌──────▼────────────────────────────────────┐
│   物理计算层            │  │  │   网络组件层                          │
│  (kinetic_energy,     │  │  │ (MultiDeterminantOrbitals,            │
│   local_energy)           │  │  │  JastrowFactor)                    │
└────────────────────────────┘  │  └────────────────────────────────────────────┘
                                │
                                ▼
                        ┌────────────────────────────────────────────────────┐
                        │  JAX核心层                          │
                        │  (jax.jit, jax.grad, jax.vmap)          │
                        │  - 自动微分                         │
                        │  - JIT编译                            │
                        │  - XLA优化                           │
                        └────────────────────────────────────────────────────┘
```

### 数据流

```
1. 初始化阶段
   ↓
   核据位置和电荷 → ExtendedFermiNet初始化
   ↓
   MCMC采样器初始化
   ↓
   训练器初始化(网络、采样器、配置)

2. 训练循环
   ↓
   MCMC采样(当前电子位置，log_psi函数)
   → 新电子位置 + 接受率
   ↓
   能量计算(新电子位置)
   → 局部能量 + 梯度
   ↓
   参数更新(Adam优化器)
   → 新网络参数
   ↓
   (重复直到收敛)
```

---

## 核心模块详解

### 3.1 网络层

#### ExtendedFermiNet

变分波函数网络的核心实现，支持多行列式和Jastrow因子。

**主要参数：**

```python
ExtendedFermiNet(
    n_electrons: int,           # 电子总数
    n_up: int,                  # 自旋向上电子数
    nuclei_config: Dict,         # 原子配置
    network_config: Dict           # 网络超参数
)
```

**网络架构：**

```
输入：电子位置 [batch, n_elec, 3]
  ↓
单粒子特征(One-Body) → [batch, n_elec, n_nuclei]
  ↓
双粒子特征(Two-Body) → [batch, n_elec, n_elec, pair_width]
  ↓
交互层(Interaction Layers) × num_layers
  ↓
轨道线性组合
  ↓
行列式计算(Slater Determinants) × n_determinants
  ↓
Log-Sum-Exp 组合
  ↓
输出：log|ψ(r)| [batch]
```

**关键方法：**

- `__call__(r_elec)`：前向传播，返回对数波函数值
- `apply(params, r_elec)`：函数式调用，用于JIT编译
- `multi_determinant_slater()`：多行列式Log-Sum-Exp组合

#### MultiDeterminantOrbitals

多行列式轨道计算模块。

**核心逻辑：**

```python
for each determinant:
    计算轨道值 → [batch, n_elec, n_elec]
    计算行列式 → slogdet → (sign, log|det|)
    
组合权重 → softmax归一化
Log-Sum-Exp: log|Σ(w_i * det_i)| 
         = max_log + log(Σ exp(log_w_i + log_det_i - max_log))
```

#### JastrowFactor

Jastrow电子相关因子，用于满足泡利不相容原理。

**公式：**

```
J(r) = exp(-0.5 × Σ|c_ij - r_i| / β)
```

其中 β 是可调参数。

---

### 3.2 物理计算层

#### kinetic_energy

计算电子的动能能量。

**公式：**

```
T = -0.5 × (|∇log ψ|² + ∇²log ψ)
```

使用JAX自动微分实现：

```python
def kinetic_energy(log_psi: Callable, r_r: jnp.ndarray) -> jnp.ndarray:
    grad = jax.grad(log_psi)(r_r)
    grad_squared_sum = jnp.sum(grad ** 2)
    
    hessian = jax.hessian(log_psi)(r_r)
    laplacian = jnp.trace(hessian)
    
    return -0.5 * (grad_squared_sum + laplacian)
```

#### total_potential

计算总势能（核-电子 + 电子-电子排斥）。

**核-电子吸引势：**

```
V_ne = -Σ_i Σ_j Z_j / |r_i - R_j|
```

**电子-电子排斥势：**

```
V_ee = Σ_{i<j} 1 / |r_i - r_j|
```

使用软化库仑势避免奇点：

```python
V_soft = 1 / sqrt(r² + α²)
```

#### local_energy

局部哈密顿顿能量。

```
E_L = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩
    = T + V_ne + V_ee
```

#### make_batched_local_energy

批量化能量计算工厂函数。

**优化策略：**

1. 预编译 AD 变换（grad, hessian）
2. 使用 `jax.vmap` 批量化
3. 避免 Python 循环

```python
def make_batched_local_energy(log_psi: Callable, n_electrons: int) -> Callable:
    # 预构建 AD 变换
    grad_log_psi_flat = jax.grad(log_psi_flat)
    hess_log_psi_flat = jax.jacfwd(grad_log_psi_flat)
    
    # 定义单样本能量计算
    def local_energy_single(params, r_single, ...):
        t = kinetic_single(params, r_single)
        v = total_potential(r_single, ...)
        return t + v
    
    # 返回批量化函数
    return jax.vmap(local_energy_single, in_axes=(None, 0, None, None))
```

**性能对比：**

| 方法 | 时间 (128样本) | 加速比 |
|------|--------------|--------|
| Python 循环 | ~60s | 1x |
| vmap + JIT | ~6s | 10x |

---

### 3.3 训练器层

#### VMCTrainer

基础变分蒙特卡洛训练器。

**核心组件：**

```python
class VMCTrainer:
    def __init__(self, network, mcmc, config):
        self.network = network
        self.mcmc = mcmc
        self.learning_rate = config["learning_rate"]
        
        # Adam 优化器状态
        self.adam_state = self._init_adam_state(network.params)
        
        # 预编译的更新函数
        self._jit_update = jax.jit(self._update_step)
```

**训练步骤：**

```python
def train_step(self, params, r_elec, key, nuclei_pos, nuclei_charge):
    # 1. MCMC 采样
    def log_psi_fn(r):
        return self.network.apply(params, r)
    r_elec_new, accept_rate = self.mcmc.sample(log_psi_fn, r_elec, key)
    
    # 2. 能量计算和梯度
    params_new, self.adam_state, mean_E, _ = self._jit_update(
        params, r_elec_new, nuclei_pos, nuclei_charge, 
        self.adam_state, self.learning_rate
    )
    
    return params_new, mean_E, accept_rate, r_elec_new
```

#### Adam优化器

自适应矩估计优化器。

**更新规则：**

```python
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²

m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - α × m̂_t / (√(v̂_t) + ε)
```

**默认超参数：**

- β₁ = 0.9 (一阶矩衰减)
- β₂ = 0.999 (二阶矩衰减)
- ε = 1e-8 (数值稳定性)

#### EnergyBasedScheduler

基于能量的学习率调度器。

**逻辑：**

```python
if 当前能量 < 最佳能量:
    重置等待计数器
elif 等待计数器 > 耐心值:
    学习率 ×= 衰减因子
```

#### ExtendedTrainer

扩展训练器，添加梯度裁剪和学习率调度。

**梯度裁剪：**

```python
def _clip_gradients(self, grads, max_norm=1.0, norm_type="inf"):
    if norm_type == "inf":
        grad_norm = jnp.max(jnp.abs(grads))
    elif norm_type == "l2":
        grad_norm = jnp.linalg.norm(grads, ord=2)
    
    scale = jnp.where(grad_norm > max_norm, max_norm / grad_norm, 1.0)
    return grads * scale
```

---

### 3.4 MCMC采样层

#### FixedStepMCMC

Metropolis-Hastings MCMC采样器。

**算法流程：**

```
对于每个样本：
    对于每个采样步骤:
        1. 提议新位置：r' = r + δ，其中 δ ~ N(0, step_size)
        2. 计算波函数比：α = |ψ(r')|² / |ψ(r)|²
        3. Metropolis 接受：
           if α ≥ 1 or rand() < α:
              接受：r = r'，接受计数 += 1
           否则：
              拒绝：保持 r
```

**关键实现：**

```python
def sample(self, log_psi_fn, r_elec, key):
    key, noise_key = random.split(key)
    noise = random.normal(noise_key, r_elec.shape) * self.step_size
    r_proposed = r_elec + noise
    
    log_psi_curr = log_psi_fn(r_elec)
    log_psi_proposed = log_psi_fn(r_proposed)
    
    ratio = jnp.exp(2.0 * (log_psi_proposed - log_psi_curr))
    
    key, uniform_key = random.split(key)
    accept = random.uniform(uniform_key, ratio.shape) < ratio
    accept = jnp.where(ratio >= 1.0, True, accept)
    
    r_new = jnp.where(accept, r_proposed, r_elec)
    accept_rate = jnp.mean(accept)
    
    return r_new, accept_rate
```

**性能优化：**

- 使用 `jax.vmap` 批量化梯度计算
- 预编译 Metropolis 判断函数
- 避免 Python 循环中的 JIT 重编译

---

## 性能优化

### JIT 编译策略

#### 关键 JIT 边界

```python
# 1. 物理计算 - 完全 JIT
@jax.jit
def kinetic_energy_optimized(log_psi, r_single):
    hessian = jax.hessian(log_psi)(r_single)
    grad = jax.grad(log_psi)(r_single)
    return -0.5 * (jnp.sum(grad**2) + jnp.trace(hessian))

# 2. 批量化 - vmap
batched_kinetic = jax.vmap(kinetic_energy_optimized, in_axes=(None, 0))

# 3. 训练更新 - JIT + 预热
@jax.jit
def update_step(params, r_elec, ...):
    loss, grads = jax.value_and_grad(energy_loss)(params, r_elec, ...)
    return adam_update(params, grads, ...)

def warmup(params, r_elec, ...):
    update_step(params, r_elec, ...)  # 预热
    jax.block_until_ready(...)
```

### 性能基准

**测试配置：** RTX 3060 6G

| 组件 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| 能量+梯度 | 5.804 ms | 1.15 ms | **5.05x** |
| 完整训练步 | 383.46 ms | 300.28 ms | **1.28x** |

### 优化检查清单

- [x] 使用 `jax.jit` 装饰所有计算密集型函数
- [x] 使用 `jax.vmap` 批量化样本处理
- [x] 避免 Python 循环中的 JAX 操作
- [x] 预编译 AD 变换（grad, hessian）
- [x] 使用 `jax.block_until_ready` 准确异步操作完成
- [x] 实现预热函数分离首次编译开销

### 内存优化

**策略：**

1. **批处理**：增加批次大小以提高 GPU 利用率
2. **梯度累积**：如果内存不足，使用小批次 + 梯度累积
3. **混合精度**：使用 float32 而非 float64（需要权衡精度）

---

## 使用指南

### 基本用法

```python
import jax
import jax.numpy as jnp
from ferminet.network import ExtendedFermiNet
from ferminet.trainer import VMCTrainer
from ferminet.mcmc import FixedStepMCMC

# 1. 配置系统
config = {
    "n_electrons": 2,
    "n_up": 1,
    "nuclei_config": {
        "positions": jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        "charges": jnp.array([1.0, 1.0])
    },
    "network": {
        "single_layer_width": 32,
        "pair_layer_width": 8,
        "num_interaction_layers": 1,
        "determinant_count": 1,
    },
    "mcmc": {
        "n_samples": 256,
        "step_size": 0.15,
        "n_steps": 5,
    },
    "learning_rate": 0.001,
}

# 2. 初始化组件
network = ExtendedFermiNet(
    config["n_electrons"], 
    config["n_up"], 
    config["nuclei_config"], 
    config["network"]
)

mcmc = FixedStepMCMC(
    step_size=config["mcmc"]["step_size"],
    n_steps=config["mcmc"]["n_steps"]
)

trainer = VMCTrainer(network, mcmc, config)

# 3. 初始化电子位置
key = jax.random.PRNGKey(42)
key, init_key = jax.random.split(key)
r_elec = jax.random.normal(init_key, (config["mcmc"]["n_samples"], 
                                           config["n_electrons"], 3))

# 4. 预热
trainer.warmup(
    trainer.network.params, 
    r_elec, 
    config["nuclei_config"]["positions"],
    config["nuclei_config"]["charges"]
)

# 5. 训练循环
params = trainer.network.params
nuclei_pos = config["nuclei_config"]["positions"]
nuclei_charge = config["nuclei_config"]["charges"]

for epoch in range(100):
    key, step_key = jax.random.split(key)
    
    params, energy, accept_rate, r_elec = trainer.train_step(
        params, r_elec, step_key, nuclei_pos, nuclei_charge
    )
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: E={energy:.6f}, Accept={accept_rate:.3f}")
```

### 使用扩展训练器

```python
from ferminet.trainer import ExtendedTrainer

config_extended = {
    **config,
    "use_scheduler": True,
    "target_energy": -1.174,  # H2 基态能量
    "gradient_clip": 1.0,
    "gradient_clip_norm": "inf",
    "scheduler_patience": 10,
    "decay_factor": 0.5,
    "min_lr": 1e-5,
}

trainer_ext = ExtendedTrainer(network, mcmc, config_extended)

for epoch in range(100):
    params, energy, accept_rate, r_elec, info = trainer_ext.train_step(
        params, r_elec, key, nuclei_pos, nuclei_charge
    )
    
    # 更新学习率
    if trainer_ext.use_scheduler:
        new_lr, decayed, _ = trainer_ext.update_scheduler(energy)
```

### 自定义网络

```python
from ferminet.network import ExtendedFermiNet
from ferminet.multi_determinant import MultiDeterminantOrbitals
from ferminet.jastrow import JastrowFactor

class CustomFermiNet(ExtendedFermiNet):
    def __init__(self, n_electrons, n_up, nuclei_config, network_config):
        super().__init__(n_electrons, n_up, nuclei_config, network_config)
        
        # 自定义配置
        self.custom_param = network_config.get("custom_param", 1.0)
        
        # 可以添加额外的层或修改现有层
        
    def __call__(self, r_elec):
        # 先调用父类方法
        base_output = super().__call__(r_elec)
        
        # 添加自定义变换
        modified_output = base_output + self.custom modification
        
        return modified_output
```

---

## API参考

### network.py

#### ExtendedFermiNet

**初始化：**

```python
ExtendedFermiNet(n_electrons, n_up, nuclei_config, network_config)
```

**参数：**

- `n_electrons`: int - 电子总数
- `n_up`: int - 自旋向上电子数
- `nuclei_config`: dict - 原子配置
  - `positions`: jnp.ndarray - [n_nuclei, 3]
  - `charges`: jnp.ndarray - [n_nuclei]
- `network_config`: dict - 网络超参数
  - `single_layer_width`: int - 单粒子层宽度（默认 32）
  - `pair_layer_width`: int - 双粒子层宽度（默认 8）
  - `num_interaction_layers`: int - 交互层数（默认 1）
  - `determinant_count`: int - 行列式数量（默认 1，最大 16）
  - `use_jastrow`: bool - 是否使用 Jastrow（默认 False）
  - `jastrow_hidden_dim`: int - Jastrow 隐藏层维度
  - `jastrow_layers`: int - Jastrow 层数

**方法：**

- `__call__(r_elec)` → jnp.ndarray
  - 前向传播
  - 输入：r_elec [batch, n_elec, 3]
  - 输出：log|ψ| [batch]

- `apply(params, r_elec)` → jnp.ndarray
  - 函数式调用
  - 用于 JIT 编译

### physics.py

#### make_batched_local_energy

**工厂函数：**

```python
make_batched_local_energy(log_psi, n_electrons) -> Callable
```

**返回的函数：**

```python
batched_energy(params, r_batch, nuclei_pos, nuclei_charge) -> jnp.ndarray
```

- 输入：
  - `params`: dict - 网络参数
  - `r_batch`: jnp.ndarray - [batch, n_elec, 3]
  - `nuclei_pos`: jnp.ndarray - [n_nuclei, 3]
  - `nuclei_charge`: jnp.ndarray - [n_nuclei]
- 输出：local_E [batch]

### trainer.py

#### VMCTrainer

**初始化：**

```python
VMCTrainer(network, mcmc, config)
```

**配置参数：**

- `learning_rate`: float - 学习率（默认 0.001）
- `beta1`: float - Adam β1（默认 0.9）
- `beta2`: float - Adam β2（默认 0.999）
- `epsilon`: float - Adam ε（默认 1e-8）

**方法：**

- `train_step(params, r_elec, key, nuclei_pos, nuclei_charge)` → tuple
  - 执行单步训练
  - 返回：(params_new, mean_E, accept_rate, r_elec_new)

- `warmup(params, r_elec, nuclei_pos, nuclei_charge)` → None
  - 预热 JIT 编译
  - 应在训练循环外调用一次

- `energy_loss(params, r_elec, nuclei_pos, nuclei_charge)` → tuple
  - 返回：(loss, mean_E)

#### ExtendedTrainer

**额外配置参数：**

- `use_scheduler`: bool - 使用学习率调度器
- `target_energy`: float - 目标基态能量
- `gradient_clip`: float - 梯度裁剪阈值
- `gradient_clip_norm`: str - 裁剪范数类型（"inf", "l2", "l1"）
- `scheduler_patience`: int - 调度器耐心值
- `decay_factor`: float - 学习率衰减因子
- `min_lr`: float - 最小学习率

**方法：**

- `update_scheduler(current_energy)` → tuple
  - 更新学习率
  - 返回：(new_lr, decayed, old_lr)

- `get_training_info()` → dict
  - 获取训练信息

### mcmc.py

#### FixedStepMCMC

**初始化：**

```python
FixedStepMCMC(step_size, n_steps)
```

**参数：**

- `step_size`: float - MCMC 步长
- `n_steps`: int - 每次采样步骤数

**方法：**

- `sample(log_psi_fn, r_elec, key)` → tuple
  - 执行 MCMC 采样
  - 返回：(r_elec_new, accept_rate)

---

## 常见问题

### Q1: 训练速度很慢，如何优化？

**A:** 检查以下几点：

1. 确保使用 JIT 编译
   ```python
   @jax.jit
   def my_function(...):
       ...
   ```

2. 使用批量化
   ```python
   batched_fn = jax.vmap(single_fn, in_axes=(None, 0))
   ```

3. 调用预热函数
   ```python
   trainer.warmup(params, r_elec, nuclei_pos, nuclei_charge)
   ```

4. 增加批次大小
   ```python
   n_samples = 512  # 从 256 增加到 512
   ```

### Q2: 遇到 "recompilation warning"？

**A:** 这表示 JIT 重编译。解决方法：

1. 确保 JIT 函数签名固定
2. 避免在 JIT 函数内部使用控制流依赖数据
3. 使用 `jax.lax.cond` 替代 Python if/else

### Q3: 能量不收敛？

**A:** 尝试以下调整：

1. 降低学习率
2. 增加 MCMC 步长
3. 增加行列式数量
4. 启用 Jastrow 因子
5. 使用学习率调度器

### Q4: CUDA out of memory？

**A:** 解决方法：

1. 减小批次大小
2. 启用梯度累积
3. 使用混合精度训练（float32）
4. 关闭不必要的调试日志

### Q5: 如何可视化训练？

**A:** 参考 `examples/` 中的示例脚本。

```python
import matplotlib.pyplot as plt

energies = []
for epoch in range(num_epochs):
    # 训练...
    energies.append(float(energy))

plt.plot(energies)
plt.xlabel('Epoch')
plt.ylabel('Energy (Hartree)')
plt.title('Training Convergence')
plt.show()
```

### Q6: 如何保存/加载模型？

**A:** JAX 参数是纯 Python 字典，使用 pickle 保存。

```python
import pickle

# 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(network.params, f)

# 加载
with open('model.pkl', 'rb') as f:
    loaded_params = pickle.load(f)
    network.params = loaded_params
```

---

## 许可证

FermiNet 采用 Apache 2.0 许可证。

---

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 编写代码并添加测试
4. 运行测试套件
5. 提交 Pull Request

---

## 联系

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/ferminet/issues)
- 文档: [docs/](docs/)
- 示例: [examples/](examples/)
