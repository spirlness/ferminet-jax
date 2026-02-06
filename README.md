# FermiNet-JAX

基于 JAX 的 FermiNet（费米神经网络）实现，用于求解多电子薛定谔方程的变分蒙特卡洛（VMC）方法。

## 目录

- [简介](#简介)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心概念](#核心概念)
- [配置系统](#配置系统)
- [计算流程](#计算流程)
- [API 参考](#api-参考)
- [示例](#示例)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 简介

FermiNet 是一种深度学习方法，用于从头算（ab initio）求解量子化学中的多电子薛定谔方程。本实现基于 DeepMind 的原始论文：

> D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes,
> "Ab Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks",
> Physical Review Research 2, 033429 (2020).

### 主要特性

- **纯函数式 API**：采用 DeepMind 风格的 `init/apply` 模式
- **JAX 原生**：支持 JIT 编译、自动微分、GPU/TPU 加速
- **KFAC 优化器**：使用 K-FAC 二阶优化加速收敛
- **模块化设计**：网络、损失函数、采样器等组件可独立使用
- **数值稳定**：内置 NaN 恢复机制和自适应 MCMC 采样

### 实测结果

在 NVIDIA RTX 3060 Laptop (6GB) 上的氦原子训练：

| 指标 | 结果 |
|------|------|
| 基态能量 | -2.9032 Ha（精确值 -2.9037 Ha，误差 0.02%）|
| 方差 | 0.004 |
| GPU 显存占用 | 5.8 GB / 6 GB (94%) |
| 每步时间 | ~20s（KFAC，batch=8192）|

---

## 安装

### 环境要求

- Python >= 3.10
- CUDA 12.x（GPU 版本）

### 使用 uv 安装（推荐）

```bash
git clone https://github.com/spirlness/ferminet-jax.git
cd ferminet-jax

uv sync --dev
uv pip install "jax[cuda12]"
```

### 使用 pip 安装

```bash
pip install -e .
pip install "jax[cuda12]>=0.4.30"
```

### 验证安装

```bash
python -c "import jax; print(jax.devices())"
uv run pytest
```

---

## 快速开始

### 命令行训练

```bash
uv run python -m ferminet.main --config src/ferminet/configs/helium.py
```

可用配置：

| 配置文件 | 用途 | GPU 显存需求 |
|---------|------|-------------|
| `helium_quick.py` | 快速测试 | < 2 GB |
| `helium.py` | 标准训练 | ~3 GB |
| `helium_max.py` | 高精度训练 | ~4 GB |
| `helium_scaled.py` | 最大规模训练 | ~6 GB |

### Python API

```python
import jax
import jax.numpy as jnp
from ferminet.networks import make_fermi_net
from ferminet.configs import helium

cfg = helium.get_config()
atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([2.0])
spins = (1, 1)

init_fn, apply_fn, orbitals_fn = make_fermi_net(atoms, charges, spins, cfg)

key = jax.random.PRNGKey(42)
params = init_fn(key)

electrons = jax.random.normal(key, (6,))
sign, log_psi = apply_fn(params, electrons, jnp.array([0, 1]), atoms, charges)
```

### 完整训练

```python
from ferminet import train
from ferminet.configs import helium

cfg = helium.get_config()
cfg.optim.iterations = 10000

result = train.train(cfg)
print(f"Completed steps: {result['step']}")
```

---

## 项目结构

```
ferminet-jax/
├── src/ferminet/
│   ├── networks.py         # FermiNet 网络架构
│   ├── hamiltonian.py      # 哈密顿量和局域能量
│   ├── mcmc.py             # Metropolis-Hastings 采样
│   ├── loss.py             # VMC 损失函数
│   ├── train.py            # 训练循环
│   ├── main.py             # CLI 入口
│   ├── base_config.py      # 默认配置
│   ├── checkpoint.py       # 检查点管理
│   ├── envelopes.py        # 包络函数
│   ├── network_blocks.py   # 网络组件
│   └── configs/            # 配置文件
│       ├── helium.py
│       ├── helium_max.py
│       ├── helium_scaled.py
│       └── hydrogen.py
├── tests/
├── examples/
└── pyproject.toml
```

---

## 核心概念

### 波函数表示

FermiNet 将多电子波函数表示为多个 Slater 行列式的加权和：

```
ψ(r₁, ..., rₙ) = Σₖ det[φᵏ↑] × det[φᵏ↓]
```

其中 `φᵏ` 是神经网络学习的轨道函数，行列式保证了费米子的反对称性。

### 网络架构

```
电子坐标 → 特征提取 → 交互层(×4) → 轨道层 → 行列式 → log|ψ|
                         ↓
              单电子流 + 双电子流
              (残差连接 + LayerNorm)
```

### 变分蒙特卡洛

通过最小化能量期望值优化波函数：

```
E[ψ] = ∫ |ψ(r)|² E_L(r) dr

其中 E_L = Hψ/ψ = -½∇²ψ/ψ + V(r)
```

使用 Metropolis-Hastings 从 |ψ|² 分布采样。

---

## 配置系统

### 关键参数

```python
from ferminet import base_config

cfg = base_config.default()

cfg.batch_size = 4096
cfg.network.determinants = 16
cfg.network.ferminet.hidden_dims = ((256, 32),) * 4

cfg.optim.optimizer = "kfac"
cfg.optim.lr.rate = 0.005
cfg.optim.kfac.damping = 0.01

cfg.mcmc.steps = 10
cfg.mcmc.move_width = 0.5
cfg.mcmc.adapt_frequency = 20
```

### 配置模板

| 场景 | batch_size | determinants | hidden_dims | 显存需求 |
|------|------------|--------------|-------------|---------|
| 快速测试 | 256 | 4 | (64,16)×2 | ~1 GB |
| 标准训练 | 2048 | 8 | (128,32)×4 | ~3 GB |
| 高精度 | 4096 | 16 | (128,32)×4 | ~4 GB |
| 最大规模 | 8192 | 32 | (256,32)×4 | ~6 GB |

---

## 计算流程

```
初始化
├── 解析配置 → 构建原子系统
├── 初始化网络参数
├── 初始化电子位置（原子周围高斯分布）
└── 初始化优化器（KFAC）

训练循环
├── MCMC 采样（10步）
│   └── 提议 → 计算接受率 → 接受/拒绝
├── 计算局域能量 E_L
├── 计算梯度（custom_jvp 无偏估计）
├── KFAC 参数更新
└── 自适应调整采样宽度

输出
├── 定期打印：Step, Energy, Variance, pmove, width
└── 定期保存检查点
```

---

## API 参考

### make_fermi_net

```python
def make_fermi_net(
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    spins: tuple[int, int],
    cfg: ConfigDict,
) -> tuple[InitFn, ApplyFn, OrbitalsFn]:
```

返回 `(init_fn, apply_fn, orbitals_fn)` 三元组。

### make_loss

```python
def make_loss(
    network: Callable,
    local_energy_fn: Callable,
    clip_local_energy: float = 5.0,
) -> LossFn:
```

创建带无偏梯度估计的 VMC 损失函数。

### make_mcmc_step

```python
def make_mcmc_step(
    network: Callable,
    batch_per_device: int,
    steps: int = 10,
) -> MCMCStepFn:
```

创建 MCMC 采样步骤函数。

---

## 示例

### 氦原子

```bash
uv run python -m ferminet.main --config src/ferminet/configs/helium.py
```

预期：能量收敛到 -2.9037 Ha（~10,000 步）

### 氢分子

```bash
uv run python -m ferminet.main --config src/ferminet/configs/hydrogen.py
```

预期：能量收敛到 -1.174 Ha（平衡键长）

### 自定义系统

```python
cfg.system.molecule = [
    ("Li", (0.0, 0.0, 0.0)),
]
cfg.system.electrons = (2, 1)
cfg.system.charges = (3,)
```

---

## 性能优化

### GPU 配置

```bash
uv pip install "jax[cuda12]"
python -c "import jax; print(jax.devices())"
```

### 显存优化

如果遇到 OOM，按优先级调整：

1. 减小 `batch_size`（影响最大）
2. 减少 `determinants`
3. 减小 `hidden_dims`

### 训练稳定性

本实现包含以下稳定性机制：

- **自适应 MCMC 宽度**：自动调整采样步长使接受率维持在 0.5-0.6
- **非有限值保护**：拒绝产生 NaN/Inf 的 MCMC 提议
- **NaN 恢复**：检测到 NaN 能量时跳过参数更新

推荐的稳定参数：

```python
cfg.optim.lr.rate = 0.005
cfg.optim.kfac.damping = 0.01
cfg.optim.clip_local_energy = 3.0
cfg.mcmc.move_width = 0.5
```

---

## 常见问题

### 训练出现 NaN

1. 降低学习率：`cfg.optim.lr.rate = 0.002`
2. 增加 KFAC 阻尼：`cfg.optim.kfac.damping = 0.05`
3. 增加局域能量裁剪：`cfg.optim.clip_local_energy = 3.0`

### pmove 过低或过高

pmove（MCMC 接受率）应在 0.5-0.6 之间。系统会自动调整 `move_width`，但如果持续异常：

- pmove < 0.3：减小初始 `move_width`
- pmove > 0.8：增大初始 `move_width`

### KFAC 编译时间长

首次运行时 JIT 编译可能需要 10-20 分钟，这是正常现象。后续步骤会快很多。

### 显存不足

参见[显存优化](#显存优化)部分。

---

## 引用

```bibtex
@article{pfau2020ferminet,
  title={Ab initio solution of the many-electron Schr{\"o}dinger equation with deep neural networks},
  author={Pfau, David and Spencer, James S and Matthews, Alexander GDG and Foulkes, W Matthew C},
  journal={Physical Review Research},
  volume={2},
  number={3},
  pages={033429},
  year={2020},
  publisher={APS}
}
```

## 许可证

Apache License 2.0
