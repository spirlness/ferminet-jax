# uv 环境与 JAX 安装指南

本项目使用 `uv` 管理依赖，并通过不同的依赖组安装适配 CPU / GPU / TPU 的 JAX 版本。

## 1) 创建虚拟环境

```bash
uv venv
source .venv/bin/activate
```

## 2) 按设备安装 JAX

### CPU

```bash
uv pip install -e ".[cpu,dev]"
```

### GPU (CUDA 12)

```bash
uv pip install -e ".[gpu,dev]" \
  --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### TPU

```bash
uv pip install -e ".[tpu,dev]" \
  --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

也可以使用脚本快速安装：

```bash
./scripts/install_jax.sh cpu
./scripts/install_jax.sh gpu
./scripts/install_jax.sh tpu
```

> 如果你的 GPU 使用的是 CUDA 11，请将 `jax[cuda12]` 改为 `jax[cuda11]` 并调整对应的安装命令。

## 3) 运行测试

```bash
python tests/test_stage2.py
python tests/test_network_stability.py
python tests/test_energy_quick.py
python tests/test_extended_debug.py
```
