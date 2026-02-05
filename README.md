# ferminet-jax

JAX 版 FermiNet（变分蒙特卡洛 / VMC）实现，主代码位于 `ferminet/`，采用 DeepMind 风格的函数式 `init/apply` API。

## 快速开始

```bash
uv sync --dev
uv run pytest
uv run python examples/test_helium.py
```

## 训练入口

- 直接调用训练函数：`ferminet/train.py:train`
- CLI（absl + ml_collections config）：

```bash
uv run python -m ferminet.main --config ferminet/configs/helium.py
```

## 代码结构

- `src/ferminet/`：核心实现（network / mcmc / hamiltonian / loss / train 等）
- `src/ferminet/configs/`：示例配置（`get_config()` 返回 `ml_collections.ConfigDict`）
- `tests/`：基于新 API 的回归测试（pytest）
- `examples/test_helium.py`：氦原子端到端小例子

## 文档

详细代码文档见：`docs/README.md`
