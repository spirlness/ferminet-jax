# uv 开发环境

本仓库使用 `uv` 管理依赖与虚拟环境。

## 安装依赖

```bash
uv sync --dev
```

## 运行测试

```bash
uv run pytest
```

## 运行示例

```bash
uv run python examples/test_helium.py
```

## 检查加速器

```bash
uv run python -c "import jax; print(jax.devices())"
```
