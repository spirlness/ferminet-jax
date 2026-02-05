# 开发说明

## 代码位置

- 主库：`ferminet/`
- 测试：`tests/`
- 示例：`examples/`

## 常用命令

```bash
uv sync --dev
uv run pytest
uv run python examples/test_helium.py
```

## 代码风格/约定

- 尽量保持 JAX 函数式：可被 `jit`/`vmap` 的函数避免副作用。
- 使用 `ParamTree`（pytree）保存参数。
- 形状约定：电子坐标通常使用扁平化 `(batch, n_electrons * ndim)`。

## 类型检查

仓库包含 `pyrightconfig.json`；如需手动检查（可选）：

```bash
uv run pyright
```
