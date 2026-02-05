# 运行方式

## 1) 跑单元测试

```bash
uv run pytest
```

## 2) 运行氦原子端到端示例

```bash
uv run python examples/test_helium.py
```

## 3) 使用 CLI 跑训练（absl + ml_collections config）

示例：

```bash
uv run python -m ferminet.main --config ferminet/configs/helium.py
```

说明：配置文件为 Python 模块，提供 `get_config()` 返回 `ml_collections.ConfigDict`。
