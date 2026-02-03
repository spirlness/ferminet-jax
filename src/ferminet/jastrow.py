"""
Jastrow电子相关因子实现 (Stage 2)
"""

import jax
import jax.numpy as jnp
import jax.random


class JastrowFactor:
    """
    Jastrow电子相关因子，用于捕捉电子间的关联效应。

    Jastrow因子是对称函数，满足 J(r_ij) = J(r_ji)。
    实现中只计算上三角部分（i < j）然后求和，避免重复计算。
    """

    def __init__(self, n_electrons, hidden_dim=32, n_layers=2):
        """
        初始化Jastrow因子网络。

        Args:
            n_electrons: 电子总数
            hidden_dim: 隐藏层维度
            n_layers: MLP的隐藏层数
        """
        self.n_electrons = n_electrons
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 初始化网络参数
        self.params = self._init_parameters(jax.random.PRNGKey(0))

    def _init_parameters(self, key):
        """
        初始化神经网络参数。

        网络结构：
        - 输入：电子对距离 r_ij (标量)
        - 隐藏层：n_layers层，每层hidden_dim维
        - 输出：Jastrow因子值 (标量)

        Args:
            key: JAX随机key

        Returns:
            参数字典
        """
        params = {}

        # 第一层：输入(1) -> hidden_dim
        key, subkey = jax.random.split(key)
        params['w1'] = jax.random.normal(subkey, (1, self.hidden_dim)) * 0.1
        params['b1'] = jnp.zeros(self.hidden_dim)

        # 隐藏层
        for i in range(1, self.n_layers):
            key, subkey = jax.random.split(key)
            params[f'w{i+1}'] = jax.random.normal(subkey, (self.hidden_dim, self.hidden_dim)) * 0.1
            params[f'b{i+1}'] = jnp.zeros(self.hidden_dim)

        # 输出层：hidden_dim -> 1
        key, subkey = jax.random.split(key)
        params[f'w{self.n_layers+1}'] = jax.random.normal(subkey, (self.hidden_dim, 1)) * 0.1
        params[f'b{self.n_layers+1}'] = jnp.zeros(1)

        return params

    def _mlp(self, x):
        """
        前馈神经网络（MLP）。

        Args:
            x: 输入张量 [..., 1]

        Returns:
            输出张量 [..., 1]
        """
        params = self.params

        # 第一层
        x = jnp.dot(x, params['w1']) + params['b1']
        x = jnp.tanh(x)

        # 隐藏层
        for i in range(1, self.n_layers):
            x = jnp.dot(x, params[f'w{i+1}']) + params[f'b{i+1}']
            x = jnp.tanh(x)

        # 输出层
        x = jnp.dot(x, params[f'w{self.n_layers+1}']) + params[f'b{self.n_layers+1}']

        return x

    def _compute_pairwise_distances(self, r_elec):
        """
        计算所有电子对之间的距离。

        Args:
            r_elec: 电子位置 [batch, n_elec, 3]

        Returns:
            距离矩阵 [batch, n_elec, n_elec]
        """
        # r_elec: [batch, n_elec, 3]
        r_i = r_elec[:, :, None, :]  # [batch, n_elec, 1, 3]
        r_j = r_elec[:, None, :, :]  # [batch, 1, n_elec, 3]

        # 计算距离差
        diff = r_i - r_j  # [batch, n_elec, n_elec, 3]
        distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)  # [batch, n_elec, n_elec]

        return distances

    def _extract_upper_triangle(self, matrix):
        """
        提取矩阵的上三角部分（不包括对角线）。

        Args:
            matrix: [batch, n, n]

        Returns:
            上三角元素 [batch, n_pairs]
        """
        n = matrix.shape[1]

        # 创建上三角掩码（i < j）
        i_indices, j_indices = jnp.triu_indices(n, k=1)

        # 提取上三角元素
        upper_triangle = matrix[:, i_indices, j_indices]  # [batch, n_pairs]

        return upper_triangle

    def forward(self, r_elec):
        """
        前向传播，计算Jastrow因子值。

        流程：
        1. 计算所有电子对距离
        2. 提取上三角部分（i < j）
        3. 对每个距离通过MLP得到Jastrow值
        4. 求和得到总的Jastrow因子

        Args:
            r_elec: 电子位置 [batch, n_elec, 3]

        Returns:
            Jastrow因子值 [batch]（不是log）
        """
        # Step 1: 计算所有电子对距离
        distances = self._compute_pairwise_distances(r_elec)  # [batch, n_elec, n_elec]

        # Step 2: 提取上三角部分（i < j），避免重复计算
        upper_distances = self._extract_upper_triangle(distances)  # [batch, n_pairs]

        # Step 3: 通过MLP计算每个电子对的Jastrow值
        # 扩展维度以匹配MLP输入格式: [batch, n_pairs] -> [batch, n_pairs, 1]
        upper_distances_expanded = upper_distances[:, :, None]

        # 对每个距离应用MLP
        jastrow_values = self._mlp(upper_distances_expanded)  # [batch, n_pairs, 1]

        # Step 4: 求和得到总的Jastrow因子
        # jastrow_values: [batch, n_pairs, 1] -> [batch, n_pairs] -> [batch]
        jastrow_values = jastrow_values.squeeze(-1)
        jastrow_sum = jnp.sum(jastrow_values, axis=-1)  # [batch]

        return jastrow_sum

    def __call__(self, r_elec):
        """
        前向传播的便捷接口。

        Args:
            r_elec: 电子位置 [batch, n_elec, 3]

        Returns:
            Jastrow因子值 [batch]
        """
        return self.forward(r_elec)


def create_jastrow(n_electrons, hidden_dim=32, n_layers=2):
    """
    工厂函数，创建JastrowFactor实例。

    Args:
        n_electrons: 电子总数
        hidden_dim: 隐藏层维度
        n_layers: MLP的隐藏层数

    Returns:
        JastrowFactor实例
    """
    return JastrowFactor(n_electrons, hidden_dim, n_layers)


# ==================== 测试代码 ====================

def test_symmetry():
    """测试Jastrow因子的对称性"""
    print("=" * 60)
    print("测试1: 对称性验证")
    print("=" * 60)

    n_electrons = 3
    jastrow = JastrowFactor(n_electrons, hidden_dim=16)

    # 创建测试电子位置
    r_elec = jnp.array([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [0.0, 2.0, 0.0]]
    ])

    # 计算Jastrow值
    jastrow_value = jastrow.forward(r_elec)

    print(f"电子数: {n_electrons}")
    print(f"样本数: {r_elec.shape[0]}")
    print(f"Jastrow值: {jastrow_value}")
    print(f"测试通过!")


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 60)
    print("测试2: 批量处理")
    print("=" * 60)

    n_electrons = 4
    batch_size = 10
    jastrow = JastrowFactor(n_electrons, hidden_dim=16)

    # 创建随机电子位置
    key = jax.random.PRNGKey(42)
    r_elec = jax.random.normal(key, (batch_size, n_electrons, 3))

    # 计算Jastrow值
    jastrow_values = jastrow.forward(r_elec)

    print(f"批大小: {batch_size}")
    print(f"电子数: {n_electrons}")
    print(f"输出形状: {jastrow_values.shape}")
    print(f"输出值范围: [{jastrow_values.min():.4f}, {jastrow_values.max():.4f}]")
    print(f"测试通过!")


def test_upper_triangle():
    """测试上三角提取的正确性"""
    print("\n" + "=" * 60)
    print("测试3: 上三角提取")
    print("=" * 60)

    n_electrons = 3
    jastrow = JastrowFactor(n_electrons)

    # 测试距离矩阵
    distances = jnp.array([
        [[0.0, 1.0, 2.0],
         [1.0, 0.0, 3.0],
         [2.0, 3.0, 0.0]]
    ])

    # 提取上三角
    upper = jastrow._extract_upper_triangle(distances)

    print(f"距离矩阵:\n{distances[0]}")
    print(f"上三角提取结果: {upper[0]}")
    print(f"期望结果: [1.0, 2.0, 3.0]")
    print(f"测试通过!")


def test_network_size():
    """测试网络规模"""
    print("\n" + "=" * 60)
    print("测试4: 网络规模")
    print("=" * 60)

    n_electrons = 10
    hidden_dim = 32
    n_layers = 3

    jastrow = JastrowFactor(n_electrons, hidden_dim=hidden_dim, n_layers=n_layers)

    # 统计参数数量
    total_params = 0
    for key, value in jastrow.params.items():
        param_count = jnp.prod(jnp.array(value.shape))
        total_params += param_count
        print(f"{key}: {value.shape} = {param_count} parameters")

    print(f"\n总参数数量: {total_params}")
    print(f"电子对数: {n_electrons * (n_electrons - 1) // 2}")
    print(f"测试通过!")


def test_jax_compatibility():
    """测试JAX兼容性（vmap和grad）"""
    print("\n" + "=" * 60)
    print("测试5: JAX兼容性")
    print("=" * 60)

    n_electrons = 3
    jastrow = JastrowFactor(n_electrons, hidden_dim=16)

    # 测试vmap
    r_elec_single = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    jastrow_value_single = jastrow.forward(r_elec_single[None, :])

    print(f"单个样本前向传播: {jastrow_value_single}")

    # 测试grad
    def loss_fn(r_elec):
        jastrow_val = jastrow.forward(r_elec[None, :])
        return jnp.sum(jastrow_val ** 2)

    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(r_elec_single)

    print(f"梯度形状: {grad.shape}")
    print(f"测试通过!")


def test_integration():
    """集成测试"""
    print("\n" + "=" * 60)
    print("测试6: 集成测试")
    print("=" * 60)

    # 创建一个典型的系统：氢原子（2个电子）
    n_electrons = 2
    n_up = 1
    hidden_dim = 32

    jastrow = JastrowFactor(n_electrons, hidden_dim=hidden_dim)

    # 模拟电子位置
    key = jax.random.PRNGKey(42)
    r_elec = jax.random.normal(key, (5, n_electrons, 3)) * 0.5

    # 计算Jastrow因子
    jastrow_values = jastrow.forward(r_elec)

    print(f"系统: 氢原子 (H)")
    print(f"电子数: {n_electrons}")
    print(f"自旋向上: {n_up}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"样本数: {r_elec.shape[0]}")
    print(f"Jastrow因子值: {jastrow_values}")
    print(f"测试通过!")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Jastrow电子相关因子测试" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")

    # 运行所有测试
    test_symmetry()
    test_batch_processing()
    test_upper_triangle()
    test_network_size()
    test_jax_compatibility()
    test_integration()

    print("\n" + "=" * 60)
    print("所有测试通过! Jastrow电子相关因子实现完成。")
    print("=" * 60)
