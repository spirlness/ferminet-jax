"""
残差连接层实现
用于FermiNet变分波函数网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    ResNet风格的残差连接块

    架构：
    - 如果维度匹配：y = F(x) + x
    - 如果维度不匹配：y = F(x)

    应用场景：
    - 电子对之间的多体波函数
    - 防止梯度消失
    - 加深网络而不损失性能
    """

    def __init__(self, in_dim, out_dim, activation='silu', use_layer_norm=True):
        """
        初始化残差块

        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            activation: 激活函数类型 ('silu', 'gelu', 'relu', 'tanh')
            use_layer_norm: 是否使用层归一化
        """
        super(ResidualBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = (in_dim == out_dim)

        # 选择激活函数
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 线性变换层
        self.linear = nn.Linear(in_dim, out_dim)

        # 可选的层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)
        else:
            self.layer_norm = None

        # 初始化权重（Xavier初始化适合tanh，He初始化适合ReLU）
        if activation in ['silu', 'gelu', 'relu']:
            nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, ..., in_dim)

        Returns:
            y: 输出张量，形状 (batch_size, ..., out_dim)
        """
        # 应用线性变换
        out = self.linear(x)

        # 应用激活函数
        out = self.activation(out)

        # 可选的层归一化
        if self.layer_norm is not None:
            out = self.layer_norm(out)

        # 残差连接（仅当维度匹配时）
        if self.use_residual:
            out = out + x

        return out


class MultiLayerResidualBlock(nn.Module):
    """
    多层残差块

    堆叠多个ResidualBlock，常用于深度网络
    """

    def __init__(self, dim, num_layers=2, activation='silu', use_layer_norm=True):
        """
        初始化多层残差块

        Args:
            dim: 特征维度（所有层相同维度）
            num_layers: 残差层数
            activation: 激活函数类型
            use_layer_norm: 是否使用层归一化
        """
        super(MultiLayerResidualBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(
                dim, dim,
                activation=activation,
                use_layer_norm=use_layer_norm
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            y: 输出张量
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class ResidualConnection(nn.Module):
    """
    通用残差连接包装器

    可以将任意模块包装为残差连接形式
    """

    def __init__(self, module, use_residual=True):
        """
        初始化残差连接包装器

        Args:
            module: 要包装的PyTorch模块
            use_residual: 是否使用残差连接
        """
        super(ResidualConnection, self).__init__()

        self.module = module
        self.use_residual = use_residual

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            y: 输出张量
        """
        out = self.module(x)

        if self.use_residual and x.shape == out.shape:
            out = out + x

        return out


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 ResidualBlock")
    print("=" * 60)

    # 测试1: 维度匹配的残差块块
    print("\n测试1: 维度匹配的残差块 (in_dim=64, out_dim=64)")
    block1 = ResidualBlock(64, 64, activation='silu')
    x = torch.randn(32, 64)
    y = block1(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"使用残差连接: {block1.use_residual}")

    # 测试2: 维度不匹配的残差块
    print("\n测试2: 维度不匹配的残差块 (in_dim=32, out_dim=64)")
    block2 = ResidualBlock(32, 64, activation='gelu')
    x = torch.randn(32, 32)
    y = block2(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"使用残差连接: {block2.use_residual}")

    # 测试3: 多层残差块
    print("\n测试3: 多层残差块 (dim=64, num_layers=3)")
    multi_block = MultiLayerResidualBlock(64, num_layers=3, activation='silu')
    x = torch.randn(32, 64)
    y = multi_block(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # 测试4: 不同激活函数
    print("\n测试4: 不同激活函数")
    activations = ['silu', 'gelu', 'relu', 'tanh']
    for act in activations:
        block = ResidualBlock(32, 32, activation=act)
        x = torch.randn(16, 32)
        y = block(x)
        print(f"  {act:8s}: 输入{x.shape} -> 输出{y.shape}")

    # 测试5: 残差连接包装器
    print("\n测试5: 残差连接包装器")
    module = nn.Sequential(
        nn.Linear(64, 128),
        nn.SiLU(),
        nn.Linear(128, 64)
    )
    residual_module = ResidualConnection(module, use_residual=True)
    x = torch.randn(32, 64)
    y = residual_module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
