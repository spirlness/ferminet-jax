"""
氢分子(H2)配置文件
用于FermiNet阶段1的训练
"""

import jax.numpy as jnp

# H2分子配置
H2_CONFIG = {
    # ========== 分子系统配置 ==========
    'n_electrons': 2,           # 总电子数
    'n_up': 1,                  # 自旋向上电子数
    'nuclei': {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],    # 第一个氢核
            [1.4, 0.0, 0.0]     # 第二个氢核，键距1.4 Bohr
        ]),
        'charges': jnp.array([1.0, 1.0])  # 两个氢核，各带+1电荷
    },

    # ========== 网络结构配置 ==========
    'network': {
        'single_layer_width': 16,           # 单电子特征层宽度（快速测试）
        'pair_layer_width': 4,             # 双电子特征层宽度（快速测试）
        'num_interaction_layers': 1,          # 相互作用层数
        'determinant_count': 1,               # 行列式数
    },

    # ========== MCMC采样配置 ==========
    'mcmc': {
        'n_samples': 64,                   # 样本数（快速测试）
        'step_size': 0.15,                 # Langevin步长
        'n_steps': 5,                      # 每次训练的MCMC步数（快速测试）
        'thermalization_steps': 20,      # 预热步数（快速测试）
    },

    # ========== 训练配置 ==========
    'training': {
        'n_epochs': 20,                   # 训练轮数（快速验证）
        'print_interval': 2,               # 打印间隔
    },

    # ========== 优化器配置 ==========
    'learning_rate': 0.001,               # 学习率
    'beta1': 0.9,                         # Adam beta1
    'beta2': 0.999,                       # Adam beta2
    'epsilon': 1e-8,                      # Adam epsilon

    # ========== 其他配置 ==========
    'seed': 42,                           # 随机数种子
    'target_energy': -1.174,             # H2基态能量参考值(Hartree)
    'name': 'H2'                          # 分子系统名称
}

# 氢原子(H)配置（用于测试）
H_CONFIG = {
    'n_electrons': 1,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([[0.0, 0.0, 0.0]]),
        'charges': jnp.array([1.0])
    },
    'network': {
        'single_layer_width': 32,
        'pair_layer_width': 8,
        'num_interaction_layers': 1,
        'determinant_count': 1,
    },
    'mcmc': {
        'n_samples': 256,
        'step_size': 0.15,
        'n_steps': 10,
        'thermalization_steps': 100,
    },
    'training': {
        'n_epochs': 500,
        'print_interval': 10,
    },
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'seed': 42,
    'target_energy': -0.5,  # 氢原子基态能量
    'name': 'H'
}

# 氦原子(He)配置
HE_CONFIG = {
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([[0.0, 0.0, 0.0]]),
        'charges': jnp.array([2.0])
    },
    'network': {
        'single_layer_width': 32,
        'pair_layer_width': 8,
        'num_interaction_layers': 1,
        'determinant_count': 1,
    },
    'mcmc': {
        'n_samples': 256,
        'step_size': 0.15,
        'n_steps': 10,
        'thermalization_steps': 100,
    },
    'training': {
        'n_epochs': 500,
        'print_interval': 10,
    },
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'seed': 42,
    'target_energy': -2.903,  # 氦原子基态能量
    'name': 'He'
}
