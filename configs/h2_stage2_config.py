"""
氢分子(H2) Stage 2 配置文件
用于扩展FermiNet的精细训练
"""

import jax.numpy as jnp

# H2分子Stage 2配置 - 扩展网络架构
H2_STAGE2_CONFIG = {
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

    # ========== 扩展网络结构配置 ==========
    'network': {
        'single_layer_width': 128,          # 单电子特征层宽度（扩展）
        'pair_layer_width': 16,            # 双电子特征层宽度（扩展）
        'num_interaction_layers': 3,        # 相互作用层数（扩展）
        'determinant_count': 4,             # 行列式数（多行列式支持）
        'use_jastrow': False,               # 是否使用Jastrow因子（可选）
        'use_residual': True,               # 是否使用残差连接
        'jastrow_alpha': 0.5,               # Jastrow因子参数（如果使用）
    },

    # ========== MCMC采样配置 ==========
    'mcmc': {
        'n_samples': 2048,                  # 样本数（增加以提高精度）
        'step_size': 0.15,                  # Langevin步长
        'n_steps': 10,                      # 每次训练的MCMC步数
        'thermalization_steps': 100,      # 预热步数（增加）
    },

    # ========== 训练配置 ==========
    'training': {
        'n_epochs': 200,                    # 训练轮数（扩展训练）
        'print_interval': 10,               # 打印间隔
    },

    # ========== 优化器配置 ==========
    'learning_rate': 0.001,                # 初始学习率
    'beta1': 0.9,                          # Adam beta1
    'beta2': 0.999,                        # Adam beta2
    'epsilon': 1e-8,                       # Adam epsilon

    # ========== 梯度裁剪配置 ==========
    'gradient_clip': 1.0,                  # 梯度裁剪阈值
    'gradient_clip_norm': 'inf',           # 裁剪范数类型 ('inf', 'l2', 'l1')

    # ========== 学习率调度器配置 ==========
    'use_scheduler': True,                 # 是否使用能量调度器
    'scheduler_patience': 20,              # 能量改善等待轮数
    'decay_factor': 0.5,                   # 学习率衰减因子
    'min_lr': 1e-5,                        # 最小学习率

    # ========== 其他配置 ==========
    'seed': 42,                            # 随机数种子
    'target_energy': -1.174,              # H2基态能量参考值(Hartree)
    'name': 'H2_Stage2'                    # 分子系统名称
}

# 更激进的配置（用于快速收敛测试）
H2_STAGE2_AGGRESSIVE_CONFIG = {
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0]
        ]),
        'charges': jnp.array([1.0, 1.0])
    },
    'network': {
        'single_layer_width': 128,
        'pair_layer_width': 16,
        'num_interaction_layers': 3,
        'determinant_count': 8,             # 8个行列式（更多行列式）
        'use_jastrow': True,                # 启用Jastrow因子
        'use_residual': True,
        'jastrow_alpha': 0.5,
    },
    'mcmc': {
        'n_samples': 2048,
        'step_size': 0.15,
        'n_steps': 10,
        'thermalization_steps': 100,
    },
    'training': {
        'n_epochs': 200,
        'print_interval': 10,
    },
    'learning_rate': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'gradient_clip': 1.0,
    'gradient_clip_norm': 'inf',
    'use_scheduler': True,
    'scheduler_patience': 15,
    'decay_factor': 0.7,
    'min_lr': 1e-5,
    'seed': 42,
    'target_energy': -1.174,
    'name': 'H2_Stage2_Aggressive'
}

# 精细收敛配置（用于最终精细调整）
H2_STAGE2_FINE_CONFIG = {
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0]
        ]),
        'charges': jnp.array([1.0, 1.0])
    },
    'network': {
        'single_layer_width': 128,
        'pair_layer_width': 16,
        'num_interaction_layers': 3,
        'determinant_count': 6,
        'use_jastrow': False,
        'use_residual': True,
        'jastrow_alpha': 0.5,
    },
    'mcmc': {
        'n_samples': 4096,                  # 更多样本
        'step_size': 0.15,
        'n_steps': 15,                      # 更多MCMC步数
        'thermalization_steps': 200,      # 更长预热
    },
    'training': {
        'n_epochs': 300,                    # 更长训练
        'print_interval': 10,
    },
    'learning_rate': 0.0005,               # 更小初始学习率
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
    'gradient_clip': 0.5,                   # 更严格梯度裁剪
    'gradient_clip_norm': 'inf',
    'use_scheduler': True,
    'scheduler_patience': 30,              # 更多耐心
    'decay_factor': 0.8,
    'min_lr': 1e-6,
    'seed': 42,
    'target_energy': -1.174,
    'name': 'H2_Stage2_Fine'
}

# 获取配置的函数
def get_stage2_config(config_name='default'):
    """
    获取Stage 2配置

    Parameters
    ----------
    config_name : str
        配置名称: 'default', 'aggressive', 'fine'

    Returns
    -------
    dict
        配置字典
    """
    configs = {
        'default': H2_STAGE2_CONFIG,
        'aggressive': H2_STAGE2_AGGRESSIVE_CONFIG,
        'fine': H2_STAGE2_FINE_CONFIG
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]
