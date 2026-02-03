"""
FermiNet Stage 2 - 稳定化训练配置
解决数值不稳定问题，采用保守超参数
"""

import jax
import jax.numpy as jnp
import jax.random as random
import time
from pathlib import Path

import sys
import os
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# 导入模块
from ferminet.network import ExtendedFermiNet
from ferminet.trainer import ExtendedTrainer
from ferminet.mcmc import FixedStepMCMC
import ferminet.physics as physics

# ============================================================================
# 配置 - 保守稳定化参数
# ============================================================================

config = {
    'name': 'H2_Stage2_Stable',
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        'charges': jnp.array([1.0, 1.0])
    },
    'network': {
        # 从单行列式开始，逐步扩展
        'single_det_config': {
            'single_layer_width': 64,
            'pair_layer_width': 8,
            'num_interaction_layers': 2,
            'determinant_count': 1,  # 从1开始
            'use_residual': True,
            'use_jastrow': False,
        },
        # 多行列式配置（后续使用）
        'multi_det_config': {
            'single_layer_width': 128,
            'pair_layer_width': 16,
            'num_interaction_layers': 3,
            'determinant_count': 2,  # 然后扩展到2
            'use_residual': True,
            'use_jastrow': False,
        },
    },
    'mcmc': {
        'n_samples': 128,  # 降低批次大小
        'step_size': 0.15,
        'n_steps': 3,
        'thermalization_steps': 20,
    },
    'training': {
        'phase1_epochs': 50,  # 单行列式训练轮数
        'phase2_epochs': 50,  # 多行列式训练轮数
        'print_interval': 5,
    },
    # 关键：降低学习率和增强梯度裁剪
    'learning_rate': 0.0001,  # 从0.001降低10倍
    'gradient_clip': 0.1,     # 从1.0增强10倍
    'gradient_clip_norm': 'inf',
    'use_scheduler': True,
    'scheduler_patience': 10,
    'decay_factor': 0.5,
    'min_lr': 1e-6,
    'target_energy': -1.174,
    'seed': 42
}

print("=" * 80)
print("FermiNet Stage 2 - 稳定化训练")
print("=" * 80)

print("\n稳定化配置:")
print(f"  初始学习率: {config['learning_rate']:.6f} (降低10倍)")
print(f"  梯度裁剪: {config['gradient_clip']:.2f} (增强10倍)")
print(f"  阶段1: 单行列式，{config['training']['phase1_epochs']}轮")
print(f"  阶段2: 多行列式，{config['training']['phase2_epochs']}轮")
print(f"  样本数: {config['mcmc']['n_samples']}")
print(f"  预热步数: {config['mcmc']['thermalization_steps']}")

# ============================================================================
# 训练阶段1：单行列式
# ============================================================================

def train_phase(config, phase_name, network_config, n_epochs, initial_params=None, initial_r_elec=None, seed=42):
    """
    训练单个阶段

    Returns:
        params: 最终参数
        r_elec: 最终电子位置
        history: 训练历史
    """
    print("\n" + "=" * 80)
    print(f"开始阶段: {phase_name}")
    print("=" * 80)

    print(f"\n网络配置:")
    for k, v in network_config.items():
        print(f"  {k}: {v}")

    # 初始化 RNG
    key = random.PRNGKey(seed)

    # 提取配置
    n_electrons = config['n_electrons']
    n_up = config['n_up']
    nuclei_config = config['nuclei']
    nuclei_pos = nuclei_config['positions']
    nuclei_charge = nuclei_config['charges']

    # 创建网络
    print("\n创建网络...")
    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, network_config)
    net_info = network.get_network_info()
    print(f"  网络类型: {net_info['type']}")
    print(f"  总参数: {net_info['total_parameters']:,}")

    # 如果提供了初始参数，加载它们
    if initial_params is not None:
        print("\n从初始参数继续...")
        # 简单的参数转移（只转移存在的键）
        for key_name in initial_params.keys():
            if key_name in network.params:
                if initial_params[key_name].shape == network.params[key_name].shape:
                    network.params[key_name] = initial_params[key_name]
                else:
                    print(f"  警告: 参数 {key_name} 形状不匹配，保留初始化")
            else:
                print(f"  警告: 参数 {key_name} 不存在于新网络")

    # 创建 MCMC 采样器
    print("\n创建 MCMC 采样器...")
    mcmc = FixedStepMCMC(
        step_size=config['mcmc']['step_size'],
        n_steps=config['mcmc']['n_steps']
    )

    # 创建训练器
    print("\n创建训练器...")
    trainer = ExtendedTrainer(network, mcmc, config)

    # 初始化或使用提供的电子位置
    if initial_r_elec is not None:
        print("\n使用提供的电子位置...")
        r_elec = initial_r_elec
    else:
        print("\n初始化电子位置...")
        n_samples = config['mcmc']['n_samples']
        key, init_key = random.split(key)
        r_elec = nuclei_pos[random.randint(init_key, (n_samples, n_electrons), 0, len(nuclei_pos))]
        key, offset_key = random.split(key)
        r_elec += random.normal(offset_key, r_elec.shape) * 0.1

    # 预热 MCMC
    print("预热 MCMC...")
    def log_psi_fn(r_batch):
        return network(r_batch)

    key, warmup_key = random.split(key)
    r_elec, key = mcmc.warmup(
        log_psi_fn,
        r_elec,
        warmup_key,
        n_warmup_steps=config['mcmc']['thermalization_steps']
    )

    # 计算初始能量
    print("\n计算初始能量...")
    params = network.params
    n_samples = config['mcmc']['n_samples']

    def log_psi_single(r):
        r_batch = r[None, :, :]
        return network(r_batch)[0]

    def compute_energy(r_elec_batch):
        energies = []
        for i in range(r_elec_batch.shape[0]):
            e_l = physics.local_energy(log_psi_single, r_elec_batch[i], nuclei_pos, nuclei_charge)
            energy_scalar = float(jnp.ravel(e_l)[0])
            energies.append(energy_scalar)
        return energies

    initial_energies = compute_energy(r_elec)
    initial_energy = sum(initial_energies) / len(initial_energies)
    initial_variance = sum((e - initial_energy)**2 for e in initial_energies) / len(initial_energies)
    print(f"  初始能量: {initial_energy:.6f} Ha, 方差: {initial_variance:.6f}")
    print(f"  目标能量: {config['target_energy']:.6f} Ha")

    # 训练循环
    print("\n" + "=" * 80)
    print(f"开始训练 ({n_epochs} 轮)")
    print("=" * 80)

    best_energy = float('inf')
    best_params = None
    history = {
        'phase': phase_name,
        'epochs': [],
        'energies': [],
        'variances': [],
        'accept_rates': [],
        'epoch_times': [],
        'grad_norms': [],
        'learning_rates': []
    }

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_timer_start = time.time()

        # 训练步骤
        key, step_key = random.split(key)
        params, mean_E, accept_rate, r_elec, train_info = trainer.train_step(
            params,
            r_elec,
            step_key,
            nuclei_pos,
            nuclei_charge
        )

        # 更新网络参数
        network.params = params

        # 计算当前能量
        current_energies = compute_energy(r_elec)
        current_energy = sum(current_energies) / len(current_energies)
        variance = sum((e - current_energy)**2 for e in current_energies) / len(current_energies)

        # 记录训练统计
        trainer.record_training_stats(current_energy, variance, accept_rate)

        # 更新调度器
        if trainer.use_scheduler:
            new_lr, decayed, old_lr = trainer.update_scheduler(current_energy)

        # 检查稳定性
        if jnp.isnan(current_energy) or jnp.isinf(current_energy):
            print(f"\n!!! 警告: Epoch {epoch+1} 能量异常 ({current_energy:.6f})")
            print("恢复到最佳参数...")
            if best_params is not None:
                params = best_params
                network.params = params
                continue

        # 检查是否获得最佳能量
        if current_energy < best_energy:
            best_energy = current_energy
            best_params = params.copy()

        # 记录历史
        epoch_time = time.time() - epoch_timer_start
        history['epochs'].append(epoch + 1)
        history['energies'].append(current_energy)
        history['variances'].append(variance)
        history['accept_rates'].append(accept_rate)
        history['epoch_times'].append(epoch_time)
        history['grad_norms'].append(train_info['grad_norm'])
        history['learning_rates'].append(train_info['learning_rate'])

        # 打印进度
        if (epoch + 1) % config['training']['print_interval'] == 0 or epoch == 0:
            elapsed_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  能量: {current_energy:.6f} Ha (最佳: {best_energy:.6f})")
            print(f"  方差: {variance:.6f}")
            print(f"  接受率: {accept_rate:.3f}")
            print(f"  梯度范数: {train_info['grad_norm']:.6f}")
            print(f"  学习率: {train_info['learning_rate']:.6f}")
            print(f"  误差: {abs(current_energy - config['target_energy']):.6f} Ha")
            print(f"  时间: {epoch_time:.2f}s, 总计: {elapsed_time:.1f}s")

            # 检查数值稳定性
            if variance > 1000:
                print(f"  ⚠️  警告: 方差过大 ({variance:.2e})")
            if train_info['grad_norm'] > 1.0:
                print(f"  ⚠️  警告: 梯度范数过大 ({train_info['grad_norm']:.2f})")

    # 阶段完成
    phase_time = time.time() - start_time
    avg_epoch_time = phase_time / n_epochs

    print("\n" + "=" * 80)
    print(f"{phase_name} 完成！")
    print("=" * 80)
    print(f"\n阶段结果:")
    print(f"  最终能量: {current_energy:.6f} Ha")
    print(f"  最佳能量: {best_energy:.6f} Ha")
    print(f"  能量误差: {abs(best_energy - config['target_energy']):.6f} Ha")
    print(f"  最终方差: {variance:.6f}")
    print(f"  阶段时间: {phase_time:.1f}s ({phase_time/60:.1f}分钟)")
    print(f"  平均 epoch 时间: {avg_epoch_time:.2f}s")

    return params, r_elec, history, best_params

# ============================================================================
# 主执行流程
# ============================================================================

# Phase 1: 单行列式训练
params_phase1, r_elec_phase1, history_phase1, best_params_phase1 = train_phase(
    config,
    "Phase 1 - 单行列式",
    config['network']['single_det_config'],
    config['training']['phase1_epochs'],
    seed=config['seed']
)

# Phase 2: 多行列式训练（使用Phase 1的最佳参数）
params_phase2, r_elec_phase2, history_phase2, best_params_phase2 = train_phase(
    config,
    "Phase 2 - 多行列式",
    config['network']['multi_det_config'],
    config['training']['phase2_epochs'],
    initial_params=best_params_phase1,
    initial_r_elec=r_elec_phase1,
    seed=config['seed'] + 1  # 不同的种子
)

# ============================================================================
# 最终结果汇总
# ============================================================================

print("\n" + "=" * 80)
print("全部训练完成！")
print("=" * 80)

total_time_per_epoch_time = (
    sum(history_phase1['epoch_times']) / len(history_phase1['epoch_times']) +

    sum(history_phase2['epoch_times']) / len(history_phase2['epoch_times'])
) / 2

print(f"\n最终结果:")
print(f"  Phase 1 最佳能量: {min(history_phase1['energies']):.6f} Ha")
print(f"  Phase 2 最佳能量: {min(history_phase2['energies']):.6f} Ha")
print(f"  目标能量: {config['target_energy']:.6f} Ha")
print(f"  最终误差: {abs(min(history_phase2['energies']) - config['target_energy']):.6f} Ha")

print(f"\n训练统计:")
print(f"  Phase 1 轮数: {len(history_phase1['epochs'])}")
print(f"  Phase 2 轮数: {len(history_phase2['epochs'])}")
print(f"  平均 epoch 时间: {total_time_per_epoch_time:.2f}s")

# 保存结果
results_dir = Path("results/stage2_stable")
results_dir.mkdir(parents=True, exist_ok=True)

results = {
    'config': config,
    'phase1': {
        'history': history_phase1,
        'best_energy': min(history_phase1['energies']),
        'final_energy': history_phase1['energies'][-1],
    },
    'phase2': {
        'history': history_phase2,
        'best_energy': min(history_phase2['energies']),
        'final_energy': history_phase2['energies'][-1],
    },
    'target_energy': config['target_energy'],
    'best_energy': min(history_phase2['energies']),
    'energy_error': abs(min(history_phase2['energies']) - config['target_energy']),
}

import pickle
save_path = results_dir / f"{config['name']}_results.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\n结果已保存至: {save_path}")

print("\n" + "=" * 80)
print("Stage 2 稳定化训练完成！")
print("=" * 80)
