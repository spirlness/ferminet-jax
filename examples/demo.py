"""
FermiNet阶段1 - 主程序
实现完整的训练流程
"""

import jax
import jax.numpy as jnp
import jax.random as random
import time
from typing import List, Tuple, Dict

import sys
import os
# Add src directory and project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ferminet.network import ExtendedFermiNet
from ferminet.mcmc import FixedStepMCMC
from ferminet.trainer import VMCTrainer


def print_banner():
    """打印程序横幅"""
    print("=" * 80)
    print(" " * 20 + "FermiNet Stage 1 - Demo Training")
    print("=" * 80)
    print("Simplified single-determinant FermiNet implementation")
    print("Target: Variational Monte Carlo training for electronic structure")
    print("=" * 80)
    print()


def initialize_system(config: Dict) -> Tuple[ExtendedFermiNet, FixedStepMCMC, VMCTrainer]:
    """
    初始化网络、MCMC采样器和训练器

    Parameters
    ----------
    config : dict
        配置字典

    Returns
    -------
    Tuple[ExtendedFermiNet, FixedStepMCMC, VMCTrainer]
        网络、MCMC采样器和训练器
    """
    print("Initializing system...")
    print(f"  Molecule: {config['name']}")
    print(f"  Electrons: {config['n_electrons']} (up: {config['n_up']}, down: {config['n_electrons'] - config['n_up']})")
    print(f"  Nuclei: {len(config['nuclei']['charges'])} with charges {list(config['nuclei']['charges'])}")
    print()

    # 初始化网络
    print("Initializing FermiNet network...")
    network = ExtendedFermiNet(
        n_electrons=config['n_electrons'],
        n_up=config['n_up'],
        nuclei_config=config['nuclei'],
        network_config=config['network']
    )
    print("  Network structure:")
    print(f"    Single layer width: {config['network']['single_layer_width']}")
    print(f"    Pair layer width: {config['network']['pair_layer_width']}")
    print(f"    Interaction layers: {config['network']['num_interaction_layers']}")
    print(f"    Determinant count: {config['network']['determinant_count']}")
    print()

    # 初始化MCMC
    print("Initializing MCMC sampler...")
    mcmc = FixedStepMCMC(
        step_size=config['mcmc']['step_size'],
        n_steps=config['mcmc']['n_steps']
    )
    print(f"  Step size: {config['mcmc']['step_size']}")
    print(f"  Steps per sample: {config['mcmc']['n_steps']}")
    print()

    # 初始化训练器
    print("Initializing VMC trainer...")
    trainer = VMCTrainer(network, mcmc, config)
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Adam (beta1={config['beta1']}, beta2={config['beta2']})")
    print()

    return network, mcmc, trainer


def initialize_electrons(config: Dict, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    初始化电子位置

    Parameters
    ----------
    config : dict
        配置字典
    key : jnp.ndarray
        JAX随机数密钥

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        电子位置和更新后的随机数密钥
    """
    key, init_key = random.split(key)

    # 初始化电子位置：从以核为中心的高斯分布中采样
    n_samples = config['mcmc']['n_samples']
    n_electrons = config['n_electrons']

    # 简单的初始化：从标准正态分布采样
    r_elec = random.normal(init_key, (n_samples, n_electrons, 3))

    print("Initializing electron positions...")
    print(f"  Number of samples: {n_samples}")
    print(f"  Initial positions sampled from N(0,1)")
    print()

    return r_elec, key


def thermalize(config: Dict,
               network: ExtendedFermiNet,
               mcmc: FixedStepMCMC,
               r_elec: jnp.ndarray,
               key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    预热MCMC采样器，使电子位置分布接近平衡分布

    Parameters
    ----------
    config : dict
        配置字典
    network : ExtendedFermiNet
        网络实例
    mcmc : FixedStepMCMC
        MCMC采样器
    r_elec : jnp.ndarray
        电子位置
    key : jnp.ndarray
        随机数密钥

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        预热后的电子位置和更新后的随机数密钥
    """
    print("Thermalizing MCMC sampler...")
    print(f"  Number of thermalization steps: {config['mcmc']['thermalization_steps']}")

    # 创建对数波函数
    def log_psi_fn(r):
        return network(r)

    r_current = r_elec
    accept_rates = []

    start_time = time.time()

    for step in range(config['mcmc']['thermalization_steps']):
        key, sample_key = random.split(key)
        r_current, accept_rate = mcmc.sample(log_psi_fn, r_current, sample_key)
        accept_rates.append(accept_rate)

        if (step + 1) % 10 == 0:
            avg_accept = jnp.mean(jnp.array(accept_rates[-10:]))
            print(f"  Step {step+1}/{config['mcmc']['thermalization_steps']}, Avg accept rate: {avg_accept:.3f}")

    end_time = time.time()
    thermalization_time = end_time - start_time

    avg_accept = jnp.mean(jnp.array(accept_rates))
    print(f"  Thermalization completed in {thermalization_time:.2f}s")
    print(f"  Overall accept rate: {avg_accept:.3f}")
    print()

    return r_current, key


def train(config: Dict,
          trainer: VMCTrainer,
          r_elec: jnp.ndarray,
          key: jnp.ndarray) -> Tuple[List[float], Dict, jnp.ndarray]:
    """
    训练循环

    Parameters
    ----------
    config : dict
        配置字典
    trainer : VMCTrainer
        训练器
    r_elec : jnp.ndarray
        电子位置
    key : jnp.ndarray
        随机数密钥

    Returns
    -------
    Tuple[List[float], Dict, jnp.ndarray]
        能量历史、训练统计信息和最终电子位置
    """
    print("Starting training...")
    print(f"  Number of epochs: {config['training']['n_epochs']}")
    print(f"  Print interval: {config['training']['print_interval']}")
    print()
    print("Epoch | Energy (Ha) | Accept Rate | Error vs FHF (mHa)")
    print("-" * 60)

    # 获取核信息
    nuclei_pos = config['nuclei']['positions']
    nuclei_charge = config['nuclei']['charges']

    # 训练循环
    energies = []
    accept_rates = []
    params = trainer.network.params
    r_current = r_elec

    start_time = time.time()

    for epoch in range(config['training']['n_epochs']):
        # 执行训练步
        key, train_key = random.split(key)
        params, energy, accept_rate, r_current = trainer.train_step(
            params, r_current, train_key, nuclei_pos, nuclei_charge
        )

        # 更新网络参数
        trainer.network.params = params

        # 记录统计信息
        energies.append(float(energy))
        accept_rates.append(float(accept_rate))

        # 打印进度
        if (epoch + 1) % config['training']['print_interval'] == 0:
            error = abs(energy - config['target_energy']) * 1000
            print(f"{epoch+1:5d} | {energy:12.6f} | {accept_rate:11.3f} | {error:16.3f}")

    end_time = time.time()
    training_time = end_time - start_time

    print("-" * 60)
    print(f"Training completed in {training_time:.2f}s")
    print()

    # 计算最终统计
    final_energy = energies[-1]
    error = abs(final_energy - config['target_energy']) * 1000
    avg_accept = jnp.mean(jnp.array(accept_rates))

    print("Final Results:")
    print(f"  Final Energy: {final_energy:.6f} Ha")
    print(f"  Target Energy: {config['target_energy']:.6f} Ha")
    print(f"  Error: {error:.3f} mHa")
    print(f"  Average Accept Rate: {avg_accept:.3f}")
    print()

    # 训练统计
    stats = {
        'final_energy': final_energy,
        'error': error,
        'avg_accept_rate': avg_accept,
        'training_time': training_time,
        'energies': energies,
        'accept_rates': accept_rates,
        'n_epochs': config['training']['n_epochs']
    }

    return energies, stats, r_current


def save_results(config: Dict, stats: Dict, params: Dict, output_dir: str = "G:\\FermiNet\\demo\\results"):
    """
    保存训练结果

    Parameters
    ----------
    config : dict
        配置字典
    stats : dict
        训练统计信息
    params : dict
        网络参数
    output_dir : str
        输出目录
    """
    import os
    import pickle

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    timestamp = int(time.time())
    filename = f"{config['name']}_results_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)

    results = {
        'config': config,
        'stats': stats,
        'params': params
    }

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {filepath}")


def main():
    """主函数"""
    # 打印横幅
    print_banner()

    # 导入配置
    from configs.h2_config import H2_CONFIG

    config = H2_CONFIG

    # 初始化随机数生成器
    key = random.PRNGKey(config['seed'])

    # 初始化系统
    network, mcmc, trainer = initialize_system(config)

    # 初始化电子位置
    r_elec, key = initialize_electrons(config, key)

    # 预热
    r_elec, key = thermalize(config, network, mcmc, r_elec, key)

    # 训练
    energies, stats, r_elec = train(config, trainer, r_elec, key)

    # 保存结果
    save_results(config, stats, trainer.network.params)

    print("=" * 80)
    print("Training finished successfully!")
    print("=" * 80)

    return energies, stats


if __name__ == "__main__":
    energies, stats = main()
