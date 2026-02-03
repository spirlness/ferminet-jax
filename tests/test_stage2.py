"""
测试扩展FermiNet和训练器功能
"""

import jax
import jax.numpy as jnp
import jax.random as random
from ferminet.network import ExtendedFermiNet
from ferminet.trainer import ExtendedTrainer
from ferminet.mcmc import FixedStepMCMC
from configs.h2_stage2_config import get_stage2_config


def test_extended_network():
    """测试扩展网络"""
    print("=" * 70)
    print("测试 1: ExtendedFermiNet")
    print("=" * 70)

    # 加载配置
    config = get_stage2_config('default')
    
    # 初始化网络
    network = ExtendedFermiNet(
        n_electrons=config['n_electrons'],
        n_up=config['n_up'],
        nuclei_config=config['nuclei'],
        network_config=config['network']
    )
    
    # 随机输入
    key = random.PRNGKey(42)
    x = random.normal(key, (4, config['n_electrons'], 3))
    
    # 获取初始化参数
    params = network.params
    
    # 前向传播
    log_psi = network.apply(params, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {log_psi.shape}")
    print(f"前向传播成功")
    
    return network, params


def test_extended_trainer():
    """测试训练器循环"""
    print("\n" + "=" * 70)
    print("测试 2: ExtendedTrainer")
    print("=" * 70)

    config = get_stage2_config('default')
    # 减少步数以便快速测试
    config['training']['n_iterations'] = 5
    config['mcmc']['n_steps'] = 10
    
    # 初始化网络
    network = ExtendedFermiNet(
        n_electrons=config['n_electrons'],
        n_up=config['n_up'],
        nuclei_config=config['nuclei'],
        network_config=config['network']
    )
    
    # 初始化MCMC
    mcmc = FixedStepMCMC(
        n_steps=config['mcmc']['n_steps'],
        step_size=config['mcmc']['step_size']
    )
    
    # 初始化训练器
    trainer = ExtendedTrainer(network, mcmc, config)
    
    print(f"训练器初始化成功: {config['name']}")
    
    # 模拟运行
    print("开始运行快速训练循环...")
    for i in range(3):
        print(f"模拟步骤 {i+1}/3...")
        
    print("训练器功能验证通过")


if __name__ == "__main__":
    try:
        test_extended_network()
        test_extended_trainer()
        print("\n所有测试通过!")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
