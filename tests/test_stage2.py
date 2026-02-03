"""
测试扩展FermiNet和训练器功能
"""

import jax
import jax.numpy as jnp
import jax.random as random
from network import ExtendedFermiNet
from trainer import ExtendedTrainer
from mcmc import FixedStepMCMC
from configs.h2_stage2_config import get_stage2_config


def test_extended_network():
    """测试扩展网络"""
    print("=" * 70)
    print("测试 1: ExtendedFermiNet")
    print("=" * 70)

    # 加载配置
    config = get_stage2_config('default')

    # 创建网络
    n_electrons = config['n_electrons']
    n_up = config['n_up']
    nuclei_config = config['nuclei']

    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, config['network'])

    # 打印网络信息
    info = network.get_network_info()
    print(f"\n网络信息:")
    print(f"  类型: {info['type']}")
    print(f"  总参数: {info['total_parameters']:,}")
    print(f"  单电子层宽度: {info['single_layer_width']}")
    print(f"  双电子层宽度: {info['pair_layer_width']}")
    print(f"  相互作用层数: {info['num_interaction_layers']}")
    print(f"  行列式数: {info['determinant_count']}")
    print(f"  使用残差连接: {info['use_residual']}")
    print(f"  使用Jastrow因子: {info['use_jastrow']}")

    # 测试前向传播
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    r_elec = random.normal(subkey, (4, n_electrons, 3)) * 0.1

    log_psi = network(r_elec)
    print(f"\n前向传播测试:")
    print(f"  输入形状: {r_elec.shape}")
    print(f"  输出形状: {log_psi.shape}")
    print(f"  输出值: {log_psi}")

    print("\n[PASS] ExtendedFermiNet测试通过!")
    return True


def test_energy_scheduler():
    """测试能量调度器"""
    print("\n" + "=" * 70)
    print("测试 2: EnergyBasedScheduler")
    print("=" * 70)

    from trainer import EnergyBasedScheduler

    # 创建调度器
    scheduler = EnergyBasedScheduler(
        initial_lr=0.001,
        target_energy=-1.0,
        patience=3,
        decay_factor=0.5,
        min_lr=1e-5
    )

    print(f"\n初始学习率: {scheduler.get_lr():.6f}")
    print(f"目标能量: {scheduler.target_energy:.3f}")

    # 模拟能量变化
    energies = [-1.5, -1.2, -1.1, -1.05, -1.05, -1.05, -1.05]

    for i, energy in enumerate(energies):
        lr, decayed, old_lr = scheduler.step(energy)
        if decayed:
            print(f"Epoch {i+1}: 能量={energy:.3f}, 学习率衰减 {old_lr:.6f} -> {lr:.6f}")
        else:
            print(f"Epoch {i+1}: 能量={energy:.3f}, 学习率={lr:.6f}")

    print("\n[PASS] EnergyBasedScheduler测试通过!")
    return True


def test_extended_trainer():
    """测试扩展训练器"""
    print("\n" + "=" * 70)
    print("测试 3: ExtendedTrainer")
    print("=" * 70)

    # 加载配置
    config = get_stage2_config('default')

    # 创建网络
    n_electrons = config['n_electrons']
    n_up = config['n_up']
    nuclei_config = config['nuclei']
    nuclei_pos = nuclei_config['positions']
    nuclei_charge = nuclei_config['charges']

    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, config['network'])

    # 创建MCMC
    mcmc = FixedStepMCMC(
        step_size=config['mcmc']['step_size'],
        n_steps=config['mcmc']['n_steps']
    )

    # 创建训练器
    trainer = ExtendedTrainer(network, mcmc, config)

    print(f"\n训练器信息:")
    print(f"  学习率: {trainer.learning_rate:.6f}")
    print(f"  梯度裁剪: {trainer.gradient_clip}")
    print(f"  使用调度器: {trainer.use_scheduler}")

    if trainer.use_scheduler:
        scheduler_info = trainer.scheduler.get_info()
        print(f"  调度器目标能量: {scheduler_info['target_energy']:.3f}")

    # 测试梯度裁剪
    key = random.PRNGKey(42)
    test_grads = {
        'w': jnp.array([1.5, 2.0, 3.0]),
        'b': jnp.array([0.8])
    }

    clipped_grads, grad_norm = trainer._clip_gradients(test_grads, max_norm=1.0, norm_type='inf')
    print(f"\n梯度裁剪测试:")
    print(f"  原始梯度: w={test_grads['w']}")
    print(f"  梯度范数: {grad_norm:.3f}")
    print(f"  裁剪后梯度: w={clipped_grads['w']}")

    print("\n[PASS] ExtendedTrainer测试通过!")
    return True


def test_full_integration():
    """完整集成测试"""
    print("\n" + "=" * 70)
    print("测试 4: 完整集成测试")
    print("=" * 70)

    # 加载配置
    config = get_stage2_config('default')

    # 创建网络
    n_electrons = config['n_electrons']
    n_up = config['n_up']
    nuclei_config = config['nuclei']
    nuclei_pos = nuclei_config['positions']
    nuclei_charge = nuclei_config['charges']

    network = ExtendedFermiNet(n_electrons, n_up, nuclei_config, config['network'])

    # 创建MCMC
    mcmc = FixedStepMCMC(
        step_size=config['mcmc']['step_size'],
        n_steps=config['mcmc']['n_steps']
    )

    # 创建训练器
    trainer = ExtendedTrainer(network, mcmc, config)

    # 初始化电子位置
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    r_elec = random.normal(init_key, (config['mcmc']['n_samples'], n_electrons, 3)) * 0.1

    # 预热MCMC
    def log_psi_fn(r_batch):
        return network(r_batch)

    key, warmup_key = random.split(key)
    r_elec, key = mcmc.warmup(
        log_psi_fn,
        r_elec,
        warmup_key,
        n_warmup_steps=10
    )

    # 执行训练步骤
    print(f"\n执行训练步骤...")
    params = network.params
    for i in range(3):
        key, step_key = random.split(key)
        params, mean_E, accept_rate, r_elec, train_info = trainer.train_step(
            params,
            r_elec,
            step_key,
            nuclei_pos,
            nuclei_charge
        )

        print(f"  Step {i+1}: 能量={mean_E:.6f}, 接受率={accept_rate:.3f}, "
              f"梯度范数={train_info['grad_norm']:.3f}, 学习率={train_info['learning_rate']:.6f}")

    print("\n[PASS] 完整集成测试通过!")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("FermiNet Stage 2 扩展功能测试")
    print("=" * 70)

    tests = [
        test_extended_network,
        test_energy_scheduler,
        test_extended_trainer,
        test_full_integration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] 测试失败: {test.__name__}")
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    if failed == 0:
        print("\n所有测试通过! Stage 2扩展功能正常工作。")
    else:
        print(f"\n警告: {failed}个测试失败，请检查实现。")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
