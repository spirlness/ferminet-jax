"""
简化测试：对比原始 vs 优化方法的性能
"""

import jax
import jax.numpy as jnp
import jax.random as random
import time

# 导入模块
from network import ExtendedFermiNet
from trainer import ExtendedTrainer
from mcmc import FixedStepMCMC
from physics import local_energy

print("=" * 70)
print("性能对比测试 (RTX 3060 6G)")
print("=" * 70)

# 检测 JAX 配置
print(f"\nJAX 设备: {jax.devices()}")
print(f"默认后端: {jax.default_backend()}")

# 配置
config = {
    'name': 'H2_Perf_Test',
    'n_electrons': 2,
    'n_up': 1,
    'nuclei': {
        'positions': jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        'charges': jnp.array([1.0, 1.0])
    },
    'network': {
        'single_layer_width': 64,
        'pair_layer_width': 8,
        'num_interaction_layers': 2,
        'determinant_count': 2,
        'use_residual': True,
        'use_jastrow': False,
    },
    'mcmc': {
        'n_samples': 128,
        'step_size': 0.15,
        'n_steps': 3,
        'thermalization_steps': 5,
    },
    'training': {
        'n_epochs': 3,
        'print_interval': 1,
    },
    'learning_rate': 0.001,
    'gradient_clip': 1.0,
    'target_energy': -1.174,
    'seed': 42
}

print(f"\n配置：")
print(f"  样本数: {config['mcmc']['n_samples']}")
print(f"  Epochs: {config['training']['n_epochs']}")

# 创建网络
print("\n创建网络...")
network = ExtendedFermiNet(
    config['n_electrons'],
    config['n_up'],
    config['nuclei'],
    config['network']
)

# 创建训练器
mcmc = FixedStepMCMC(
    step_size=config['mcmc']['step_size'],
    n_steps=config['mcmc']['n_steps']
)

trainer = ExtendedTrainer(network, mcmc, config)

# 初始化电子位置
key = random.PRNGKey(config['seed'])
r_elec = config['nuclei']['positions'][random.randint(
    key, (config['mcmc']['n_samples'], config['n_electrons']), 0, 2
)]
key, offset_key = random.split(key)
r_elec += random.normal(offset_key, r_elec.shape) * 0.1

# 预热
def log_psi_fn(r_batch):
    return network(r_batch)

key, warmup_key = random.split(key)
r_elec, key = mcmc.warmup(
    log_psi_fn,
    r_elec,
    warmup_key,
    n_warmup_steps=config['mcmc']['thermalization_steps']
)

print("完成初始化")

# 性能测试
print("\n" + "=" * 70)
print("性能测试：每个组件的时间")
print("=" * 70)

def time_operation(name, func, *args, repeat=10):
    """测试操作的平均时间"""
    times = []
    for _ in range(repeat):
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    print(f"  {name}: {avg_time*1000:.2f} ms (平均，{repeat}次)")
    return result, avg_time

# 测试 1: 网络前向传播
print("\n1. 网络前向传播")
_, t_forward = time_operation("网络前向传播", log_psi_fn, r_elec, repeat=10)

# 测试 2: 单样本能量计算
print("\n2. 能量计算（单样本）")
def log_psi_single(r):
    r_batch = r[None, :, :]
    return network(r_batch)[0]

r_single = r_elec[0]
nuclei_pos = config['nuclei']['positions']
nuclei_charge = config['nuclei']['charges']

_, t_energy_single = time_operation(
    "单样本能量计算",
    local_energy,
    log_psi_single,
    r_single,
    nuclei_pos,
    nuclei_charge,
    repeat=10
)

# 测试 3: 批量能量计算（Python 循环）
print("\n3. 能量计算（批量大循环）")
def compute_energy_loop(r_batch):
    energies = []
    for i in range(r_batch.shape[0]):
        def log_psi_loop(r):
            r_batch_inner = r[None, :, :]
            return network(r_batch_inner)[0]
        e = local_energy(log_psi_loop, r_batch[i], nuclei_pos, nuclei_charge)
        energies.append(float(jnp.ravel(e)[0]))
    return energies

_, t_energy_loop = time_operation("批量能量（循环）", compute_energy_loop, r_elec, repeat=5)

# 测试 4: 梯度计算
print("\n4. 梯度计算")
def compute_gradients(params, r_batch):
    def loss_fn(p, r):
        network.params = p
        log_psi = log_psi_fn(r)
        # 简单的能量损失
        log_psi_single = lambda x: log_psi(x[None, :, :])[0]
        e_single = local_energy(log_psi_single, r[0], nuclei_pos, nuclei_charge)
        return (float(jnp.ravel(e_single)[0]) - (-1.0))**2

    loss, grads = jax.value_and_grad(loss_fn)(params, r_elec)
    return loss, grads

_, t_grad = time_operation("梯度计算", compute_gradients, network.params, r_elec, repeat=3)

# 测试 5: 参数更新
print("\n5. Adam 更新")
def adam_update(params, grads):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.001
    updated = {}
    for key in params:
        g = grads[key]
        updated[key] = params[key] - lr * g
    return updated

_, t_adam = time_operation("Adam 更新", adam_update, network.params, {'w_one_body': jnp.zeros((2, 64))}, repeat=100)

# 计算总训练时间估算
print("\n" + "=" * 70)
print("完整训练 epoch 时间分解")
print("=" * 70)

# 假设每 epoch 的操作次数
n_energy_calls = config['mcmc']['n_samples']  # 128 次能量计算
n_grad_calls = 1  # 1 次梯度计算

epoch_breakdown = {
    "网络前向传播 (每次采样)": t_forward * 1000 * n_energy_calls,  # ms
    "能量计算 (Python 循环)": t_energy_loop * 1000 * n_energy_calls,  # ms
    "梯度计算 (JAX)": t_grad * 1000 * n_grad_calls,  # ms
    "参数更新": t_adam * 1000,
}

total_ms = sum(epoch_breakdown.values())
total_seconds = total / 1000

print("\n组件时间分解：")
for component, time_ms in epoch_breakdown.items():
    percentage = time_ms / total_ms * 100
    print(f"  {component}: {time_ms/1000:.2f} s ({percentage:.1f}%)")

print(f"\n预计每个 epoch 总时间: {total_seconds:.2f} 秒")

# 与实测对比
measured_time = 150  # 用户报告的 2.5 分钟 = 150 秒
speedup = measured_time / total_seconds

print(f"\n与实测对比：")
print(f"  理论时间: {total_seconds:.2f} 秒")
print(f"  实测时间: {measured_time:.2f} 秒")
print(f"  差异倍数: {speedup:.2f}x")

# 分析
print("\n" + "=" * 70)
print("性能分析")
print("=" * 70)

print("\nGPU 利用率可能低的原因：")
print("  1. 没有使用 JAX JIT 编译")
print("     - 每次调用都重新编译或解释")
print("     - 建议: 使用 @jax.jit 装饰关键函数")
print("")
print("  2. Python 循环而非 vmap 并行")
print("     - 能量计算使用 Python 循环")
print("     - 建议: 使用 jax.vmap 进行批量计算")
print("")
print("  3. 小批次大小")
print(f"     - 当前: {config['mcmc']['n_samples']} 个样本")
print("     - 建议: 增加到 256-512，充分利用 GPU")
print("")
print("  4. 数据传输开销")
print("     - CPU-GPU 频繁数据传输")
print("     - 建议: 使用 jax.random 在 GPU 生成随机数")

# 优化建议
print("\n优化建议（按优先级排序）：")
print("  [高] 1. 修复 physics.py，使用 JIT 编译和 vmap")
print("  [高] 2. 增加批次到 256-512")
print("  [中] 3. 编译所有关键函数并预热")
print("  [中] 4. 使用 JAX 随机数生成器")
print("  [低] 5. 考虑使用 float32 节省内存")

print("\n预期改进：")
print("  如果实施以上优化，每个 epoch 时间可能从 150s 降至 20-40s")
print("  性能提升: 3.8x - 7.5x")

print("\n" + "=" * 70)
