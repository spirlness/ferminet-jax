"""
固定步长Langevin MCMC采样器

实现了用于量子蒙特卡洛波函数采样的Langevin动力学方法，
采用固定步长策略，适用于FermiNet电子结构计算。
"""

import jax
import jax.numpy as jnp
import jax.random as random


class FixedStepMCMC:
    """固定步长Langevin MCMC采样器"""

    def __init__(self, step_size=0.15, n_steps=10):
        """
        初始化固定步长MCMC采样器

        Args:
            step_size: Langevin动力学的时间步长 (通常0.1-0.2)
            n_steps: 每次sample调用执行的Langevin更新步数
        """
        self.step_size = step_size
        self.n_steps = n_steps

    def sample(self, log_psi_fn, r_elec, key, grad_log_psi_fn=None):
        """
        使用Langevin动力学进行Metropolis采样

        Args:
            log_psi_fn: 波函数对数函数，输入位置返回log|ψ(r)|，形状为 [batch]
            r_elec: 当前电子位置，形状 [batch, n_elec, 3]
            key: JAX随机数key
            grad_log_psi_fn: 可选的预计算梯度函数。如果提供，将使用它而不是重新计算。
                             输入 [batch, n_elec, 3]，输出 [batch, n_elec, 3]

        Returns:
            r_new: 新的电子位置，形状 [batch, n_elec, 3]
            accept_rate: 接受率，标量值
        """
        # 计算当前波函数对数值
        log_psi_current = log_psi_fn(r_elec)

        if grad_log_psi_fn is None:
            # 创建单样本的梯度函数（标量输出）
            def single_log_psi(r_single):
                """包装器，返回标量"""
                return log_psi_fn(r_single[jnp.newaxis, :, :])[0]

            grad_log_psi_single = jax.grad(single_log_psi)

            # 向量化梯度函数
            grad_log_psi_fn = jax.vmap(grad_log_psi_single)

        # 执行多步Langevin更新
        def step_fn(carry, _):
            r_curr, log_psi, key, acc = carry
            key, subkey = random.split(key)
            r_proposed, log_psi_proposed, mask = self._langevin_step(
                r_curr, log_psi, grad_log_psi_fn, log_psi_fn, subkey
            )
            acc = acc + mask.sum()
            return (r_proposed, log_psi_proposed, key, acc), None

        init_carry = (r_elec, log_psi_current, key, 0.0)
        (r_current, log_psi_val, key, accepted), _ = jax.lax.scan(
            step_fn, init_carry, length=self.n_steps
        )

        total = self.n_steps * r_elec.shape[0]

        # 计算接受率
        accept_rate = accepted / total

        return r_current, accept_rate

    def _langevin_step(self, r_elec, log_psi_val, grad_log_psi_fn, batch_log_psi_fn, key):
        """
        执行单步Langevin动力学更新

        Langevin动力学公式:
        r' = r + 0.5 * ∇log ψ(r) * dt + N(0, dt)

        Args:
            r_elec: 当前电子位置 [batch, n_elec, 3]
            log_psi_val: 当前log|ψ(r)|值 [batch]
            grad_log_psi_fn: 对数波函数梯度函数（单样本）
            batch_log_psi_fn: 对数波函数函数（批量，形状 [batch, n_elec, 3] -> [batch]）
            key: JAX随机数key

        Returns:
            r_new: 更新后的位置
            log_psi_new: 新的log|ψ(r)|值
            accepted: 接受掩码 [batch]
        """
        batch_size = r_elec.shape[0]

        # 1. 计算梯度: ∇log ψ(r)
        grad_log_psi = grad_log_psi_fn(r_elec)  # [batch, n_elec, 3]

        # 2. 计算漂移项: drift = 0.5 * ∇log ψ(r) * dt
        drift = 0.5 * grad_log_psi * self.step_size

        # 3. 生成高斯噪声: N(0, dt)
        key, noise_key = random.split(key)
        noise = random.normal(noise_key, shape=r_elec.shape) * jnp.sqrt(self.step_size)

        # 4. 生成提议位置
        r_proposed = r_elec + drift + noise

        # 5. 计算提议位置的波函数值
        log_psi_proposed = batch_log_psi_fn(r_proposed)

        # 6. Metropolis接受/拒绝
        key, accept_key = random.split(key)
        accepted = self._metropolis_accept(
            log_psi_val,
            log_psi_proposed,
            accept_key
        )

        # 7. 根据接受掩码选择位置
        r_new = jnp.where(
            accepted[:, jnp.newaxis, jnp.newaxis],
            r_proposed,
            r_elec
        )

        # 8. 根据接受掩码选择波函数值
        log_psi_new = jnp.where(accepted, log_psi_proposed, log_psi_val)

        return r_new, log_psi_new, accepted

    def _metropolis_accept(self, log_psi_current, log_psi_proposed, key):
        """
        Metropolis接受/拒绝判断

        接受概率:
        accept_prob = min(1, |ψ(r')|² / |ψ(r)|²)
                   = min(1, exp(2 * (log ψ(r') - log ψ(r))))

        Args:
            log_psi_current: 当前log|ψ(r)|值 [batch]
            log_psi_proposed: 提议log|ψ(r')|值 [batch]
            key: JAX随机数key

        Returns:
            accepted: 接受掩码 [batch]
        """
        # 计算log接受比率
        log_accept_ratio = 2.0 * (log_psi_proposed - log_psi_current)

        # 限制log_accept_ratio避免数值溢出
        log_accept_ratio = jnp.clip(log_accept_ratio, -100, 100)

        # 生成均匀随机数
        u = random.uniform(key, shape=log_psi_current.shape)

        # 接受条件: u < exp(log_accept_ratio)
        accepted = u < jnp.exp(log_accept_ratio)

        return accepted

    def _gaussian_proposal(self, r_elec, key):
        """
        生成纯高斯提议（用于调试或对比）

        Args:
            r_elec: 当前电子位置 [batch, n_elec, 3]
            key: JAX随机数key

        Returns:
            r_proposed: 提议位置
        """
        noise = random.normal(key, shape=r_elec.shape) * jnp.sqrt(self.step_size)
        r_proposed = r_elec + noise
        return r_proposed

    def warmup(self, log_psi_fn, r_elec, key, n_warmup_steps=100):
        """
        预热阶段，快速进行多步MCMC以达到平衡分布

        Args:
            log_psi_fn: 波函数对数函数
            r_elec: 初始电子位置
            key: JAX随机数key
            n_warmup_steps: 预热步数

        Returns:
            r_warmed: 预热后的电子位置
            key: 更新后的随机数key
        """
        r_current = r_elec

        for _ in range(n_warmup_steps):
            key, subkey = random.split(key)
            r_current, _ = self.sample(log_psi_fn, r_current, subkey)

        return r_current, key


# 测试和示例函数
def _test_mcmc():
    """简单的测试函数"""
    import numpy as np

    # 创建测试波函数：三维高斯分布
    def test_log_psi(r):
        """简单的测试波函数：log|ψ(r)| = -0.5 * sum(r²)"""
        log_psi = -0.5 * jnp.sum(r ** 2, axis=(1, 2))
        return log_psi

    # 初始化MCMC采样器
    mcmc = FixedStepMCMC(step_size=0.1, n_steps=10)

    # 初始化随机数生成器
    key = random.PRNGKey(42)
    key, init_key = random.split(key)

    # 初始电子位置 [batch=4, n_elec=2, dim=3]
    batch_size = 4
    n_elec = 2
    r_elec = random.normal(init_key, shape=(batch_size, n_elec, 3))

    print("初始位置:")
    print(np.array(r_elec))

    # 执行采样
    key, sample_key = random.split(key)
    r_new, accept_rate = mcmc.sample(test_log_psi, r_elec, sample_key)

    print(f"\n接受率: {accept_rate:.3f}")
    print("新位置:")
    print(np.array(r_new))

    # 测试预热
    key, warmup_key = random.split(key)
    r_warmed, key = mcmc.warmup(test_log_psi, r_elec, warmup_key, n_warmup_steps=20)

    print("\n预热后的位置:")
    print(np.array(r_warmed))

    print("\n测试通过!")


if __name__ == "__main__":
    _test_mcmc()
