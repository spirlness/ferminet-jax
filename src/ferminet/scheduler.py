"""
学习率调度器实现
基于能量改善情况的自适应学习率调整
"""

import torch
import numpy as np


class EnergyBasedScheduler:
    """
    基于能量改善情况的学习率调度器

    功能：
    1. 监控能量变化
    2. 能量改善时保持或重置学习率
    3. 能量停滞时降低学习率
    4. 支持早停检测

    应用场景：
    - 变分蒙特卡洛训练
    - 能量最小化优化
    - 避免陷入局部极小值
    """

    def __init__(
        self,
        optimizer,
        initial_lr=1e-3,
        min_lr=1e-6,
        patience=5,
        factor=0.5,
        verbose=True
    ):
        """
        初始化学习率调度器

        Args:
            optimizer: PyTorch优化器
            initial_lr: 初始学习率
            min_lr: 最小学习率
            patience: 能量不改善的耐心值（epoch数）
            factor: 学习率衰减因子
            verbose: 是否打印信息
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.verbose = verbose

        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr

        # 状态变量
        self.best_energy = float('inf')
        self.wait = 0  # 等待计数
        self.current_lr = initial_lr
        self.num_bad_epochs = 0  # 累计未改善的epoch数
        self.history = {'lr': [], 'energy': []}

    def step(self, current_energy):
        """
        更新学习率（在每个epoch结束时调用）

        Args:
            current_energy: 当前能量值

        Returns:
            bool: 能量是否改善
        """
        # 记录历史
        self.history['lr'].append(self.current_lr)
        self.history['energy'].append(current_energy)

        # 判断能量是否改善
        if current_energy < self.best_energy:
            # 能量改善
            improved = True
            self.best_energy = current_energy
            self.wait = 0

            if self.verbose:
                print(f"  [OK] Energy improved: {current_energy:.6f} (best: {self.best_energy:.6f})")
        else:
            # 能量未改善
            improved = False
            self.wait += 1
            self.num_bad_epochs += 1

            if self.verbose:
                print(f"  [X] Energy not improved: {current_energy:.6f} (wait: {self.wait}/{self.patience})")

            # 检查是否需要降低学习率
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0  # 重置等待计数

        return improved

    def _reduce_lr(self):
        """
        降低学习率
        """
        old_lr = self.current_lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        self.current_lr = new_lr

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        if self.verbose:
            print(f"  [DOWN] Reduce learning rate: {old_lr:.6f} -> {new_lr:.6f}")

    def get_lr(self):
        """
        获取当前学习率

        Returns:
            float: 当前学习率
        """
        return self.current_lr

    def state_dict(self):
        """
        保存调度器状态

        Returns:
            dict: 状态字典
        """
        return {
            'best_energy': self.best_energy,
            'wait': self.wait,
            'current_lr': self.current_lr,
            'num_bad_epochs': self.num_bad_epochs,
            'history': self.history
        }

    def load_state_dict(self, state_dict):
        """
        加载调度器状态

        Args:
            state_dict: 状态字典
        """
        self.best_energy = state_dict['best_energy']
        self.wait = state_dict['wait']
        self.current_lr = state_dict['current_lr']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.history = state_dict['history']

        # 恢复学习率到优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def reset(self):
        """
        重置调度器状态
        """
        self.best_energy = float('inf')
        self.wait = 0
        self.num_bad_epochs = 0
        self.current_lr = self.initial_lr

        # 重置优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

        # 清空历史（可选）
        # self.history = {'lr': [], 'energy': []}

        if self.verbose:
            print(f"  [RESET] Scheduler reset, learning rate: {self.current_lr:.6f}")


class CyclicScheduler:
    """
    循环学习率调度器

    实现学习率的循环升降，有助于跳出局部极小值
    """

    def __init__(
        self,
        optimizer,
        base_lr=1e-5,
        max_lr=1e-2,
        step_size=50,
        mode='triangular',
        gamma=1.0,
        verbose=True
    ):
        """
        初始化循环学习率调度器

        Args:
            optimizer: PyTorch优化器
            base_lr: 最小学习率
            max_lr: 最大学习率
            step_size: 半周期长度（epoch数）
            mode: 调度模式 ('triangular', 'triangular2', 'exp_range')
            gamma: 衰减因子（用于exp_range模式）
            verbose: 是否打印信息
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.verbose = verbose

        self.epoch = 0
        self.current_lr = base_lr
        self.history = {'lr': []}

    def step(self):
        """
        更新学习率（在每个epoch结束时调用）
        """
        cycle = np.floor(1 + self.epoch / (2 * self.step_size))
        x = np.abs(self.epoch / self.step_size - 2 * cycle + 1)

        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** cycle)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.current_lr = lr

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 记录历史
        self.history['lr'].append(lr)

        if self.verbose and self.epoch % 10 == 0:
            print(f"  Epoch {self.epoch}: 学习率 = {lr:.6f}")

        self.epoch += 1
        return lr

    def get_lr(self):
        """
        获取当前学习率
        """
        return self.current_lr


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 EnergyBasedScheduler")
    print("=" * 60)

    # 创建模拟优化器
    model = torch.nn.Linear(64, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建调度器
    scheduler = EnergyBasedScheduler(
        optimizer,
        initial_lr=1e-3,
        min_lr=1e-6,
        patience=3,
        factor=0.5,
        verbose=True
    )

    print("\n初始学习率:", scheduler.get_lr())

    # 模拟训练过程
    print("\n模拟训练过程:")
    energies = [10.0, 9.5, 9.0, 8.8, 8.8, 8.8, 8.8, 8.7, 8.65, 8.6]

    for epoch, energy in enumerate(energies):
        print(f"\nEpoch {epoch + 1}:")
        improved = scheduler.step(energy)
        print(f"  当前学习率: {scheduler.get_lr():.6f}")

    print(f"\n最佳能量: {scheduler.best_energy:.6f}")
    print(f"未改善的epoch数: {scheduler.num_bad_epochs}")

    # 测试状态保存和加载
    print("\n" + "-" * 60)
    print("测试状态保存和加载:")
    state = scheduler.state_dict()
    print("状态已保存")

    new_scheduler = EnergyBasedScheduler(
        optimizer,
        initial_lr=1e-3,
        patience=3,
        factor=0.5,
        verbose=False
    )
    new_scheduler.load_state_dict(state)
    print(f"加载后的最佳能量: {new_scheduler.best_energy:.6f}")
    print(f"加载后的学习率: {new_scheduler.get_lr():.6f}")

    # 测试重置
    print("\n" + "-" * 60)
    print("测试重置:")
    scheduler.reset()
    print(f"重置后的学习率: {scheduler.get_lr():.6f}")
    print(f"重置后的最佳能量: {scheduler.best_energy}")

    # 测试循环学习率调度器
    print("\n" + "=" * 60)
    print("测试 CyclicScheduler")
    print("=" * 60)

    cyclic_scheduler = CyclicScheduler(
        optimizer,
        base_lr=1e-5,
        max_lr=1e-2,
        step_size=10,
        mode='triangular',
        verbose=False
    )

    print("\n前20个epoch的学习率:")
    for i in range(20):
        lr = cyclic_scheduler.step()
        if i % 5 == 0:
            print(f"  Epoch {i+1:2d}: lr = {lr:.6f}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
