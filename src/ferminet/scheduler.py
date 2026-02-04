"""
Learning rate scheduler implementation
"""

class EnergyBasedScheduler:
    """
    Learning rate scheduler that adjusts learning rate based on energy error.
    Lowers learning rate when energy approaches target for fine-tuning.
    """

    def __init__(
        self,
        initial_lr=0.001,
        target_energy=-1.174,
        patience=10,
        decay_factor=0.5,
        min_lr=1e-5,
    ):
        """
        Initialize energy scheduler
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.target_energy = target_energy
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_lr = min_lr

        self.best_energy = float("inf")
        self.wait_count = 0
        self.epoch_count = 0

    def step(self, current_energy):
        """
        Update learning rate
        """
        self.epoch_count += 1

        # Check if energy improved
        if current_energy < self.best_energy:
            self.best_energy = current_energy
            self.wait_count = 0
        else:
            self.wait_count += 1

        # Decay learning rate if patience exceeded
        if self.wait_count >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            self.wait_count = 0
            return self.current_lr, True, old_lr  # Return (new_lr, decayed, old_lr)

        return self.current_lr, False, None

    def get_lr(self):
        """Get current learning rate"""
        return self.current_lr

    def get_info(self):
        """Get scheduler info"""
        return {
            "initial_lr": self.initial_lr,
            "current_lr": self.current_lr,
            "target_energy": self.target_energy,
            "best_energy": self.best_energy,
            "wait_count": self.wait_count,
            "epoch_count": self.epoch_count,
        }
