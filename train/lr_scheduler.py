# train/lr_scheduler.py
import numpy as np
from typing import Sequence


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """
    Build a per-optimizer-step learning rate schedule with:
      1) Linear warmup: start_warmup_value -> base_value over (warmup_epochs * niter_per_ep) steps
      2) Cosine decay: base_value -> final_value over remaining steps

    Args:
        base_value: Peak (post-warmup) learning rate.
        final_value: Final learning rate at the end of training.
        epochs: Total number of epochs.
        niter_per_ep: Number of optimizer steps per epoch (after gradient accumulation).
        warmup_epochs: Number of warmup epochs.
        start_warmup_value: Starting LR at the first step (often 0 or very small).

    Returns:
        schedule: 1D numpy array of length epochs * niter_per_ep with per-step LR values.
    """
    total_steps = epochs * niter_per_ep
    warmup_steps = int(warmup_epochs * niter_per_ep)

    # Warmup phase (linear interpolation)
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps, dtype=np.float32)
    else:
        warmup_schedule = np.array([], dtype=np.float32)

    # Cosine decay phase
    remaining = total_steps - warmup_steps
    if remaining > 0:
        iters = np.arange(remaining, dtype=np.float32)
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1.0 + np.cos(np.pi * iters / remaining)
        )
    else:
        cosine_schedule = np.array([], dtype=np.float32)

    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    assert len(schedule) == total_steps, f"Schedule length mismatch: {len(schedule)} vs {total_steps}"
    return schedule


class StepLRSchedule:
    """
    Lightweight container for a precomputed LR schedule.

    Typical usage (manual control in training loop or Lightning's optimizer_step):
        sched = StepLRSchedule(schedule=cosine_scheduler(...))
        lr = sched.get(step)

    This class does NOT modify any optimizer by itself; you must apply the value.
    """
    def __init__(self, schedule: Sequence[float]):
        self._schedule = np.array(schedule, dtype=np.float32)

    def get(self, step: int) -> float:
        """
        Get LR for a given (0-based) step index. If step exceeds length, return last value.
        """
        if step < 0:
            raise ValueError("step must be non-negative")
        if step >= len(self._schedule):
            return float(self._schedule[-1])
        return float(self._schedule[step])

    @property
    def values(self) -> np.ndarray:
        """Return the full schedule array."""
        return self._schedule

    def __len__(self):
        return len(self._schedule)


# Optional helper for integration hints
def build_cosine_with_warmup(
    lr: float,
    lr_end: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    lr_start: float = 0.0,
) -> StepLRSchedule:
    """
    Convenience wrapper returning StepLRSchedule.
    Mirrors the signature often used with argument parsers.

    Args:
        lr: Peak LR after warmup (base_value).
        lr_end: Final LR at last step.
        epochs: Total epochs.
        steps_per_epoch: Optimizer steps per epoch (after gradient accumulation).
        warmup_epochs: Warmup epochs.
        lr_start: Starting LR at first step.

    Returns:
        StepLRSchedule instance.
    """
    sched = cosine_scheduler(
        base_value=lr,
        final_value=lr_end,
        epochs=epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_start,
    )
    return StepLRSchedule(sched)


# Example (for documentation/testing):
if __name__ == "__main__":
    epochs = 5
    steps_per_epoch = 10
    sched = build_cosine_with_warmup(
        lr=1e-3,
        lr_end=1e-5,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=1,
        lr_start=0.0,
    )
    print("Total steps:", len(sched))
    print("First 12 values:", sched.values[:12])
    print("Last 5 values:", sched.values[-5:])