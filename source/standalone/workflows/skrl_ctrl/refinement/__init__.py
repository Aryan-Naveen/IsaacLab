"""Few-shot / offline refinement helpers (collection + offline training hooks)."""

from .collection import collect_until_all_envs_done_one_episode
from .offline import offline_train_steps
from .orchestrator import RefinementOrchestrator
from .plotting import plot_reference_vs_samples
from .recording import record_transition
from .types import RolloutResult

__all__ = [
    "RolloutResult",
    "collect_until_all_envs_done_one_episode",
    "offline_train_steps",
    "plot_reference_vs_samples",
    "record_transition",
    "RefinementOrchestrator",
]
