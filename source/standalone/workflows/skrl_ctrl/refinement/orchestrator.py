from __future__ import annotations

from typing import Any


class RefinementOrchestrator:
    """Skeleton for repeated collect-B → offline-X rounds.

    Subclass or replace methods to plug in your metrics, checkpointing, and plotting.
    """

    def __init__(self, env: Any, agent: Any, memory: Any) -> None:
        self.env = env
        self.agent = agent
        self.memory = memory

    def run_round(self) -> None:
        """TODO: clear/refill memory, call collection then offline_train_steps."""
        raise NotImplementedError

    def run(self, num_rounds: int) -> None:
        for _ in range(num_rounds):
            self.run_round()
