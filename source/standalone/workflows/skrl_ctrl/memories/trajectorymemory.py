from __future__ import annotations

from typing import Literal

import torch

from skrl.memories.torch import Memory


class TrajectoryRandomMemory(Memory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = True,
        traj_len: int = 1,
    ) -> None:
        """Random trajectory sampling memory.

        Samples contiguous trajectories of length ``traj_len`` and returns
        tensors with shape ``(batch_size, traj_len, data_dim)``.

        :param replacement: Whether to sample starting points with replacement.
        """
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )
        self._replacement = replacement
        self.traj_len = traj_len

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
    ) -> list[list[torch.Tensor]]:
        """Sample random trajectories.

        :param names: Tensor names to sample.
        :param batch_size: Number of trajectories.
        :param mini_batches: Number of mini-batches.
        :param traj_len: Length of each trajectory.

        :return: A list of mini-batches. Each mini-batch is a list of tensors
                 aligned with ``names`` and shaped:
                 ``(batch_size, traj_len, data_dim)``.
        """
        if self.traj_len < 1:
            raise ValueError("traj_len must be >= 1")

        # total valid flat transitions
        total_size = len(self)

        # valid starting positions (avoid crossing env boundaries)
        # flat index = time * num_envs + env
        max_start = total_size - (self.traj_len - 1) * self.num_envs
        if max_start <= 0:
            raise ValueError("Not enough data to sample trajectories")

        if self._replacement:
            start_indexes = torch.randint(0, max_start, (batch_size,))
        else:
            start_indexes = torch.randperm(max_start, dtype=torch.long)[:batch_size]

        # offsets for a single trajectory
        offsets = torch.arange(
            0,
            self.traj_len * self.num_envs,
            self.num_envs,
            device=start_indexes.device,
        )

        # (batch_size, traj_len) -> flat indexes
        traj_indexes = start_indexes[:, None] + offsets[None, :]
        self.sampling_indexes = traj_indexes

        # split into mini-batches
        if mini_batches > 1:
            batch_splits = torch.chunk(traj_indexes, mini_batches, dim=0)
        else:
            batch_splits = [traj_indexes]

        out = []
        for split in batch_splits:
            tensors_mb = []
            for name in names:
                if name not in self.tensors:
                    tensors_mb.append(None)
                    continue

                data = self.tensors_view[name][split.reshape(-1)]
                data = data.view(split.shape[0], self.traj_len, *data.shape[1:])
                tensors_mb.append(data)

            out.append(tensors_mb)

        return out
