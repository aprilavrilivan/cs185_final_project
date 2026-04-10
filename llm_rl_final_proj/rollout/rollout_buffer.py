from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    N = batch.input_ids.shape[0]
    indices = torch.randperm(N, generator=generator, device=batch.input_ids.device) if shuffle else torch.arange(N, device=batch.input_ids.device)
    for start in range(0, N, minibatch_size):
        end = start + minibatch_size
        batch_indices = indices[start:end]
        minibatch = RolloutBatch(
            input_ids=batch.input_ids[batch_indices],
            attention_mask=batch.attention_mask[batch_indices],
            completion_mask=batch.completion_mask[batch_indices],
            old_logprobs=batch.old_logprobs[batch_indices],
            ref_logprobs=batch.ref_logprobs[batch_indices],
            rewards=batch.rewards[batch_indices],
            advantages=batch.advantages[batch_indices],
            task_names=[batch.task_names[i.item()] for i in batch_indices] if batch.task_names is not None else None,
            completion_texts=[batch.completion_texts[i.item()] for i in batch_indices] if batch.completion_texts is not None else None,
        )
        if device is not None:
            minibatch = minibatch.to(device)
        yield minibatch
