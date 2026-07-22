"""Grammar-constrained formula sampling, independent from the training engine."""
from __future__ import annotations

import torch

from .ops import OPS_CONFIG
from .vm import is_infected_propagating, is_sign_restoring


class ConstrainedSampler:
    def __init__(
        self,
        vocab_size: int,
        feat_offset: int,
        arity_map: dict[int, int],
        positive_only_ids: set[int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.feat_offset = feat_offset
        self.arity_map = arity_map
        self.delta: dict[int, int] = {}
        for token_id in range(vocab_size):
            if token_id < feat_offset:
                self.delta[token_id] = 1
            else:
                self.delta[token_id] = 1 - arity_map.get(token_id, 1)
        self.positive_only_ids = positive_only_ids or set()
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for index, config in enumerate(OPS_CONFIG):
            token_id = index + feat_offset
            if is_infected_propagating(config[0]):
                self.infected_propagating_ids.add(token_id)
            if is_sign_restoring(config[0]):
                self.sign_restore_ids.add(token_id)

    def valid_mask(
        self,
        stack_depth: int,
        step_idx: int,
        total_steps: int,
        device: torch.device,
        prev_token: int | None = None,
        infected_chain_len: int = 0,
    ) -> torch.Tensor:
        del prev_token
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for token_id in range(self.vocab_size):
            new_depth = stack_depth + self.delta[token_id]
            if new_depth < 1:
                mask[token_id] = False
                continue
            min_future = new_depth + (remaining - 1) * -2
            max_future = new_depth + (remaining - 1)
            if 1 < min_future or 1 > max_future:
                mask[token_id] = False
            if (
                infected_chain_len >= 2
                and token_id in self.infected_propagating_ids
            ):
                mask[token_id] = False
            if infected_chain_len >= 3 and (
                token_id in self.infected_propagating_ids
                or token_id in self.positive_only_ids
            ):
                mask[token_id] = False
        if not mask.any():
            for token_id in range(self.vocab_size):
                if stack_depth + self.delta[token_id] >= 1:
                    mask[token_id] = True
        return mask

    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        stack_depths: list[int],
        step_idx: int,
        total_steps: int,
        prev_tokens: list[int | None] | None = None,
        infected_chain_lens: list[int] | None = None,
    ) -> torch.Tensor:
        del prev_tokens
        if logits.ndim != 2 or logits.shape[1] != self.vocab_size:
            raise ValueError("logits must have shape [batch, vocab_size]")
        if len(stack_depths) != logits.shape[0]:
            raise ValueError("stack_depths must match logits batch size")
        if (
            infected_chain_lens is not None
            and len(infected_chain_lens) != logits.shape[0]
        ):
            raise ValueError("infected_chain_lens must match logits batch size")

        device = logits.device
        depths = torch.tensor(stack_depths, dtype=torch.int64, device=device)[:, None]
        deltas = torch.tensor(
            tuple(self.delta[token_id] for token_id in range(self.vocab_size)),
            dtype=torch.int64,
            device=device,
        )[None, :]
        new_depths = depths + deltas
        remaining = total_steps - step_idx
        valid = new_depths >= 1
        valid &= new_depths + (remaining - 1) * -2 <= 1
        valid &= new_depths + (remaining - 1) >= 1

        infections = torch.tensor(
            infected_chain_lens or [0] * logits.shape[0],
            dtype=torch.int64,
            device=device,
        )[:, None]
        propagating = torch.tensor(
            tuple(
                token_id in self.infected_propagating_ids
                for token_id in range(self.vocab_size)
            ),
            dtype=torch.bool,
            device=device,
        )[None, :]
        positive_only = torch.tensor(
            tuple(
                token_id in self.positive_only_ids
                for token_id in range(self.vocab_size)
            ),
            dtype=torch.bool,
            device=device,
        )[None, :]
        valid &= ~((infections >= 2) & propagating)
        valid &= ~((infections >= 3) & (propagating | positive_only))

        fallback_rows = ~valid.any(dim=1)
        fallback = new_depths >= 1
        valid = torch.where(fallback_rows[:, None], fallback, valid)
        return logits.masked_fill(~valid, -1e9)

    def advance_batch_state(
        self,
        tokens: torch.Tensor,
        stack_depths: list[int],
        prev_tokens: list[int | None],
        infected_chain_lens: list[int],
    ) -> None:
        """Advance Python grammar state after one bulk token transfer."""
        if tokens.ndim != 1:
            raise ValueError("tokens must have rank 1")
        batch_size = tokens.shape[0]
        if not (
            len(stack_depths)
            == len(prev_tokens)
            == len(infected_chain_lens)
            == batch_size
        ):
            raise ValueError("sampling state must match token batch size")
        for index, token in enumerate(tokens.detach().cpu().tolist()):
            stack_depths[index] += self.delta[token]
            prev_tokens[index] = token
            infected_chain_lens[index] = self.update_infection(
                token, infected_chain_lens[index]
            )

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        if token in self.sign_restore_ids:
            return 0
        if token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len


__all__ = ["ConstrainedSampler"]
