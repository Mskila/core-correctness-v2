from __future__ import annotations

import torch

from model_core.sampling import ConstrainedSampler


def _sampler() -> ConstrainedSampler:
    sampler = ConstrainedSampler(
        vocab_size=9,
        feat_offset=3,
        arity_map={3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 1},
        positive_only_ids={6},
    )
    sampler.infected_propagating_ids = {7}
    sampler.sign_restore_ids = {8}
    return sampler


def test_batched_mask_is_exactly_equal_to_scalar_reference() -> None:
    sampler = _sampler()
    logits = torch.arange(45, dtype=torch.float32).reshape(5, 9)
    depths = [1, 2, 3, 1, 4]
    previous = [None, 0, 3, 7, 8]
    infections = [0, 1, 2, 3, 4]

    actual = sampler.apply_mask_to_logits(
        logits,
        depths,
        step_idx=3,
        total_steps=8,
        prev_tokens=previous,
        infected_chain_lens=infections,
    )
    expected = logits.clone()
    for batch_index, depth in enumerate(depths):
        valid = sampler.valid_mask(
            depth,
            3,
            8,
            logits.device,
            prev_token=previous[batch_index],
            infected_chain_len=infections[batch_index],
        )
        expected[batch_index][~valid] = -1e9

    assert torch.equal(actual, expected)
    assert torch.equal(logits, torch.arange(45, dtype=torch.float32).reshape(5, 9))


def test_advance_batch_state_matches_scalar_updates() -> None:
    sampler = _sampler()
    tokens = torch.tensor([0, 4, 6, 7, 8], dtype=torch.long)
    depths = [1, 2, 3, 4, 5]
    previous: list[int | None] = [None] * 5
    infections = [0, 1, 2, 1, 4]
    expected_depths = list(depths)
    expected_previous = list(previous)
    expected_infections = list(infections)
    for index, token in enumerate(tokens.tolist()):
        expected_depths[index] += sampler.delta[token]
        expected_previous[index] = token
        expected_infections[index] = sampler.update_infection(
            token, expected_infections[index]
        )

    sampler.advance_batch_state(tokens, depths, previous, infections)

    assert depths == expected_depths
    assert previous == expected_previous
    assert infections == expected_infections
