import pytest
import torch

from metatrain.pet.modules.transformer import AttentionBlock


TOTAL_DIM = 8
NUM_REAL_TOKENS = 3


# small attention block with random weights
def _block():
    torch.manual_seed(0)

    block = AttentionBlock(total_dim=TOTAL_DIM, num_heads=2, temperature=1.0)
    block.eval()

    return block


# real tokens followed by one padded slot
def _tokens_with_padded_slot():
    torch.manual_seed(1)

    tokens = torch.randn(1, NUM_REAL_TOKENS + 1, TOTAL_DIM)
    padded = tokens[0, NUM_REAL_TOKENS]

    # make a huge magnitude of the padded slot
    tokens[0, NUM_REAL_TOKENS] = 1000.0 * padded / padded.norm()

    # exclude the padded slot for every query
    factors = torch.ones(1, NUM_REAL_TOKENS + 1, NUM_REAL_TOKENS + 1)
    factors[:, :, NUM_REAL_TOKENS] = 0.0

    return tokens, factors


@pytest.mark.parametrize("use_manual_attention", [True, False])
def test_zero_factor_keys_are_excluded(use_manual_attention):
    # check if padded token affect real tokens even though its cutoff factor is zero

    block = _block()
    tokens, factors = _tokens_with_padded_slot()

    output = block(tokens, factors, use_manual_attention)

    # reference = the block with removed padded token entirely
    real = slice(0, NUM_REAL_TOKENS)
    reference = block(tokens[:, real], factors[:, real, real], use_manual_attention)

    # the padded slot must not change the output of any real token
    assert torch.allclose(output[:, real], reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_manual_attention", [True, False])
def test_zero_factor_keys_have_zero_gradient(use_manual_attention):
    # check if we differentiate through the mask without invalid gradients

    block = _block()
    tokens, factors = _tokens_with_padded_slot()

    # we differentiate with respect to cuttof factors
    factors.requires_grad_(True)

    # forces flow through the cutoff factors so zero factors must not give nan
    output = block(tokens, factors, use_manual_attention)
    output.sum().backward()

    assert torch.isfinite(factors.grad).all()

    # the excluded key receives no gradient
    assert torch.all(factors.grad[:, :, NUM_REAL_TOKENS] == 0.0)
