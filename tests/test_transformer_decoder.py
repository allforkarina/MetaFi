from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from models.transformer_decoder import (
    AveragedHeadSelfAttention,
    PoseDecoder,
    TransformerDecoderModule,
)


def test_attention_block_preserves_sequence_shape() -> None:
    block = AveragedHeadSelfAttention(embed_dim=204, num_heads=3)
    x = torch.randn(2, 512, 204)

    y = block(x)

    assert y.shape == (2, 512, 204)


def test_attention_block_uses_instance_norm() -> None:
    block = AveragedHeadSelfAttention(embed_dim=204, num_heads=3)

    assert isinstance(block.norm, nn.InstanceNorm1d)
    assert block.norm.num_features == 204


def test_attention_block_does_not_use_standard_multihead_attention() -> None:
    block = AveragedHeadSelfAttention(embed_dim=204, num_heads=3)

    assert not any(isinstance(module, nn.MultiheadAttention) for module in block.modules())


def test_attention_block_averages_attention_matrices_to_token_attention() -> None:
    block = AveragedHeadSelfAttention(embed_dim=204, num_heads=3)
    x = torch.randn(2, 512, 204)

    q = block._reshape_heads(block.q_proj(x))
    k = block._reshape_heads(block.k_proj(x))
    averaged_attention = block._compute_averaged_attention(q, k)

    assert averaged_attention.shape == (2, 512, 512)


def test_decoder_output_shape() -> None:
    decoder = PoseDecoder()
    x = torch.randn(2, 512, 17, 12)

    y = decoder(x)

    assert y.shape == (2, 2, 17, 12)


def test_transformer_decoder_output_shape() -> None:
    module = TransformerDecoderModule()
    x = torch.randn(2, 512, 17, 12)

    y = module(x)

    assert y.shape == (2, 17, 2)


def test_transformer_decoder_flattens_to_expected_shape() -> None:
    module = TransformerDecoderModule()
    x = torch.randn(2, 512, 17, 12)

    flattened = module._flatten_spatial(x)

    assert flattened.shape == (2, 512, 204)


def test_transformer_decoder_positional_embedding_shape() -> None:
    module = TransformerDecoderModule()

    assert tuple(module.pos_embed.shape) == (1, 512, 204)


def test_transformer_decoder_restores_spatial_shape() -> None:
    module = TransformerDecoderModule()
    x = torch.randn(2, 512, 17, 12)

    restored = module._restore_spatial(module.attention(module._flatten_spatial(x)))

    assert restored.shape == (2, 512, 17, 12)


def test_transformer_decoder_rejects_wrong_input_shape() -> None:
    module = TransformerDecoderModule()
    x = torch.randn(2, 256, 17, 12)

    with pytest.raises(ValueError, match=r"\[B, 512, 17, 12\]"):
        module(x)
