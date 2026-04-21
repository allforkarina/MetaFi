from __future__ import annotations

"""Transformer-style attention and decoder for MM-Fi pose regression.

The module consumes Shared CNN features with shape [B, 512, 17, 12], treats the
512 channel dimension as the token axis, and regresses final 2D keypoint
coordinates with shape [B, 17, 2].
"""

import math

import torch
from torch import Tensor, nn, softmax


class AveragedHeadSelfAttention(nn.Module):
    """Single-layer self-attention with attention-matrix averaging across heads."""

    def __init__(self, embed_dim: int = 204, num_heads: int = 3) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Multi-head for different attention patterns, finding the best feature combinations for pose regression.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Attention Q, K, V projections.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm = nn.InstanceNorm1d(embed_dim, affine=True)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape                                                         # batch_size, tokens (512), feature_dim (204)
        return x.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)    # divide the feature_dim -> heads, dim | shape: batch_size, num_heads, 512, head_dim

    def _compute_averaged_attention(self, q: Tensor, k: Tensor) -> Tensor:
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)               # matrix multiplication, scaled by sqrt of dim
        attn_per_head = torch.softmax(attn_logits, dim=-1)                                          # softmax at the dimension of tokens (512)
        return attn_per_head.mean(dim=1)                                                            # average pool

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input with shape [B, N, {self.embed_dim}], got {tuple(x.shape)}"
            )

        q = self._reshape_heads(self.q_proj(x))                                                     # Init q,k,v as the shape of input
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        averaged_attention = self._compute_averaged_attention(q, k)                                 # Q dot K with scale for attention

        attended_values = torch.einsum("bij,bhjd->bhid", averaged_attention, v)                     # sum up the value and average pool
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()                          # change the head the token channel dim
        attended_values = attended_values.view(x.shape[0], x.shape[1], self.embed_dim)              # concat head and dim back to original shape

        output = x + attended_values                                                                # residual connection
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)
        return output


class PoseDecoder(nn.Module):
    """Map transformer features to 2D coordinate channels."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1)                         # channel: 512 -> 32
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)                           # channel: 32 -> 2, keypoints

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (512, 17, 12):
            raise ValueError(f"Expected input with shape [B, 512, 17, 12], got {tuple(x.shape)}")

        x = self.conv1(x)   # 512 -> 32
        x = self.relu(x)    # 32 -> 32
        x = self.conv2(x)   # 32 -> 2
        return x


class TransformerDecoderModule(nn.Module):
    """Convert Shared CNN features into final [B, 17, 2] keypoint predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 512
        self.embedding_dim = 17 * 12

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embedding_dim))  # 1, 512, 204
        self.attention = AveragedHeadSelfAttention(
            embed_dim=self.embedding_dim,                                                   # embedding dim = 204
            num_heads=3,                                                                    # channel of each head = 204 / 3 = 68
        )                                           
        self.decoder = PoseDecoder()                                                        # regression to keypoints

    @staticmethod
    def _flatten_spatial(x: Tensor) -> Tensor:
        return x.flatten(start_dim=2)

    @staticmethod
    def _restore_spatial(x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], 512, 17, 12)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (512, 17, 12):
            raise ValueError(f"Expected input with shape [B, 512, 17, 12], got {tuple(x.shape)}")

        x = self._flatten_spatial(x)            # B, 512, 17, 12 -> B, 512, 204
        x = x + self.pos_embed                  # add learnable positional embedding (Init is )
        x = self.attention(x)                   # self-attention with averaged attention across heads, shape unchanged
        x = self._restore_spatial(x)            # flatten the embedding dim, from 204 to 17, 12
        x = self.decoder(x)                     # regression to keypoints, from 512 -> 32 -> 2
        x = x.mean(dim=-1)                      # average pool over the spatial dimensions
        x = x.transpose(1, 2)
        return x
