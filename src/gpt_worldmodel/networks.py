"""Networks for GPT-WorldModel."""

from beartype import beartype
from einops import rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float, jaxtyped
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import Tensor, nn
from torch.nn import functional as tf


class MultiHeadAttention(nn.Module):
    """
    4D+ Extension of `torch.nn.MultiHeadAttention`.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input (after the first linear layer)
    num_heads : int
        The number of heads in the MHA models (must divide d_model)
    causal : bool
        Whether to use causal attention.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(2, 10, 16, 16, 64)
    >>> MultiHeadAttention(d_model=64, num_heads=8).forward(x).shape
    torch.Size([2, 10, 16, 16, 64])
    """

    def __init__(self, d_model: int, num_heads: int, causal: bool) -> None:  # noqa: FBT001
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=True)
        self.causal = causal
        self.pre_attn_trans = Rearrange(
            pattern="b l ... (num_heads d_model) -> b ... num_heads l d_model",
            num_heads=num_heads,
        )
        self.post_attn_trans = Rearrange("b ... num_heads l d -> b l ... (num_heads d)")
        self.pos_emb = RotaryEmbedding(dim=32, freqs_for="pixel", max_freq=256)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L * D"]) -> Float[Tensor, "B L * D"]:
        """
        Compute multi-head attention.

        Parameters
        ----------
        x : Float[Tensor, "B L * D"]
            The input tensor.

        Returns
        -------
        Float[Tensor, "B L * D"]
            The output after performing attention.
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        freqs = self.pos_emb.get_axial_freqs(*x.shape[1:-1])
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        q = self.pre_attn_trans(q)
        k = self.pre_attn_trans(k)
        v = self.pre_attn_trans(v)

        out = tf.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.fc.forward(self.post_attn_trans(out))


class Block(nn.Module):
    """
    Spatio-Temporal Attention Block.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    d_model : int
        Model dimension.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(2, 16, 100, 64)
    >>> block = STBlock(num_heads=8, d_model=64)
    >>> block(x).shape
    torch.Size([2, 16, 100, 64])
    """

    def __init__(self, num_heads: int, d_model: int) -> None:
        super().__init__()
        self.spatial_block = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            causal=False,
        )
        self.temporal_block = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            causal=True,
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.GELU(),
            nn.Linear(in_features=d_model * 4, out_features=d_model),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L H W D"]) -> Float[Tensor, "B L H W D"]:
        """
        順伝播.

        Parameters
        ----------
        x : Float[Tensor, "B L H W D"]
            Input Tensor.

        Returns
        -------
        Float[Tensor, "B L H W D"]
            Output Tensor.
        """
        _, seq_len, h, w, _ = x.shape
        x_sc = rearrange(x, "b l h w c -> (b l) (h w) c")
        x_sc = self.spatial_block(self.spatial_norm(x_sc)).add(x_sc)
        x_tc = rearrange(x_sc, "(b l) hw c -> (b hw) l c", l=seq_len)
        x_tc = self.temporal_block(x_tc).add(x_tc)
        x_tsc = self.mlp(self.temporal_norm(x_tc)).add(x_tc)
        return rearrange(x_tsc, "(b h w) l c -> b l h w c", h=h, w=w)  # type: ignore[no-any-return]
