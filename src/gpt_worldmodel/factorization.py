"""
Methods for handling large voc. embeddings.

Modified from https://github.com/1x-technologies/1xgpt/blob/main/genie/factorization_utils.py.
"""

import math

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from numpy.random import MT19937, Generator
from torch import Tensor, nn


class FactorizedEmbedding(nn.Module):
    """
    Each token's embedding is the sum of the embeddings in each factorized vocabulary.

    Parameters
    ----------
    factored_vocab_size: int
        Size of each factored vocabulary
    num_factored_vocabs: int
        Number of factored vocabularies
    d_model: int
        Dimension of the embeddings
    mask_token_id: int
        Token id of the mask token
    """

    def __init__(
        self,
        factored_vocab_size: int,
        num_factored_vocabs: int,
        d_model: int,
        mask_token_id: int,
    ) -> None:
        super().__init__()
        self.factored_vocab_size = factored_vocab_size
        self.num_factored_vocabs = num_factored_vocabs
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.factored_embeds = nn.ParameterList(
            [nn.Embedding(factored_vocab_size, d_model) for _ in range(num_factored_vocabs)],
        )
        self.mask_token_embed = nn.Parameter(torch.zeros(1, d_model))

    @jaxtyped(typechecker=beartype)
    def forward(self, input_ids: Int[Tensor, "*"]) -> Float[Tensor, "* d_model"]:
        """
        Get embeddings for input token ids.

        Parameters
        ----------
        input_ids: Int[Tensor, "*"]
            Int tensor [0, factored_vocab_size ** num_factored_vocabs) with any shape.

        Returns
        -------
        Float[Tensor, "* d_model"]
            Factored embeddings.
        """
        input_ids = input_ids.long()
        embeds = self.mask_token_embed.repeat(*input_ids.size(), 1)
        is_not_mask = input_ids != self.mask_token_id

        factored_token_ids = factorize_token_ids(
            token_ids=input_ids[is_not_mask],
            num_factored_vocabs=self.num_factored_vocabs,
            factored_vocab_size=self.factored_vocab_size,
        )
        unmasked_embeds = [
            factored_embed(factored_token_ids)
            for factored_embed, factored_token_ids in zip(
                self.factored_embeds,
                factored_token_ids.unbind(-1),
                strict=True,
            )
        ]

        embeds[is_not_mask] = torch.sum(torch.stack(unmasked_embeds), dim=0)
        return embeds


def factorize_token_ids(
    token_ids: Tensor,
    num_factored_vocabs: int,
    factored_vocab_size: int,
) -> Tensor:
    """
    Factorize token ids into multiple vocabularies.

    Parameters
    ----------
    token_ids: Tensor
        any size tensor with token id values in [0, image_vocab_size = 2**18).
    num_factored_vocabs: int
        Number of factored vocabularies
    factored_vocab_size: int
        Size of each factored vocabulary

    Returns
    -------
    Tensor
        Size token_ids.size() + (num_factored_vocabs,), where the last dimension has token ids in
        each individual vocabulary, with values in [0, factored_vocab_size = 512)
    """
    powers = factored_vocab_size ** torch.arange(num_factored_vocabs, device=token_ids.device)
    return (token_ids.unsqueeze(-1) // powers) % factored_vocab_size  # type: ignore[no-any-return]


def unfactorize_token_ids(
    factored_token_ids: Tensor,
    num_factored_vocabs: int,
    factored_vocab_size: int,
) -> Tensor:
    """
    Inverse of `factorize_token_ids`.

    Parameters
    ----------
    factored_token_ids: Tensor
        Size token_ids.size() + (num_factored_vocabs,), with values in [0, factored_vocab_size = 512)
    num_factored_vocabs: int
        Number of factored vocabularies
    factored_vocab_size: int
        Size of each factored vocabulary

    Returns
    -------
    Tensor
        Size token_ids.size()[:-1], with values in [0, image_vocab_size = 2**18)
    """
    powers = factored_vocab_size ** torch.arange(num_factored_vocabs, device=factored_token_ids.device)
    return (factored_token_ids * powers).sum(dim=-1)  # type: ignore[no-any-return]


class MaskGitAugmentation:
    """
    Token masking from [1].

    References
    ----------
    [1] https://arxiv.org/abs/2202.04200
    """

    def __init__(
        self,
        num_factored_vocabs: int,
        factored_vocab_size: int,
        num_prompt_frames: int = 8,
        max_corrupt_rate: float = 0.2,
        non_mlm_ratio: float = 0.5,
    ) -> None:
        self.max_corrupt_rate = max_corrupt_rate
        self.num_factored_vocabs = num_factored_vocabs
        self.factored_vocab_size = factored_vocab_size
        self.num_prompt_frames = num_prompt_frames
        self.non_mlm_ratio = non_mlm_ratio
        self.randgen = Generator(MT19937(42))

    def __call__(self, indices: Tensor) -> Tensor:
        """
        Apply corruption to the input token.

        Parameters
        ----------
        indices : Tensor
            Input token. Shape: [seq_len, h, w].

        Returns
        -------
        Tensor
            Corrupted token. Shape: [L, H, w].
        """
        seq_len, height, width = indices.shape

        factorized_tokens = factorize_token_ids(
            token_ids=indices,
            num_factored_vocabs=self.num_factored_vocabs,
            factored_vocab_size=self.factored_vocab_size,
        )

        r = torch.rand(factorized_tokens.size(), device=indices.device)
        u01 = torch.rand((), device=indices.device)
        random_patches_mask = r < self.max_corrupt_rate * u01
        random_values = torch.randint(
            low=0,
            high=self.factored_vocab_size,
            size=factorized_tokens.size(),
            dtype=torch.long,
            device=indices.device,
        )
        factorized_tokens[random_patches_mask] = random_values[random_patches_mask]

        if self.randgen.random() < self.non_mlm_ratio:
            first_masked_frame = self.randgen.integers(self.num_prompt_frames, seq_len - 1)
            correct_rate = self.randgen.uniform(0.25, 1.0)
            for i in range(seq_len - first_masked_frame):
                correct_rate *= self.randgen.uniform(0.9, 1.0)
                r = torch.rand((height, width, self.num_factored_vocabs), device=indices.device)
                random_patches_mask = r > correct_rate
                factorized_tokens[first_masked_frame + i][random_patches_mask] = random_values[first_masked_frame + i][
                    random_patches_mask
                ]
        else:
            first_masked_frame = 1

        mask = torch.zeros(1)
        while mask.max() == 0:
            mask_prob = cosine_schedule(torch.rand(seq_len - first_masked_frame, 1, 1, device=indices.device))
            r = torch.rand_like(indices[first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob

        indices = unfactorize_token_ids(factorized_tokens, self.num_factored_vocabs, self.factored_vocab_size)
        indices[first_masked_frame:][mask] = self.factored_vocab_size**self.num_factored_vocabs
        return indices


def cosine_schedule(u: float | Tensor) -> float | Tensor:
    """
    Clamp u to [0, 1] and return cosine schedule.

    Parameters
    ----------
    u : float | Tensor
        Value between 0 and 1.

    Returns
    -------
    float | Tensor
        Cosine scheduled value.
    """
    if isinstance(u, torch.Tensor):
        return torch.cos(u * torch.pi / 2)
    return math.cos(u * math.pi / 2)
