"""
Data Augmentation/Transformation.

Every methods are implemented as a class to be used with `LightningCLI`.
"""

from pathlib import Path

import torch
import wget
from einops import pack, rearrange, repeat
from numpy.random import MT19937, Generator
from torch import Tensor


class RandomWindow:
    """Randomly slice sequence data."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.randgen = Generator(MT19937(42))

    def __call__(self, data: Tensor) -> Tensor:
        """
        Select start idx with `randgen2` and slice data.

        Parameters
        ----------
        data : Tensor
            Sequence data to be sliced. Shape: [seq_len, *].

        Returns
        -------
        Tensor
            Sliced data. Shape: [window_size, *].
        """
        seq_len = data.shape[0]
        start_idx = self.randgen.integers(0, seq_len - self.window_size)
        return data[start_idx : start_idx + self.window_size]


class NormalizeAction:
    """
    Normalize 3D+ tensor with given max and min values.

    Parameters
    ----------
    max_array : list[float]
        Max values of the tensor along the last dim.
    min_array : list[float]
        Min values of the tensor along the last dim.
    """

    def __init__(self, max_array: list[float], min_array: list[float]) -> None:
        self.max_array = Tensor(max_array)
        self.min_array = Tensor(min_array)

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply normalization.

        Parameters
        ----------
        data : Tensor
            Data to be normalized. Shape: [batch*, dim].

        Returns
        -------
        Tensor
            Normalized data.
        """
        copy_data = data.detach().clone()
        copy_data -= self.min_array
        copy_data *= 1.0 / (self.max_array - self.min_array)
        copy_data *= 2.0
        copy_data += -1.0
        return copy_data


class PackAction:
    """Temporal packing of action tensor."""

    def __init__(self, size: int, step: int, padding: int) -> None:
        self.size = size
        self.step = step
        self.padding = padding

    def __call__(self, data: Tensor) -> Tensor:
        """
        Pack action tensor.

        Parameters
        ----------
        data : Tensor
            Action tensor. [L, D]

        Returns
        -------
        Tensor
            Packed action tensor. [L, D * size]
        """
        first_pad = repeat(data[0], "d -> padding d", padding=self.padding)
        last_pad = repeat(data[-1], "d -> padding d", padding=self.padding)
        data, *_ = pack([first_pad, data, last_pad], "* d")
        data = data.unfold(dimension=0, size=self.size, step=self.step)
        return rearrange(data, "l d folded -> l (d folded)")


class Tokenizer:
    """Encode video tensor into tokens."""

    def __init__(self, model_name: str) -> None:
        local_dir = Path(".model_cache") / model_name
        encoder_path = local_dir / "encoder.jit"
        if not encoder_path.exists():
            local_dir.mkdir(exist_ok=True, parents=True)
            url = f"https://huggingface.co/nvidia/{model_name}/resolve/main/encoder.jit"
            wget.download(url, str(encoder_path))
        self.encoder = torch.jit.load(encoder_path).eval()  # type: ignore[no-untyped-call]

    def __call__(self, video: Tensor) -> Tensor:
        """
        Tokenize video.

        Parameters
        ----------
        video : Tensor
            Video data. Shape: [B* C H W].

        Returns
        -------
        Tensor
            Codes. Shape: [B* H' W'].
        """
        self.encoder.to("cuda")
        video = rearrange(video, "t c h w -> 1 c t h w")
        video = video.to("cuda")
        indices, *_ = self.encoder(video)
        return indices.squeeze(0).to("cpu")  # type: ignore[no-any-return]


class RemoveHead:
    """Remove the first element of the tensor."""

    def __init__(self, n: int = 1) -> None:
        self.n = n

    def __call__(self, data: Tensor) -> Tensor:
        """
        Remove the first element of the tensor.

        Parameters
        ----------
        data : Tensor
            Data to be processed. Shape: [seq_len, *].

        Returns
        -------
        Tensor
            Data with removed dimension.
            Shape: [seq_len-1, *].
        """
        return data[self.n:]


class RemoveTail:
    """Remove the last element of the tensor."""

    def __init__(self, n: int = 1) -> None:
        self.n = n

    def __call__(self, data: Tensor) -> Tensor:
        """
        Remove the last element of the tensor.

        Parameters
        ----------
        data : Tensor
            Data to be processed. Shape: [seq_len, *].

        Returns
        -------
        Tensor
            Data with removed dimension.
            Shape: [seq_len-1, *].
        """
        return data[:-self.n]
