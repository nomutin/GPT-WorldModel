"""Modal-agnostic DataModule."""

import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
import wget
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, StackDataset
from tqdm import tqdm

Transform: TypeAlias = Callable[[Tensor], Tensor]


def split_path_list(path_list: list[Path], train_ratio: float = 0.8) -> tuple[list[Path], list[Path]]:
    """
    Split the path list into train and test.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    train_ratio : float
        Ratio of train data.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Train and test path list.
    """
    train_len = int(len(path_list) * train_ratio)
    return path_list[:train_len], path_list[train_len:]


def load_tensor(path: Path) -> Tensor:
    """
    Load tensor from file(.npy, .pt).

    Parameters
    ----------
    path : Path
        File path.

    Returns
    -------
    Tensor
        Loaded tensor.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    if path.suffix == ".npy":
        return Tensor(np.load(path))
    if path.suffix == ".pt" and isinstance(tensor := torch.load(path, weights_only=False), Tensor):
        return tensor
    msg = f"Unknown file extension: {path.suffix}"
    raise ValueError(msg)


class EpisodeDataset(Dataset[Tensor]):
    """
    Dataset for single modality data.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    transform : Transform
        Transform function.
    """

    def __init__(self, path_list: list[Path], transform: Transform) -> None:
        super().__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of data.

        Returns
        -------
        int
            Number of data(Len of path_list).
        """
        return len(self.path_list)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Get the data at the index and apply the transform.

        Parameters
        ----------
        idx : int
            Index of the data.

        Returns
        -------
        Tensor
            Transformed data.
        """
        return self.transform(load_tensor(self.path_list[idx]))


@dataclass
class DataConfig:
    """Single modal(action, observation, ...) data configuration."""

    prefix: str
    preprocess: Transform
    train_input_transform: Transform
    train_target_transform: Transform
    val_input_transform: Transform
    val_target_transform: Transform
    predict_input_transform: Transform
    predict_target_transform: Transform


@dataclass
class DataModuleConfig:
    """Configuration for EpisodeDataModule."""

    data_name: str
    processed_data_name: str
    batch_size: int
    num_workers: int
    gdrive_id: str
    train_ratio: float
    data_defs: tuple[DataConfig, ...]

    @property
    def data_dir(self) -> Path:
        """Path to the data directory."""
        return Path("data") / self.data_name

    @property
    def processed_data_dir(self) -> Path:
        """Path to the processed data directory."""
        return Path("data") / self.processed_data_name

    def load_from_gdrive(self) -> None:
        """Download data from Google Drive."""
        url = f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={self.gdrive_id}"
        filename = Path("tmp.tar.gz")
        wget.download(url, str(filename))
        with tarfile.open(filename, "r:gz") as f:
            f.extractall(path=Path("data"), filter="data")
        Path(filename).unlink(missing_ok=False)


class EpisodeDataModule(LightningDataModule):
    """
    Modal-Agnostic DataModule.

    Train/Val/Test dataloaders yields [modal1(input), modal1(target), modal2(input), modal2(target), ...].
    """

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        """Save processed data to `{data_name}_processed_episode` directory."""
        if not self.config.data_dir.exists():
            self.config.load_from_gdrive()

        if self.config.processed_data_dir.exists():
            return

        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        for data_type in self.config.data_defs:
            for path in tqdm(sorted(self.config.data_dir.glob(data_type.prefix))):
                tensor = data_type.preprocess(load_tensor(path))
                new_path = self.config.processed_data_dir / f"{path.stem}.pt"
                torch.save(tensor.detach().clone(), new_path)

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        """Create datasets."""
        train_dataset_list, val_dataset_list, predict_dataset_list = [], [], []
        for data_type in self.config.data_defs:
            path_list = sorted(self.config.processed_data_dir.glob(data_type.prefix))
            train_path_list, val_path_list = split_path_list(path_list)
            train_dataset_list.extend([
                EpisodeDataset(train_path_list, data_type.train_input_transform),
                EpisodeDataset(train_path_list, data_type.train_target_transform),
            ])
            val_dataset_list.extend([
                EpisodeDataset(val_path_list, data_type.val_input_transform),
                EpisodeDataset(val_path_list, data_type.val_target_transform),
            ])

            predict_path_list = []
            for train_path, val_path in zip(train_path_list, val_path_list, strict=False):
                predict_path_list.extend([train_path, val_path])
            predict_dataset_list.extend([
                EpisodeDataset(predict_path_list, data_type.predict_input_transform),
                EpisodeDataset(predict_path_list, data_type.predict_target_transform),
            ])

        self.train_dataset = StackDataset(*train_dataset_list)
        self.val_dataset = StackDataset(*val_dataset_list)
        self.predict_dataset = StackDataset(*predict_dataset_list)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        TrainDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        ValidationDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader. Shuffle is False.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def predict_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        PredictionDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader. Shuffle is False.
        """
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )
