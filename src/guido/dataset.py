import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

COMMAND_MAP = {"forward": 0, "left": 1, "right": 2}
COMMAND_MIRROR = {0: 0, 1: 2, 2: 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_INPUT_SIZE = 256


def _sorted_pkl_files(directory: str) -> list[str]:
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return [os.path.join(directory, f) for f in files]


def _encode_history(history: np.ndarray) -> np.ndarray:
    """(21,3) [x,y,heading] → (21,4) [x, y, sin(h), cos(h)]"""
    xy, h = history[:, :2], history[:, 2]
    return np.stack([xy[:, 0], xy[:, 1], np.sin(h), np.cos(h)], axis=1).astype(np.float32)


def _mirror(camera, history, future, command):
    """Horizontal flip + x-axis mirror of trajectories + left↔right command."""
    camera = T.functional.horizontal_flip(camera)
    h = history.copy()
    h[:, 0] *= -1
    h[:, 2] *= -1
    f = None if future is None else (lambda c: (c.__setitem__(slice(None), c.copy()), c)[1])(future.copy())
    if future is not None:
        f = future.copy()
        f[:, 0] *= -1
    return camera, h, f, COMMAND_MIRROR[command]


class DrivingDataset(Dataset):
    """
    Loads nuPlan .pkl files.

    Augmentations (training only, all off by default = baseline behaviour):
      mirror_p       : prob of horizontal flip + trajectory mirror + cmd swap.
      hist_noise_std : σ of Gaussian noise added to history (x, y) in metres.

    Both default to 0.0 so the baseline config needs no changes.
    set_epoch(epoch) ramps mirror_p linearly over warmup_epochs to avoid
    confusing an untrained model.
    """

    def __init__(
        self,
        file_list: list[str],
        *,
        augment: bool = False,
        test: bool = False,
        mirror_p: float = 0.0,
        hist_noise_std: float = 0.0,
        mirror_warmup: int = 10,
    ):
        self.samples = file_list
        self.test = test
        self.augment = augment
        self.mirror_p = mirror_p
        self.hist_noise_std = hist_noise_std
        self.mirror_warmup = mirror_warmup
        self._eff_mirror_p = 0.0  # updated by set_epoch()

        if augment:
            self.transform = T.Compose(
                [
                    T.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE), antialias=True),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE), antialias=True),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def set_epoch(self, epoch: int) -> None:
        """Ramp mirror probability 0 → mirror_p over mirror_warmup epochs."""
        ramp = min(1.0, epoch / max(self.mirror_warmup, 1))
        self._eff_mirror_p = self.mirror_p * ramp

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        with open(self.samples[idx], "rb") as f:
            data = pickle.load(f)

        command = COMMAND_MAP[data["driving_command"]]
        history = _encode_history(data["sdc_history_feature"])  # (21,4) np.float32
        future = None if self.test else data["sdc_future_feature"].astype(np.float32)

        camera = torch.from_numpy(data["camera"]).permute(2, 0, 1)  # (3,H,W) uint8
        camera = self.transform(camera)  # (3,256,256) float

        if self.augment:
            if self.mirror_p > 0 and torch.rand(1).item() < self._eff_mirror_p:
                camera, history, future, command = _mirror(camera, history, future, command)
            if self.hist_noise_std > 0:
                history[:, :2] += np.random.normal(0, self.hist_noise_std, (21, 2)).astype(np.float32)

        sample = {
            "camera": camera,
            "history": torch.from_numpy(history),
            "command": torch.tensor(command, dtype=torch.long),
        }
        if future is not None:
            sample["future"] = torch.from_numpy(future)
        return sample


def make_datasets(
    data_dir: str,
    mirror_p: float = 0.0,
    hist_noise_std: float = 0.0,
    mirror_warmup: int = 10,
):
    train_ds = DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "train")),
        augment=True,
        mirror_p=mirror_p,
        hist_noise_std=hist_noise_std,
        mirror_warmup=mirror_warmup,
    )
    val_ds = DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "val")),
        augment=False,
    )
    return train_ds, val_ds


def make_test_dataset(data_dir: str) -> DrivingDataset:
    return DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "test_public")),
        augment=False,
        test=True,
    )
