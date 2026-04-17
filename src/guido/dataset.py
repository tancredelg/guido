import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

# ── Constants ──────────────────────────────────────────────────────────────────

COMMAND_MAP = {"forward": 0, "left": 1, "right": 2}

# DINOv2 was trained with standard ImageNet normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv3 patch size is 16 → input must be a multiple of 16.
# 256 × 256 is the recommended default per the DINOv3 README transform.
DINO_INPUT_SIZE = 256


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sorted_pkl_files(directory: str) -> list[str]:
    """Return .pkl paths in a directory sorted numerically by filename stem."""
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return [os.path.join(directory, f) for f in files]


def _encode_history(history: np.ndarray) -> torch.Tensor:
    """
    Convert raw (21, 3) history [x, y, heading_rad] into (21, 4)
    [x, y, sin(heading), cos(heading)].

    Representing heading as (sin, cos) avoids the discontinuity at ±π and
    keeps all features in a comparable numerical range.
    """
    xy = history[:, :2]  # (21, 2)
    heading = history[:, 2]  # (21,)
    encoded = np.stack([xy[:, 0], xy[:, 1], np.sin(heading), np.cos(heading)], axis=1)  # (21, 4)
    return torch.from_numpy(encoded).float()


# ── Dataset ────────────────────────────────────────────────────────────────────


class DrivingDataset(Dataset):
    """
    Loads pre-processed nuPlan pickle files and returns dicts ready for the
    DrivingPlanner model.

    Each sample dict contains:
        camera   : FloatTensor (3, 224, 224) – normalised, DINOv2-ready
        history  : FloatTensor (21, 4)       – [x, y, sin(h), cos(h)]
        command  : LongTensor  ()            – 0=forward, 1=left, 2=right
        future   : FloatTensor (60, 3)       – [x, y, heading] (train/val only)

    Args:
        file_list : list of absolute paths to .pkl files.
        augment   : if True, apply colour jitter to the camera image.
                    Should only be True for the training split.
        test      : if True, skip loading 'future' (unavailable in test set).
    """

    def __init__(self, file_list: list[str], augment: bool = False, test: bool = False):
        self.samples = file_list
        self.test = test

        # Colour jitter is safe because it does not alter the trajectory.
        # We deliberately avoid geometric augmentations (flip, crop) since they
        # would require mirroring/offsetting the trajectory labels.
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        with open(self.samples[idx], "rb") as f:
            data = pickle.load(f)

        # Camera: (H, W, 3) uint8 → (3, H, W) then resize + normalise
        camera = torch.from_numpy(data["camera"]).permute(2, 0, 1)  # (3, 200, 300)
        camera = self.transform(camera)  # (3, 224, 224)

        history = _encode_history(data["sdc_history_feature"])  # (21, 4)
        command = torch.tensor(COMMAND_MAP[data["driving_command"]], dtype=torch.long)

        sample = {"camera": camera, "history": history, "command": command}

        if not self.test:
            sample["future"] = torch.from_numpy(data["sdc_future_feature"]).float()  # (60, 3)

        return sample


# ── Convenience constructors ───────────────────────────────────────────────────


def make_datasets(data_dir: str) -> tuple[DrivingDataset, DrivingDataset]:
    """Return (train_dataset, val_dataset) from a root data directory."""
    train_ds = DrivingDataset(_sorted_pkl_files(os.path.join(data_dir, "train")), augment=True)
    val_ds = DrivingDataset(_sorted_pkl_files(os.path.join(data_dir, "val")), augment=False)
    return train_ds, val_ds


def make_test_dataset(data_dir: str) -> DrivingDataset:
    """Return the test dataset (no future labels)."""
    return DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "test_public")),
        augment=False,
        test=True,
    )
