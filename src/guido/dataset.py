import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

COMMAND_MAP    = {"forward": 0, "left": 1, "right": 2}
COMMAND_MIRROR = {0: 0, 1: 2, 2: 1}
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# Native image is 200×300. Round each dimension to nearest multiple of 16
# so ViT-B/16 patch division is exact without squashing the aspect ratio.
# 200 → 192 (12 patches), 300 → 304 (19 patches) → 228 patch tokens total.
DINO_H, DINO_W = 192, 304
PATCH_SIZE = 16
GRID_H = DINO_H // PATCH_SIZE   # 12
GRID_W = DINO_W // PATCH_SIZE   # 19


def _sorted_pkl_files(directory: str) -> list[str]:
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return [os.path.join(directory, f) for f in files]


def _encode_history(history: np.ndarray) -> np.ndarray:
    """(21,3) [x,y,heading] → (21,4) [x, y, sin(h), cos(h)]"""
    xy, h = history[:, :2], history[:, 2]
    return np.stack([xy[:, 0], xy[:, 1], np.sin(h), np.cos(h)], axis=1).astype(np.float32)


def _mirror(camera, history, future, command):
    camera  = T.functional.horizontal_flip(camera)
    h       = history.copy(); h[:, 0] *= -1; h[:, 2] *= -1
    f       = None if future is None else (future.copy().__setitem__(slice(None, None), future.copy()) or future.copy())
    if future is not None:
        f = future.copy(); f[:, 0] *= -1
    return camera, h, f, COMMAND_MIRROR[command]


class DrivingDataset(Dataset):
    """
    Loads nuPlan .pkl files.

    Images are resized to (DINO_H, DINO_W) = (192, 304) — nearest multiples
    of 16 to the native 200×300, preserving aspect ratio for the ViT.

    Augmentations (training only):
      mirror_p         : prob of horizontal flip + trajectory mirror + cmd swap
      hist_noise_std   : σ of Gaussian noise on history (x, y) in metres
      hist_dropout_p   : prob of zeroing out the last 1-5 history steps
      random_crop_scale: (min, max) crop scale for RandomResizedCrop; None = off
    """

    def __init__(
        self,
        file_list: list[str],
        *,
        augment: bool         = False,
        test: bool            = False,
        mirror_p: float       = 0.0,
        hist_noise_std: float = 0.0,
        hist_dropout_p: float = 0.0,
        mirror_warmup: int    = 10,
        load_aux: bool        = False,
    ):
        self.samples         = file_list
        self.test            = test
        self.augment         = augment
        self.mirror_p        = mirror_p
        self.hist_noise_std  = hist_noise_std
        self.hist_dropout_p  = hist_dropout_p
        self.mirror_warmup   = mirror_warmup
        self.load_aux        = load_aux
        self._eff_mirror_p   = 0.0

        base = [
            T.Resize((DINO_H, DINO_W), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        aug = [
            T.Resize((DINO_H, DINO_W), antialias=True),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            T.RandomGrayscale(p=0.05),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = T.Compose(aug if augment else base)

    def set_epoch(self, epoch: int) -> None:
        ramp = min(1.0, epoch / max(self.mirror_warmup, 1))
        self._eff_mirror_p = self.mirror_p * ramp

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        with open(self.samples[idx], "rb") as f:
            data = pickle.load(f)

        command = COMMAND_MAP[data["driving_command"]]
        history = _encode_history(data["sdc_history_feature"])
        future  = None if self.test else data["sdc_future_feature"].astype(np.float32)

        camera = torch.from_numpy(data["camera"]).permute(2, 0, 1)
        camera = self.transform(camera)

        if self.augment:
            if self.mirror_p > 0 and torch.rand(1).item() < self._eff_mirror_p:
                camera, history, future, command = _mirror(camera, history, future, command)
            if self.hist_noise_std > 0:
                history[:, :2] += np.random.normal(0, self.hist_noise_std, (21, 2)).astype(np.float32)
            if self.hist_dropout_p > 0 and torch.rand(1).item() < self.hist_dropout_p:
                # Zero out last 1–5 history steps to force robustness to incomplete history
                n_drop = torch.randint(1, 6, (1,)).item()
                history[-n_drop:, :] = 0.0

        sample = {
            "camera":  camera,
            "history": torch.from_numpy(history),
            "command": torch.tensor(command, dtype=torch.long),
        }
        if future is not None:
            sample["future"] = torch.from_numpy(future)

        if self.load_aux:
            if "depth" in data:
                raw_depth = np.array(data["depth"], dtype=np.float32).squeeze()
                depth_m   = (255.0 - raw_depth) / 255.0 * 100.0
                sample["depth"] = torch.from_numpy(depth_m).unsqueeze(0)  # (1, H, W)

            if "semantic_label" in data:
                seg = np.array(data["semantic_label"])
                if seg.ndim == 3:
                    seg = seg.squeeze(-1)
                sample["semantic_label"] = torch.from_numpy(seg.astype(np.int64))

        return sample


def make_datasets(
    data_dir: str,
    mirror_p: float       = 0.0,
    hist_noise_std: float = 0.0,
    hist_dropout_p: float = 0.0,
    mirror_warmup: int    = 10,
    load_aux: bool        = False,
):
    train_ds = DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "train")),
        augment=True, mirror_p=mirror_p,
        hist_noise_std=hist_noise_std, hist_dropout_p=hist_dropout_p,
        mirror_warmup=mirror_warmup, load_aux=load_aux,
    )
    val_ds = DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "val")),
        augment=False, load_aux=load_aux,
    )
    return train_ds, val_ds


def make_test_dataset(data_dir: str) -> "DrivingDataset":
    return DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, "test_public")),
        augment=False, test=True,
    )