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
            # Colour/lighting variation — real scenes have much wider distribution
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.05),
            # Camera blur — common in real dashcam footage
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
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

        # driving_command: synthetic = string ('forward'/'left'/'right')
        #                  real      = one-hot array [0,1,0,0] or absent
        raw_cmd = data.get("driving_command", "forward")
        if isinstance(raw_cmd, str):
            command = COMMAND_MAP.get(raw_cmd, 0)
        else:
            arr = np.asarray(raw_cmd)
            command = int(arr.argmax()) if arr.ndim > 0 else 0
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


def make_datasets_mixed(
    data_dir: str,
    real_train_frac: float = 1.0,   # fraction of val_real to include in training
    mirror_p: float       = 0.2,
    hist_noise_std: float = 0.01,
    hist_dropout_p: float = 0.1,
    mirror_warmup: int    = 5,
    load_aux: bool        = False,
):
    """
    Phase 3: mix synthetic train + real val_real for training.
    Validates on the held-out portion of val_real.

    real_train_frac: fraction of val_real used for training (rest = val).
    With 1000 real samples and frac=0.7: 700 real train + 300 real val.
    """
    import random as _random
    synth_files = _sorted_pkl_files(os.path.join(data_dir, "train"))
    real_dir    = os.path.join(data_dir, "val_real")
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(
            f"Real data not found at {real_dir}. "
            "Run notebooks/download_phase3_data.sh first."
        )
    real_files = _sorted_pkl_files(real_dir)
    n_real_train = int(len(real_files) * real_train_frac)
    real_train   = real_files[:n_real_train]
    real_val     = real_files[n_real_train:]

    train_files = synth_files + real_train
    # Shuffle so real and synthetic samples are interleaved
    _random.shuffle(train_files)

    train_ds = DrivingDataset(
        train_files,
        augment=True, mirror_p=mirror_p,
        hist_noise_std=hist_noise_std, hist_dropout_p=hist_dropout_p,
        mirror_warmup=mirror_warmup, load_aux=load_aux,
    )
    # If no real val samples remain, fall back to synthetic val
    val_files = real_val if real_val else _sorted_pkl_files(os.path.join(data_dir, "val"))
    val_ds = DrivingDataset(val_files, augment=False, load_aux=False)

    return train_ds, val_ds


def make_test_dataset(data_dir: str, real: bool = False) -> "DrivingDataset":
    subdir = "test_public_real" if real else "test_public"
    return DrivingDataset(
        _sorted_pkl_files(os.path.join(data_dir, subdir)),
        augment=False, test=True,
    )