"""
Quick data exploration script for Phase 2 dataset.
Run with: uv run notebooks/explore_phase2_data.py

Prints shapes, dtypes, and value ranges for all keys in a few sample pkl files.
Answers: image resolution, depth range (metric?), seg label count and encoding.
"""
import pickle
import numpy as np
import os

DATA_DIR = os.environ.get("DATA_DIR", "data")

for split in ("train", "val", "test_public"):
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[skip] {split_dir} not found")
        continue

    files = sorted(
        [f for f in os.listdir(split_dir) if f.endswith(".pkl")],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    if not files:
        print(f"[skip] no pkl files in {split_dir}")
        continue

    print(f"\n{'='*60}")
    print(f"Split: {split}  ({len(files)} samples)")
    print(f"{'='*60}")

    path = os.path.join(split_dir, files[0])
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"  keys: {sorted(data.keys())}")
    for key, val in sorted(data.items()):
        arr = np.array(val)
        if not np.issubdtype(arr.dtype, np.number):
            print(f"  {key:30s} value={val!r}")
            continue
        print(f"  {key:30s} shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        if key == "semantic_label":
            unique = np.unique(arr)
            print(f"  {'':30s} unique classes ({len(unique)}): {unique}")
    # ── Depth decoding investigation ────────────────────────────────────────
    # Raw depth is uint8 in range ~180-254. This is almost certainly encoded.
    # Common schemes for nuPlan/CARLA synthetic depth:
    #   1. metric = (255 - value) * scale          (inverted)
    #   2. metric = value / 255 * max_range        (linear)
    #   3. metric = exp(value / k)                 (log-encoded)
    #   4. metric = (value - 128) * scale          (signed)
    #
    # We check against a CARLA-style encoding used in several AV datasets:
    #   depth_m = (R + G*256 + B*65536) / (256^3 - 1) * 1000
    # But since we only have 1 channel (not RGB), it's likely simpler.
    # The notebook uses raw values directly in L1 loss without decoding,
    # suggesting the course intends raw uint8 as the target.
    # We plot both raw and a few candidate decodings to check which is metric.

    with open(os.path.join(split_dir, files[0]), "rb") as f:
        d0 = pickle.load(f)
    raw = np.array(d0["depth"], dtype=np.float32).squeeze()   # (H, W)

    print(f"\n  Depth decoding candidates (sample 0):")
    print(f"    raw uint8:            min={raw.min():.1f}  max={raw.max():.1f}  mean={raw.mean():.1f}")
    cands = {
        "linear /255*100m":     raw / 255.0 * 100.0,
        "(255-raw)/255*100m":   (255 - raw) / 255.0 * 100.0,
        "raw/10 (0-25m)":       raw / 10.0,
        "(raw-128)/10":         (raw - 128.0) / 10.0,
        "exp(raw/50)":          np.exp(raw / 50.0),
    }
    for name, decoded in cands.items():
        print(f"    {name:30s}  min={decoded.min():.2f}  max={decoded.max():.2f}  mean={decoded.mean():.2f}")

    # Check if the nuPlan convention applies: depth stored as distance in cm → /100 for metres
    # (nuPlan synthetic uses 0=invalid, values in cm, clipped at 25000cm = 250m)
    print(f"\n  If nuPlan cm convention: min={raw.min()/100:.2f}m  max={raw.max()/100:.2f}m")
    print(f"  If CARLA 8bit: depth_m = (value/255)*far, far=1000m → "
          f"min={raw.min()/255*1000:.1f}m  max={raw.max()/255*1000:.1f}m")

    # Consistency check over 50 samples
    print(f"\n  Shape consistency (50 samples):")
    shapes = {}
    for fname in files[:50]:
        with open(os.path.join(split_dir, fname), "rb") as f:
            d = pickle.load(f)
        for key, val in d.items():
            shapes.setdefault(key, set()).add(np.array(val).shape)
    for key, s_set in sorted(shapes.items()):
        tag = "✓" if len(s_set) == 1 else "✗ INCONSISTENT"
        print(f"  {key:30s} {s_set}  {tag}")