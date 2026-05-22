"""
Quick exploration of Phase 3 real data vs synthetic.
Run: uv run notebooks/explore_phase3_data.py --data-dir data/
"""
import os, pickle, sys, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
args = parser.parse_args()

for split, dirname in [("synthetic train", "train"), ("real val", "val_real"),
                        ("real test", "test_public_real")]:
    d = os.path.join(args.data_dir, dirname)
    if not os.path.isdir(d):
        print(f"[skip] {d} not found"); continue
    files = [f for f in os.listdir(d) if f.endswith(".pkl")]
    print(f"\n{'='*55}")
    print(f"{split}: {len(files)} samples  ({d})")
    with open(os.path.join(d, files[0]), "rb") as f:
        data = pickle.load(f)
    print(f"  keys: {sorted(data.keys())}")
    cam = np.array(data["camera"])
    print(f"  camera: shape={cam.shape}  dtype={cam.dtype}  "
          f"min={cam.min()}  max={cam.max()}  mean={cam.mean():.1f}")
    hist = np.array(data["sdc_history_feature"])
    print(f"  history: shape={hist.shape}  range [{hist.min():.3f}, {hist.max():.3f}]")
    if "sdc_future_feature" in data:
        fut = np.array(data["sdc_future_feature"])
        print(f"  future:  shape={fut.shape}  range [{fut.min():.3f}, {fut.max():.3f}]")
    if "driving_command" in data:
        print(f"  command: {data['driving_command']!r}")
    # Check a few more samples for command diversity
    cmds = set()
    for fn in files[:min(200, len(files))]:
        with open(os.path.join(d, fn), "rb") as f:
            d2 = pickle.load(f)
        if "driving_command" in d2:
            # print(f"  {fn}: {d2['driving_command']!r}")
            try:
                cmds.add(d2["driving_command"])
            except Exception as e:
                print(f"    [error adding driving_command to set] {e}")
    print(f"  commands seen in first 200: {cmds}")