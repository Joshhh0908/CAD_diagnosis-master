import os
import shutil
import re

# run from your CAD_diagnosis-master directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# find all .pth files in the base directory
pth_files = [f for f in os.listdir(base_dir) if f.endswith('.pth')]

for fname in pth_files:
    # extract run name — everything before _epoch or _best
    match = re.match(r'^(model_[^_]+(?:_\d+x\d+x\d+)?)_(epoch\d+|best)\.pth$', fname)
    if not match:
        print(f"Skipping unrecognised filename: {fname}")
        continue

    run_name = match.group(1)   # e.g. model_46x40x10 or model_58x40x8
    folder   = os.path.join(base_dir, run_name)
    os.makedirs(folder, exist_ok=True)

    src = os.path.join(base_dir, fname)
    dst = os.path.join(folder,   fname)
    shutil.move(src, dst)
    print(f"Moved {fname} -> {run_name}/")

print("\nDone.")