# Step 1: Copy the exported CVAT dataset and normalize image formats.
# YOLO training requires RGB images — RGBA (transparent PNGs from CVAT) must be converted.

import shutil
from pathlib import Path
from PIL import Image

SRC = Path("Project25YoloExport")
DST = Path("Project25YoloReady")

# Always start fresh so re-runs don't accumulate stale files
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

converted = 0
for img_path in (DST / "images").glob("*.png"):
    img = Image.open(img_path)
    if img.mode == "RGBA":
        img.convert("RGB").save(img_path)
        converted += 1

print(f"Copied dataset to {DST}")
print(f"Converted {converted} RGBA images to RGB")
