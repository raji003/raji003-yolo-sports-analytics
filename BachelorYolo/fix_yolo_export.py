import os
import shutil
from urllib.parse import unquote
from pathlib import Path

export_dir = Path("C:/Users/abrah/Downloads/BachelorYolo/Project25YoloExport")
images_source = Path("C:/Users/abrah/Downloads/BachelorYolo/PreLabelImages")

labels_dir = export_dir / "labels"
images_out_dir = export_dir / "images"

# ─────────────────────────────────────────────
# STEP 1: Discover label files (flat or split)
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Discovering label files")
print("=" * 60)

# Collect (split_name, label_path) tuples.
# Support both flat labels/ and labels/train|val|test/ layouts.
label_entries = []  # list of (split, Path)

for item in sorted(labels_dir.iterdir()):
    if item.is_dir():
        split = item.name  # e.g. "train", "val", "test"
        for f in sorted(item.glob("*.txt")):
            label_entries.append((split, f))
    elif item.suffix == ".txt":
        label_entries.append((".", item))  # flat (no split)

print(f"Total label files found : {len(label_entries)}")
splits_found = sorted(set(s for s, _ in label_entries))
print(f"Splits found            : {splits_found}")

# ─────────────────────────────────────────────
# STEP 2: Show sample before renaming
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Sample label filenames BEFORE decoding (first 10)")
print("=" * 60)
for split, p in label_entries[:10]:
    print(f"  [{split}]  {p.name}")

# ─────────────────────────────────────────────
# STEP 3: Rename URL-encoded label files
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Renaming URL-encoded label files")
print("=" * 60)

renamed_count = 0
skipped_count = 0
rename_log = []

updated_entries = []  # rebuild with new paths after rename

for split, old_path in label_entries:
    decoded_name = unquote(old_path.name)
    if decoded_name != old_path.name:
        new_path = old_path.parent / decoded_name
        if new_path.exists() and new_path != old_path:
            print(f"  SKIP (target exists): {old_path.name} -> {decoded_name}")
            skipped_count += 1
            updated_entries.append((split, new_path))
        else:
            old_path.rename(new_path)
            rename_log.append((old_path.name, decoded_name))
            renamed_count += 1
            updated_entries.append((split, new_path))
    else:
        updated_entries.append((split, old_path))

print(f"Renamed : {renamed_count}")
print(f"Skipped (already existed): {skipped_count}")
print(f"Already clean (no change): {len(label_entries) - renamed_count - skipped_count}")

if rename_log:
    print()
    print("Sample renames (first 10):")
    for old, new in rename_log[:10]:
        print(f"  {old!r:40s}  ->  {new!r}")

# ─────────────────────────────────────────────
# STEP 4: Sample filenames AFTER decoding
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Sample label filenames AFTER decoding (first 10)")
print("=" * 60)
for split, p in updated_entries[:10]:
    print(f"  [{split}]  {p.name}")

# ─────────────────────────────────────────────
# STEP 5: Build image index from PreLabelImages (recursive)
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Indexing source images from PreLabelImages")
print("=" * 60)

image_extensions = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
image_index = {}  # stem -> Path

for img_path in images_source.rglob("*"):
    if img_path.is_file() and img_path.suffix.lower() in image_extensions:
        stem = img_path.stem  # e.g. "{4372}_0001"
        if stem in image_index:
            print(f"  WARNING: duplicate stem '{stem}': {image_index[stem]} vs {img_path}")
        else:
            image_index[stem] = img_path

print(f"Total source images indexed: {len(image_index)}")
print("Sample image stems (first 10):")
for stem in sorted(image_index.keys())[:10]:
    print(f"  {stem!r}  ->  {image_index[stem]}")

# ─────────────────────────────────────────────
# STEP 6: Copy matched images into images/ subfolder
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: Copying matched images into export images/ folder")
print("=" * 60)

copied_count = 0
already_exists_count = 0
no_match_labels = []   # labels with no matching image
copied_stems = set()

for split, label_path in updated_entries:
    stem = label_path.stem  # e.g. "{4372}_0001"
    if stem in image_index:
        src = image_index[stem]
        if split == ".":
            dest_dir = images_out_dir
        else:
            dest_dir = images_out_dir / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            already_exists_count += 1
        else:
            shutil.copy2(src, dest)
            copied_count += 1
        copied_stems.add(stem)
    else:
        no_match_labels.append((split, label_path.name, stem))

print(f"Images copied           : {copied_count}")
print(f"Already existed (skipped): {already_exists_count}")
print(f"Labels with NO match    : {len(no_match_labels)}")

# ─────────────────────────────────────────────
# STEP 7: Images with no label
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 7: Images with no corresponding label")
print("=" * 60)

label_stems = {p.stem for _, p in updated_entries}
no_label_images = [(stem, path) for stem, path in image_index.items() if stem not in label_stems]

print(f"Images with NO label: {len(no_label_images)}")
if no_label_images:
    print("First 20:")
    for stem, path in sorted(no_label_images)[:20]:
        print(f"  {stem!r}  ->  {path}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total label files           : {len(updated_entries)}")
print(f"  - URL-decoded (renamed)   : {renamed_count}")
print(f"  - Already clean           : {len(label_entries) - renamed_count - skipped_count}")
print(f"Total source images indexed : {len(image_index)}")
print(f"Images copied to export     : {copied_count}")
print(f"Images already existed      : {already_exists_count}")
print(f"Labels with NO image match  : {len(no_match_labels)}")
print(f"Images with NO label        : {len(no_label_images)}")

if no_match_labels:
    print()
    print("Labels with NO matching image (all):")
    for split, fname, stem in sorted(no_match_labels):
        print(f"  [{split}]  {fname}  (looking for stem: {stem!r})")
