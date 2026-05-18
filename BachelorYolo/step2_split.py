# Step 2: Split the dataset into train / val / test by game ID.
#
# Splitting by game (not by individual frame) prevents data leakage — frames from
# the same game are visually similar, so mixing them across splits would inflate
# validation metrics.
#
# Games that contain rare classes (fewer than RARE_THRESHOLD annotations total)
# are always kept in train to avoid those classes disappearing from training data.
# The remaining games are split ~80 / 10 / 10.

import random
import shutil
from collections import defaultdict
from pathlib import Path

DATASET = Path("Project25YoloReady")
SEED = 42
RARE_THRESHOLD = 20  # classes with fewer total annotations than this are considered rare

images_dir = DATASET / "images"
labels_dir = DATASET / "labels"

# Group image stems by game ID (first segment of filename before "_")
games = defaultdict(list)
for img in images_dir.iterdir():
    if img.is_file():
        game_id = img.name.split("_")[0]
        games[game_id].append(img.stem)

# Count total annotations per class across the whole dataset
class_counts = defaultdict(int)
for lbl in labels_dir.glob("*.txt"):
    for line in lbl.read_text(encoding="utf-8").strip().splitlines():
        if line:
            class_counts[int(line.split()[0])] += 1

rare_classes = {cls for cls, cnt in class_counts.items() if cnt < RARE_THRESHOLD}

def has_rare(stems):
    """Return True if any label file for these stems contains a rare class."""
    for stem in stems:
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").strip().splitlines():
                if line and int(line.split()[0]) in rare_classes:
                    return True
    return False

# Separate games that must stay in train from those that can be split freely
forced_train, free_games = [], []
for game_id, stems in games.items():
    (forced_train if has_rare(stems) else free_games).append(game_id)

random.seed(SEED)
random.shuffle(free_games)

n = len(free_games)
n_val = max(1, round(n * 0.1))
n_test = max(1, round(n * 0.1))

val_games = set(free_games[:n_val])
test_games = set(free_games[n_val:n_val + n_test])
train_games = set(free_games[n_val + n_test:]) | set(forced_train)

def get_split(game_id):
    if game_id in val_games: return "val"
    if game_id in test_games: return "test"
    return "train"

for split in ("train", "val", "test"):
    (images_dir / split).mkdir(exist_ok=True)
    (labels_dir / split).mkdir(exist_ok=True)

# Move each image and its corresponding label into the correct split subfolder
for game_id, stems in games.items():
    split = get_split(game_id)
    for stem in stems:
        for ext in (".jpeg", ".jpg", ".png"):
            src_img = images_dir / f"{stem}{ext}"
            if src_img.exists():
                shutil.move(str(src_img), images_dir / split / src_img.name)
                break
        src_lbl = labels_dir / f"{stem}.txt"
        if src_lbl.exists():
            shutil.move(str(src_lbl), labels_dir / split / src_lbl.name)

for split in ("train", "val", "test"):
    imgs = len(list((images_dir / split).iterdir()))
    lbls = len(list((labels_dir / split).iterdir()))
    print(f"{split}: {imgs} images, {lbls} labels")

print(f"\nForced to train (contain rare classes): {len(forced_train)} games")
print(f"Rare class threshold: < {RARE_THRESHOLD} annotations")
