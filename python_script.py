import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================

SOURCE_DIR = "dataset"          # your original dataset folder
OUTPUT_DIR = "dataset_split"    # new split dataset folder

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# =========================
# CHECK RATIOS
# =========================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(SEED)

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "val", "test"]:
    Path(os.path.join(OUTPUT_DIR, split)).mkdir(parents=True, exist_ok=True)

# =========================
# PROCESS EACH CLASS
# =========================

classes = os.listdir(SOURCE_DIR)

for class_name in classes:

    class_path = os.path.join(SOURCE_DIR, class_name)

    # Skip if not folder
    if not os.path.isdir(class_path):
        continue

    # Get image files only
    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]

    # Shuffle images
    random.shuffle(image_files)

    total_images = len(image_files)

    # Calculate split sizes
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    # Split files
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    print(f"\nClass: {class_name}")
    print(f"Total: {total_images}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

    # Copy files
    for split_name, files in splits.items():

        split_class_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        Path(split_class_dir).mkdir(parents=True, exist_ok=True)

        for file_name in files:

            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(split_class_dir, file_name)

            shutil.copy2(src_path, dst_path)

print("\nDataset successfully split!")