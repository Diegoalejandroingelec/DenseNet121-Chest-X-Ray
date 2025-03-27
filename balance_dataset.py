import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# A) Paths & Parameters
# -----------------------------
original_base_dir = "./dataset"         # Original dataset folder with train, test, val splits
balanced_base_dir = "./balanced_dataset"  # New folder for the balanced & re-split dataset
IMG_SIZE = (512, 512)

# New split proportions
split_proportions = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# -----------------------------
# B) Gather All Images by Class
# -----------------------------
# Get union of class names from all splits
all_classes = set()
for split in ["train", "test", "val"]:
    split_dir = os.path.join(original_base_dir, split)
    if os.path.isdir(split_dir):
        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                all_classes.add(cls)
all_classes = sorted(list(all_classes))

# Create a dictionary to store image paths per class
images_by_class = {cls: [] for cls in all_classes}

# Collect file paths from each split for every class
for split in ["train", "test", "val"]:
    split_dir = os.path.join(original_base_dir, split)
    for cls in all_classes:
        class_dir = os.path.join(split_dir, cls)
        if os.path.exists(class_dir):
            image_files = [os.path.join(class_dir, f)
                           for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            images_by_class[cls].extend(image_files)

# Print the total counts per class
original_counts = {cls: len(images_by_class[cls]) for cls in all_classes}
print("Original combined counts per class:", original_counts)

# -----------------------------
# C) Balance the Dataset
# -----------------------------
# Determine the minimum count across all classes
min_count = min(original_counts.values())
print("Minimum count across classes:", min_count)

# For each class, randomly sample min_count images to balance the dataset
balanced_images_by_class = {}
for cls, file_list in images_by_class.items():
    if len(file_list) >= min_count:
        balanced_images_by_class[cls] = random.sample(file_list, min_count)
    else:
        print(f"Warning: Class {cls} has less than the minimum count. Using all available images.")
        balanced_images_by_class[cls] = file_list

# -----------------------------
# D) Split the Balanced Dataset
# -----------------------------
# Dictionary to store the split data: for each split, store image paths for each class
split_data = {split: {cls: [] for cls in all_classes} for split in split_proportions.keys()}

for cls, file_list in balanced_images_by_class.items():
    random.shuffle(file_list)  # Shuffle to ensure randomness
    n_total = len(file_list)
    n_train = int(n_total * split_proportions["train"])
    n_val = int(n_total * split_proportions["val"])
    # The rest go to test
    n_test = n_total - n_train - n_val

    split_data["train"][cls] = file_list[:n_train]
    split_data["val"][cls]   = file_list[n_train:n_train+n_val]
    split_data["test"][cls]  = file_list[n_train+n_val:]
    
    print(f"Class {cls}: total={n_total}, train={n_train}, val={n_val}, test={n_test}")

# -----------------------------
# E) Copy, Resize, and Save Images
# -----------------------------
# Loop through each split and class, create destination folders, and process images
for split in split_data:
    for cls in split_data[split]:
        dest_dir = os.path.join(balanced_base_dir, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for src_path in split_data[split][cls]:
            try:
                with Image.open(src_path) as img:
                    img_resized = img.resize(IMG_SIZE, resample=Image.Resampling.LANCZOS)
                    # Save with the original filename
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(dest_dir, filename)
                    img_resized.save(dest_path)
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

# -----------------------------
# F) Plot Final Distribution
# -----------------------------
final_counts = {}
for split in split_data:
    final_counts[split] = {}
    for cls in all_classes:
        dir_path = os.path.join(balanced_base_dir, split, cls)
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            final_counts[split][cls] = len(images)
        else:
            final_counts[split][cls] = 0

print("Final distribution per split:", final_counts)

# Plot distribution for each split
for split, counts in final_counts.items():
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.title(f"Final Distribution for {split.capitalize()}")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
