import os
import shutil
import glob
from sklearn.model_selection import train_test_split

DATA_DIR = './final_data'  # Replace with your path
OUTPUT_DIR = './dataset'    # Where train/val/test folders will be created
CLASSES = ['Atelectasis', 'Cardiomegaly', 'No Finding', 'Nodule', 'Pneumothorax']

# Ratios for splitting:
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in CLASSES:
    # Collect all images for the current class
    class_dir = os.path.join(DATA_DIR, class_name)
    all_images = glob.glob(os.path.join(class_dir, '*.*'))  # *.jpg / *.png / etc.
    labels = [class_name] * len(all_images)
    
    # 1) Split into train_val vs test
    X_train_val, X_test, _, _ = train_test_split(
        all_images, 
        labels, 
        test_size=TEST_RATIO, 
        stratify=labels,
        random_state=42
    )
    
    # 2) Split train_val into train vs val
    val_split = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # portion from remaining data
    X_train, X_val, _, _ = train_test_split(
        X_train_val,
        [class_name]*len(X_train_val),
        test_size=val_split,
        stratify=[class_name]*len(X_train_val),
        random_state=42
    )
    
    # Helper function to copy images
    def copy_images(image_list, subset_name):
        subset_dir = os.path.join(OUTPUT_DIR, subset_name, class_name)
        os.makedirs(subset_dir, exist_ok=True)
        for img_path in image_list:
            shutil.copy(img_path, subset_dir)
    
    # Copy the files
    copy_images(X_train, 'train')
    copy_images(X_val,   'val')
    copy_images(X_test,  'test')
