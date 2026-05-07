import os
import cv2
import numpy as np

def load_yale_dataset(data_folder, img_size=(64, 64)):
    images = []
    labels = []

    for filename in sorted(os.listdir(data_folder)):
        filepath = os.path.join(data_folder, filename)

        # ── Skip anything that is not a subject image ──────────
        if os.path.isdir(filepath):
            continue
        if not filename.startswith('subject'):   # skips Readme, .info, etc.
            continue

        # Extract subject number from filename (e.g. subject01 → label 1)
        subject_part = filename.split('.')[0]        # "subject01"
        try:
            label = int(subject_part.replace('subject', ''))
        except ValueError:
            print(f"Skipping unrecognised file: {filename}")
            continue

        # Load as greyscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL for .gif or extensionless files
            try:
                from PIL import Image
                img = np.array(Image.open(filepath).convert('L'))
            except Exception as e:
                print(f"Could not open {filename}: {e}")
                continue

        # Resize and normalise
        img = cv2.resize(img, img_size)
        img = img / 255.0                            # values in [0, 1]
        images.append(img.flatten())                 # 64x64 = 4096 values
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)
    print(f"Loaded {X.shape[0]} images, {len(set(y))} subjects, feature dim = {X.shape[1]}")
    return X, y