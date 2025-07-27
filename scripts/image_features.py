import cv2
import numpy as np
import pandas as pd
import os
import face_recognition
import matplotlib.pyplot as plt

IMAGE_DIR = "../data/images"
OUTPUT_CSV = "../data/image_features.csv"

def augment_image(img):
    aug_imgs = []
    aug_types = []
    # Original
    aug_imgs.append(img)
    aug_types.append("original")
    # Rotate +15
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1)
    aug_imgs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    aug_types.append("rotate+15")
    # Rotate -15
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -15, 1)
    aug_imgs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
    aug_types.append("rotate-15")
    # Horizontal flip
    aug_imgs.append(cv2.flip(img, 1))
    aug_types.append("hflip")
    # Grayscale
    aug_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    aug_types.append("grayscale")
    return aug_imgs, aug_types

def extract_features(img):
    # Face embedding (128-d)
    if len(img.shape) == 2:  # grayscale to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)
    embedding = encodings[0] if encodings else np.zeros(128)
    # Histogram (flattened)
    hist = cv2.calcHist([img_rgb], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()
    return embedding, hist

rows = []
for fname in os.listdir(IMAGE_DIR):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img = cv2.imread(os.path.join(IMAGE_DIR, fname))
        aug_imgs, aug_types = augment_image(img)

        # Visualization: Show original and augmentations
        plt.figure(figsize=(15, 3))
        for i, (aug_img, aug_type) in enumerate(zip(aug_imgs, aug_types)):
            plt.subplot(1, len(aug_imgs), i+1)
            if len(aug_img.shape) == 2:
                plt.imshow(aug_img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            plt.title(aug_type)
            plt.axis('off')
        plt.suptitle(f"Augmentations for {fname}")
        plt.show()

        for aug_img, aug_type in zip(aug_imgs, aug_types):
            embedding, hist = extract_features(aug_img)
            row = {"filename": fname, "augmentation": aug_type}
            for i, val in enumerate(embedding):
                row[f"embedding_{i}"] = val
            for i, val in enumerate(hist):
                row[f"hist_{i}"] = val
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
