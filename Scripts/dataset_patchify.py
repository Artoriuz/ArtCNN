import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import os

patch_size = 256

for myFile in tqdm(sorted(glob.glob('./*.png'))):
    image = cv2.imread(myFile, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]

    patches = [
        image[y:y+patch_size, x:x+patch_size]
        for y in range(0, h - patch_size + 1, patch_size)
        for x in range(0, w - patch_size + 1, patch_size)
    ]

    for i, patch in enumerate(patches):
        cv2.imwrite(f"{Path(myFile).stem}-{i:04d}.png", patch)

    os.remove(myFile)
