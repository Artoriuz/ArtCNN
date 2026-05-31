import cv2
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm

rotations = [0, 90, 180, 270]

filelist = sorted(glob.glob('./*.png'))

for myFile in tqdm(filelist):
    img = cv2.imread(myFile, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error reading image: {myFile}")
        continue

    for rotation in rotations:
        rotated_img = np.rot90(img, rotation // 90)
        cv2.imwrite(str(Path(myFile).stem) + str(rotation) + ".png", rotated_img)

    flipped_img = cv2.flip(img, 1)

    for rotation in rotations:
        rotated_flipped_img = np.rot90(flipped_img, rotation // 90)
        cv2.imwrite(str(Path(myFile).stem) + str(rotation) + "f.png", rotated_flipped_img)

    os.remove(myFile)
