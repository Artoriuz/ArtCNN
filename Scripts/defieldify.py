from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

even = cv2.imread("even.png", cv2.IMREAD_COLOR)
even = cv2.cvtColor(even, cv2.COLOR_BGR2GRAY)

odd = cv2.imread("ArtCNN_R8F64_DEINT.png", cv2.IMREAD_COLOR)
odd = cv2.cvtColor(odd, cv2.COLOR_BGR2GRAY)

output = np.zeros((1080, 1920))
output[::2, :] = even
output[1::2, :] = odd

cv2.imwrite("output.png", output)
