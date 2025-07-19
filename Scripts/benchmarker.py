import jax.numpy as jnp
import dm_pix as pix
import jax
import cv2
import glob

def msssim(y_true, y_pred, max_val = 1):
    level = 5
    weights = jnp.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    msssim = []
    for _ in range(level):
        ssim_res = pix.ssim(y_true, y_pred, max_val = max_val)
        msssim.append(ssim_res)
        b, h, w, c = y_true.shape
        y_true = jax.image.resize(y_true, (b, h // 2, w // 2, c), method="bilinear")
        y_pred = jax.image.resize(y_pred, (b, h // 2, w // 2, c), method="bilinear")
    msssim = jnp.sum(jnp.stack(msssim, axis=1) * weights, axis=1)
    return msssim

output_file = open("benchmark_result.txt", "w")

print("Starting benchmarks")

reference = cv2.imread('./reference.png', cv2.IMREAD_COLOR)
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY, 0).astype(float) / 255.0
reference = jnp.asarray(reference)
reference = jnp.expand_dims(reference, 0)
reference = jnp.expand_dims(reference, -1)

filelist = sorted(glob.glob('./*.png'))
for myFile in filelist:
    if not "downscaled" in myFile:
        image = cv2.imread(myFile, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0).astype(float) / 255.0
        image = jnp.asarray(image)
        image = jnp.expand_dims(image, 0)
        image = jnp.expand_dims(image, -1)

        mae_score = pix.mae(reference, image)
        psnr_score = pix.psnr(reference, image)
        ssim_score = pix.ssim(reference, image, max_val = 1)
        msssim_score = msssim(reference, image, max_val = 1)

        print(f"{myFile} - MAE: {mae_score}, PSNR: {psnr_score}, SSIM: {ssim_score}, MS-SSIM: {msssim_score}\n")
        output_file.write(f"{myFile}, {mae_score}, {psnr_score}, {ssim_score}, {msssim_score}\n")

output_file.close()
