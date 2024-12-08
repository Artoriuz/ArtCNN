from scipy.ndimage import convolve
import numpy as np
import cv2
import glob
import skimage

def msssim(im1, im2, data_range = 255, channel_axis = None):
    level = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    msssim = []
    for _ in range(level):
        ssim_res = skimage.metrics.structural_similarity(im1 = im1, im2 = im2, data_range = data_range, channel_axis = channel_axis)
        msssim.append(ssim_res)
        if channel_axis:
            filtered_im1 = np.zeros_like(im1)
            filtered_im2 = np.zeros_like(im2)
            for channel in range(channel_axis):
                filtered_im1[:, :, channel] = convolve(im1[:, :, channel], downsample_filter, mode='reflect')
                filtered_im2[:, :, channel] = convolve(im2[:, :, channel], downsample_filter, mode='reflect')
            im1 = filtered_im1[::2, ::2, :]
            im2 = filtered_im2[::2, ::2, :]
        else:    
            filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
            filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
            im1 = filtered_im1[::2, ::2]
            im2 = filtered_im2[::2, ::2]
    return np.average(np.array(msssim), weights=weights)

output_file = open("benchmark_result.txt", "w")

print("Starting benchmarks")

reference = cv2.imread('./reference.png', cv2.IMREAD_COLOR)
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY, 0).astype(float) / 255.0

filelist = sorted(glob.glob('./*.png'))
for myFile in filelist:
    if not "downscaled" in myFile:
        image = cv2.imread(myFile, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0).astype(float) / 255.0

        mae_score = np.mean(np.absolute(reference - image))
        psnr_score = skimage.metrics.peak_signal_noise_ratio(reference, image)
        ssim_score = skimage.metrics.structural_similarity(reference, image, data_range = 1, channel_axis = None)
        msssim_score = msssim(reference, image, data_range = 1, channel_axis = None)
        
        print(f"{myFile} - MAE: {mae_score}, PSNR: {psnr_score}, SSIM: {ssim_score}, MS-SSIM: {msssim_score}\n")
        output_file.write(f"{myFile}, {mae_score}, {psnr_score}, {ssim_score}, {msssim_score}\n")

output_file.close()
