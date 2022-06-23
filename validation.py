from math import log10, sqrt
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--sr', type=str,
                    help='path of SR image')
parser.add_argument('--gt', type=str,
                    help='path of GT image')              
args = parser.parse_args()
  
def PSNR(gt_img, sr_img, crop_size=4):

    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]


    gt_img = gt_img / 255.
    sr_img = sr_img / 255.


    cropped_sr_img = (cropped_sr_img * 255).astype(np.float64)
    cropped_gt_img = (cropped_gt_img * 255).astype(np.float64)

    
    mse = np.mean((cropped_gt_img - cropped_sr_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     gt = cv2.imread(args.gt)
     print(f'gt.shape = {gt.shape}')
     sr = cv2.imread(args.sr, 1)
     print(f'sr.shape = {sr.shape}')
     value = PSNR(gt, sr)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()