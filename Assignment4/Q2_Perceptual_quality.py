import numpy as np 
from scipy.io import loadmat
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import lpips
import torch
import torchvision.transforms as transforms
import math
mat_data = loadmat('C:/Users/nilad/Downloads/AIP asgmt4/hw5/hw5.mat')
A = mat_data['blur_dmos']
B = mat_data['blur_orgs']
C = mat_data['refnames_blur']
psnr=np.zeros(145)
ssim_vals=np.zeros(145)
d_alex=np.zeros(145)
d_vgg=np.zeros(145)
transform = transforms.Compose([
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])
for i in range(144):
    name=str(C[0,i])
    name = name.strip("[]")
    name=name.strip("''")
    ref_image= cv2.imread("C:/Users/nilad/Downloads/AIP asgmt4/hw5/refimgs/"+(name))
    blur_image=cv2.imread("C:/Users/nilad/Downloads/AIP asgmt4/hw5/gblur/img"+str(i+1)+".bmp")
    psnr[i] =10 * math.log10((255 ** 2) / ((ref_image - blur_image) ** 2).mean())

    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
    ssim_vals[i], _ = ssim(ref_gray, blur_gray, full=True)
    image_normalized = transform(ref_image)
    ref_image1 = image_normalized.unsqueeze(0)
    image_normalized = transform(blur_image)
    blur_image1 = image_normalized.unsqueeze(0)
    loss_fn1 = lpips.LPIPS(net='alex')
    d_alex[i] = loss_fn1.forward(ref_image1,blur_image1)
    loss_fn2 = lpips.LPIPS(net='vgg')
    d_vgg[i] = loss_fn2.forward(ref_image1,blur_image1)

x=(A[:,:145]).T
spearman_corr1,_= spearmanr(psnr.reshape(-1,1), x)
spearman_corr2,_= spearmanr(ssim_vals.reshape(-1,1), x)
spearman_corr3,_= spearmanr(d_alex.reshape(-1,1), x)
spearman_corr4,_= spearmanr(d_vgg.reshape(-1,1), x)
print("Spearman coefficient: With PSNR:",spearman_corr1," SSIM",spearman_corr2," LPIPS(Alexnet):",spearman_corr3, "VGG:",spearman_corr4)