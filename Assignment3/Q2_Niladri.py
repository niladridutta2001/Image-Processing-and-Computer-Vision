import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.signal import convolve2d
def gaussian_filter(size, std_dev):
    center = size // 2
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, size))
    distance_squared = (x - center)**2 + (y - center)**2
    kernel = np.exp(-distance_squared / (2 * std_dev**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(image, size, std_dev):
    custom_gaussian_filter = gaussian_filter(size, std_dev)
    filtered_image = convolve2d(image, custom_gaussian_filter, mode='same', boundary='symm')
    return filtered_image

def adaptive_mmse(image,kernel):
    # print(np.max(image),np.min(image))
    image_height, image_width = image.shape
    patch_size = 32
    overlap = 16
    a=np.zeros((kernel.shape[0],kernel.shape[1]))
    center = kernel.shape[0] // 2
    a[center,center]=1
    varz1=np.sum(np.square(kernel-a))*100
    modified_image = np.zeros_like(image, dtype=float)
    pixel_count = np.zeros_like(image, dtype=int)
    for y in range(0, image_height - patch_size + 1, patch_size - overlap):
        for x in range(0, image_width - patch_size + 1, patch_size - overlap):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_variance = np.var(patch)
            # print(patch_variance)
            # print("Patch at (y={}, x={}):".format(y,x), patch.shape, patch_variance)
            patch_modified = patch * ((patch_variance- varz1)/ patch_variance)
            # print( ((patch_variance- varz1)/ patch_variance))
            modified_image[y:y+patch_size, x:x+patch_size] += patch_modified
            pixel_count[y:y+patch_size, x:x+patch_size] += 1
    modified_image /= pixel_count
    vary1=np.var(image)
    high= ((vary1-varz1)/vary1)*image
    return modified_image,high

def SURE_shrink(image,kernel):
    image_height, image_width = image.shape
    sub=np.min(image)+np.max(image)
    patch_size = 32
    overlap = 16
    a=np.zeros((kernel.shape[0],kernel.shape[1]))
    center = kernel.shape[0] // 2
    a[center,center]=1
    varz1=np.sum((np.square(kernel-a)))*100
    modified_image = np.zeros_like(image, dtype=float)
    pixel_count = np.zeros_like(image, dtype=int)
    for y in range(0, image_height - patch_size + 1, patch_size - overlap):
        for x in range(0, image_width - patch_size + 1, patch_size - overlap):
            patch = image[y:y+patch_size, x:x+patch_size]
            # max_values = np.maximum(-patch, -t)
            t_values = np.linspace(0, 50,500)
            sure = []
            for t in t_values:
                g = np.minimum(np.abs(patch),t)
                dg = np.where(np.abs(patch) < t, -1, 0)
                sureval = np.sum(np.square(g))+ (np.sum(dg)*2*varz1)
                sure.append(sureval)
            min_value = min(sure)
            # t=sure.index(min_value)
            tmin = t_values[sure.index(min_value)]
            # print(tmin)
            patch_modified = np.where((patch) < 0, -1, 1)*np.maximum(0, np.abs(patch) - tmin)
            modified_image[y:y+patch_size, x:x+patch_size] += patch_modified
            pixel_count[y:y+patch_size, x:x+patch_size] += 1
    modified_image /= pixel_count
    return modified_image

def filter1(image):
    noise = np.random.normal(0, 10, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    mse=np.zeros(15)
    f_len=np.array([3,7,11])
    std_dev=np.array([0.1,1,2,4,8])
    k=0
    for i in range(3):
        for j in range(5):
            blurred_image = apply_gaussian_filter(noisy_image,f_len[i], std_dev[j])
            mse[k]=np.sum(np.square(blurred_image-image))/(image.shape[0]*image.shape[1])
            k+=1
    best=np.argmin(mse)
    print(mse,best)
    print("The least MSE occurs at filter size:",f_len[best//5],"and std dev",std_dev[best%5])
    smooth=apply_gaussian_filter(noisy_image,f_len[1],std_dev[3])
    high_img=noisy_image-smooth
    kernel=gaussian_filter(f_len[1],std_dev[3])
    new_high_img1,new_high_img2= adaptive_mmse(high_img,kernel)
    new_img1=smooth+new_high_img1

    print("MSE with only smoothing filter is",np.sum(np.square(smooth-image))/(image.shape[0]*image.shape[1]))
    print("MSE with adaptive MMSE filter is:",np.sum(np.square(new_img1-image))/(image.shape[0]*image.shape[1]))
    new_high_img3=SURE_shrink(high_img,kernel)
    new_img3=smooth+new_high_img3
    print("MSE with adaptive shrinkage filter is:",np.sum(np.square(new_img3-image))/(image.shape[0]*image.shape[1]))

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(smooth, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(new_img1, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(new_img3, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__=="__main__":
    input_image= cv2.imread("C:/Users/nilad/Downloads/lighthouse2.bmp",cv2.IMREAD_GRAYSCALE)
    input_image=input_image.astype(np.float64)
    filter1(input_image)