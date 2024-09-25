import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.io
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def Ncut_seg3(image):    
    height, width = image.shape[0], image.shape[1]
    image2=image
    image = image.flatten()
    abs_diff = np.abs(np.float32(image[:, np.newaxis]) - np.float32(image[np.newaxis, :]))
    print("Processing three segments output...") 
    W = np.exp(-abs_diff**2/25)
    horiz, vert = np.meshgrid(np.arange(height), np.arange(width))
    horiz = horiz.flatten()
    vert = vert.flatten()

    dist = np.sqrt((horiz[:, np.newaxis] - horiz[np.newaxis, :]) ** 2 + (vert[:, np.newaxis] - vert[np.newaxis, :]) ** 2)
    weight = np.exp(-dist / 100)
    W=(weight)*W

    D = np.zeros_like(W)
    np.fill_diagonal(D, np.sum(W, axis=1))
    D2=np.diag(1 / np.sqrt(np.diag(D)))
    A = np.matmul(D2 , np.matmul((D - W) , D2) )
    
    eigval, eig_vs = eigsh(A, k=6, which='SA')
    # print(eig_vs.shape)
    eig_vs= D2.dot(eig_vs[:,3])

    segmented_img = eig_vs.reshape(image2.shape[0], image2.shape[1])
    segmented_img=segmented_img*10000

    plt.figure(figsize=(20,20))

    maxs,mins=np.max(segmented_img),np.min(segmented_img)
    # print(maxs,mins)
    flat_segmented_display = segmented_img.flatten()
    segmented_img_display = segmented_img.copy()
    threshold1=0.0
    threshold2=0.45
    segmented_img_display[segmented_img_display < threshold1] = 0

    segmented_img_display[segmented_img_display >threshold2] = 255

    segmented_img_display[(segmented_img_display >threshold1) & (segmented_img_display <= threshold2)] =128
    
    plt.imshow(segmented_img_display, cmap="viridis")
    plt.show()

def Ncut_seg2(image):    
    height, width = image.shape[0], image.shape[1]
    image2=image
    image = image.flatten()
    abs_diff = np.abs(np.float32(image[:, np.newaxis]) - np.float32(image[np.newaxis, :]))
    print("Processing two segments output...")
    W = np.exp(-abs_diff**2/25)
    horiz, vert = np.meshgrid(np.arange(height), np.arange(width))
    horiz = horiz.flatten()
    vert = vert.flatten()

    dist = np.sqrt((horiz[:, np.newaxis] - horiz[np.newaxis, :]) ** 2 + (vert[:, np.newaxis] - vert[np.newaxis, :]) ** 2)

    weight = np.exp(-dist / 100)
    W=(weight)*W
    D = np.zeros_like(W)
    np.fill_diagonal(D, np.sum(W, axis=1))
    D2=np.diag(1 / np.sqrt(np.diag(D)))
    A = np.matmul(D2 , np.matmul((D - W) , D2) ) 
    eigval, eig_vs = eigsh(A, k=2, which='SA')
    
    
    eig_vs= D2.dot(eig_vs[:,1])
    med=np.median(eig_vs)
    segmented_img = eig_vs.reshape(image2.shape[0], image2.shape[1])
    plt.figure(figsize=(20,20))
    plt.imshow(segmented_img>0, cmap="viridis")
    plt.show()
        
def Kmeans(image):
    print(image.shape)
    rows, cols, _ = image.shape
    image_2d = image.reshape(rows * cols, -1)

    min_clusters = 3
    max_clusters = 6

    plt.figure(figsize=(12, 8))
    for k in range(min_clusters, max_clusters + 1):
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(image_2d)
        labels = kmeans.labels_
        segmented_img = labels.reshape(rows, cols)

        # Plot the segmented image
        plt.subplot(2, 2, k - min_clusters + 1)
        plt.imshow(segmented_img, cmap='viridis')
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        plt.title(f'Number of Clusters (K) = {k}', fontdict=title_font)
        plt.axis('off')  

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    image_path1 = "C:/Users/nilad/Downloads/AIP asgmt2/img2.jpeg"
    image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(image_path1)
    image = cv2.resize(image, (int(0.1*image.shape[1]), int(0.1*image.shape[0])))
    Kmeans(image1)
    Ncut_seg2(image)
    Ncut_seg3(image) #tuned only for image 2, totally off for other images  '''
