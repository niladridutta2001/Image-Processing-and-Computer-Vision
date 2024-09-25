import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset, TensorDataset
from torchvision.transforms import ToTensor, Resize
from torchvision.io import read_image
from tqdm import tqdm
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
resnet18 = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def obtain_files(path):
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files]
    return files

def make_data(path, image_shape):
    images_filepaths = obtain_files(os.path.join(path, 'images'))
    masks_filepaths = obtain_files(os.path.join(path, 'masks'))
    images = []
    masks = []
    with open("C:/Users/nilad/Downloads/AIP asgmt2/dataset/label2cmap.json") as json_file:
        label2cmap = json.load(json_file)

    color_map = [label2cmap[labels] for labels in label2cmap]

    color_tree = KDTree(color_map.copy())

    for image_path in images_filepaths:
        images.append(cv2.resize(cv2.imread(image_path), image_shape))

    for mask_path in masks_filepaths:
        mask = cv2.resize(cv2.imread(mask_path), image_shape)
        mask_labeled = color_tree.query(mask*255)[1]
        masks.append(np.eye(len(color_map))[mask_labeled])

    # print(f'{len(images)} files found')

    return images,masks

images, masks = make_data("C:/Users/nilad/Downloads/AIP asgmt2/dataset", image_shape = (224,224))
train_images, test_images, train_masks, test_masks = train_test_split(images,masks,test_size = 0.2)

train_images = np.array(train_images)
train_masks = np.array(train_masks)
test_images = np.array(test_images)
test_masks = np.array(test_masks)
trainimages_tensor = torch.from_numpy(train_images,).to(torch.float)
trainmasks_tensor = torch.from_numpy(train_masks,).to(torch.float)
testimages_tensor = torch.from_numpy(test_images,).to(torch.float)
testmasks_tensor = torch.from_numpy(test_masks,).to(torch.float)

train_data = TensorDataset(trainimages_tensor, trainmasks_tensor)
test_data = TensorDataset(testimages_tensor, testmasks_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

for param in resnet18.parameters():
    param.requires_grad = False

def IOU(mask, output):
    num_channels, _, _ = mask.shape
    iou = 0
    for i in range(num_channels):

        intersection = np.logical_and(mask[i, :, :], output[i, :, :]).sum()
        union = np.logical_or(mask[i, :, :], output[i, :, :]).sum()
        iou += (intersection / union) * 100

    average_iou = iou / num_channels

    return average_iou
def train_model(model):
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader):
            images=images.to(device)
            masks=masks.to(device)
            optimizer.zero_grad()
            images=images.permute(0,3,1,2)
            masks=masks.permute(0,3,1,2)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

def Accuracy(mask, output):
    num_channels, height, width = mask.shape
    total_pixels = height * width
    correct_pixels = np.logical_and(mask, output).sum()
    accuracy = correct_pixels / total_pixels
    return accuracy
def output_display(model):
    ioulist=[]
    acclist=[] 
    model=model.to(device)
    for images, masks in test_loader:
        masks=masks.permute(0,3,1,2)
        images=images.permute(0,3,1,2)
        images,masks=images.to(device),masks.to(device)
        outputs = model(images)
        i=0
        for mask, output in zip(masks, outputs):
            i+=1
            mask=mask.cpu().detach().numpy()
            output=output.cpu().detach().numpy()

            iou = IOU(mask,output)
            ioulist.append(iou)
            
            accuracy = Accuracy(mask,output)
            acclist.append(accuracy)
            if i<10:
                labels=output.argmax(axis=0)
                img=(labels/8)*225
                img = np.clip(img.astype(np.uint8), 0, 255)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                plt.imshow(mask.argmax(axis=0))
                plt.axis('off')
                plt.show()
        return np.mean(acclist),np.mean(ioulist) 

class Skip(nn.Module):
    def __init__(self, frozen):
        super(Skip, self).__init__()
        resnet=list(frozen.children())
        self.l1 = nn.Sequential(resnet[0],resnet[1],resnet[2],resnet[3])
        self.l2 = resnet[4]
        self.l3 = resnet[5]
        self.l4 = resnet[6]
        self.l5 = resnet[7]
        self.l6=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),nn.ReLU())
        self.l7=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(128),nn.ReLU())
        self.l8=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),nn.ReLU())
        self.l9 = nn.Sequential(nn.Conv2d(64, 9, kernel_size=1,stride=1,padding=0),
        nn.AdaptiveAvgPool2d((224, 224))
        )


    def forward(self,x):
      x=self.l1(x)
      op1=x
      x=self.l2(x)
      op2=x
      x=self.l3(x)
      op3=x
      x=self.l4(x)
      op4=x
      x=self.l5(x)
      x=self.l6(x)
      x=x+op4
      x=self.l7(x)
      x=x+op3
      x=self.l8(x)
      x=x+op2
      x=self.l9(x)
      return x

class noSkip(nn.Module):
    def __init__(self, frozen):
        super(noSkip, self).__init__()
        resnet=list(frozen.children())
        self.l1 = nn.Sequential(*resnet)[:4]
        self.l2 = resnet[4]
        self.l3 = resnet[5]
        self.l4 = resnet[6]
        self.l5 = resnet[7]
        self.l6=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),nn.ReLU())
        self.l7=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(128),nn.ReLU())
        self.l8=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),nn.ReLU())
        self.l9 = nn.Sequential(nn.Conv2d(64, 9, kernel_size=1,stride=1,padding=0),
        nn.AdaptiveAvgPool2d((224, 224))
        )


    def forward(self,x):
      x=self.l1(x)
      x=self.l2(x)
      x=self.l3(x)
      x=self.l4(x)
      x=self.l5(x)
      x=self.l6(x)
      x=self.l7(x)
      x=self.l8(x)
      x=self.l9(x)
      return x

model1=Skip(resnet18)
model2=noSkip(resnet18)
print("Training model with skips...")
train_model(model1)
print("Training model without skips...")
train_model(model2)
file_path1 = "./skipmodel.pt"
torch.save(model1.state_dict(), file_path1)
file_path2 = "./no_skipmodel.pt"
torch.save(model2.state_dict(), file_path2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.load_state_dict(torch.load(file_path1))
model1.eval()
model1.to(device)
model2.load_state_dict(torch.load(file_path2))
model2.eval()
model2.to(device)
acclist1, ioulist1=output_display(model1)
acclist2, ioulist2=output_display(model2)
# print(f"the accuracy with skip connection is: {np.mean(acclist1)} and the IoU is {np.mean(ioulist1)}")
# print(f"the accuracy without skip connection is: {np.mean(acclist2)} and the IoU is {np.mean(ioulist2)}")