import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


class KittiDataset(Dataset):
    

    def __init__(self, lidar_dir, intensity_dir,incidence_dir,binary_dir,label_dir,reflectance_dir,rgb_transform=None,
        lidar_transform=None,incidence_transform=None,intensity_transform=None,
        binary_transform=None,color_transform=None,label_transform=None, reflectance_transform=None):

        self.lidar_dir = lidar_dir
        #self.rgb_dir = rgb_dir
        #self.color_dir = color_dir
        self.intensity_dir = intensity_dir
        self.incidence_dir = incidence_dir
        self.binary_dir = binary_dir
        self.label_dir = label_dir 
        self.reflectance_dir = reflectance_dir

        self.lidar_images = os.listdir(lidar_dir)
        #self.rgb_images = os.listdir(rgb_dir)
        #self.color_images = os.listdir(color_dir)
        self.intensity_images = os.listdir(intensity_dir)
        self.incidence_images = os.listdir(incidence_dir)
        self.binary_images = os.listdir(binary_dir)
        self.label_images = os.listdir(label_dir)
        self.reflectance_images = os.listdir(reflectance_dir)

        self.rgb_transform = rgb_transform 
        self.lidar_transform = lidar_transform 
        self.incidence_transform=incidence_transform
        self.intensity_transform=intensity_transform
        self.binary_transform=binary_transform
        self.color_transform=color_transform
        self.label_transform=label_transform
        self.reflectance_transform=reflectance_transform 
        

    def __len__(self):
        return len(self.lidar_images)

    def __getitem__(self, index):
        lidar_path = os.path.join(self.lidar_dir, self.lidar_images[index])
        #rgb_path = os.path.join(self.rgb_dir, self.rgb_images[index])
        intensity_path = os.path.join(self.intensity_dir, self.intensity_images[index])
        incidence_path = os.path.join(self.incidence_dir , self.incidence_images[index])
        binary_path = os.path.join(self.binary_dir , self.binary_images[index])
        #color_path = os.path.join(self.color_dir , self.color_images[index])
        label_path = os.path.join(self.label_dir , self.label_images[index])
        reflectance_path = os.path.join(self.reflectance_dir , self.reflectance_images[index])
        
 

        lidar = Image.open(lidar_path).convert("L")
        #rgb = Image.open(rgb_path)
        intensity = Image.open(intensity_path).convert("L")
        incidence = Image.open(incidence_path).convert("L")
        binary = Image.open(binary_path).convert("L")
        #color = Image.open(color_path).convert("L")
        label = Image.open(label_path).convert("L")
        reflectance = Image.open(reflectance_path).convert("L")

        if self.lidar_transform is not None:
            intensity = self.intensity_transform(intensity) #For adding a channel
            lidar = self.lidar_transform(lidar)
            #rgb = self.rgb_transform(rgb)  # RGB image already has 3 channels
            incidence = self.incidence_transform(incidence)
            binary = self.binary_transform(binary)
            binary = (binary > 0.5).float()
            #color = self.color_transform(color)
            #color = (color > 0.5).float()
            label = self.label_transform(label)
            reflectance = self.reflectance_transform(reflectance)

        #Change Accordingly
    
        
        #concatenated_img = torch.cat((binary,incidence,lidar), dim=0) # T6 #in_channels=3
        #concatenated_img = torch.cat((binary,incidence,lidar, reflectance), dim=0) # T5 #in_channels=4
        #concatenated_img = torch.cat((binary, label,lidar), dim=0) #T4 # #in_channels=3
        #concatenated_img = torch.cat((binary, label,lidar, reflectance), dim=0) #T3 # #in_channels=4
        #concatenated_img = torch.cat((binary, label,incidence,lidar), dim=0) #T2 #in_channels=4
        concatenated_img = torch.cat((binary, label,incidence,lidar, reflectance), dim=0) #T1 #in_channels=5
      

   
         

        return concatenated_img,intensity