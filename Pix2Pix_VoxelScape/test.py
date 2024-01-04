import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import config
from generator import Generator
from loss import MaskedL2Loss, MaskedL1Loss, L1Loss
from train import test_fn
from transform_utils import rgb_transform,lidar_transform,intensity_transform,incidence_transform,binary_transform,color_transform,label_transform
from utils import (

    get_loaders,
   
)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)




folder = "T1"

def main():
    
    loss_fn = MaskedL2Loss()
    loss_fn_Masked_L1 = MaskedL1Loss()
    loss_fn_L1 = L1Loss()

    model = Generator(in_channels=config.IN_CHANNELS, out_channels=1)  # Initialize your model architecture
    checkpoint = torch.load('/DATA2/Vivek/Code/Implementation/GAN/Pix2Pix_2.0/{folder}/gen.pth.tar_{folder}_epoch_200'.format(folder=folder))

    model.load_state_dict(checkpoint['state_dict'])
    model.to(config.DEVICE)
    model.eval()
    _, _, test_loader = get_loaders(
        config.TRAIN_Lidar_DIR, 
        config.TRAIN_RGB_DIR,
        config.TRAIN_COLOR_DIR, 
        config.TRAIN_Intensity_DIR,
        config.TRAIN_Incidence_DIR, 
        config.TRAIN_Binary_DIR,
        config.TRAIN_LABEL_DIR,

        config.VAL_Lidar_DIR, 
        config.VAL_RGB_DIR, 
        config.VAL_COLOR_DIR,
        config.VAL_Intensity_DIR,
        config.VAL_Incidence_DIR,
        config.VAL_Binary_DIR,
        config.VAL_LABEL_DIR,

        config.TEST_Lidar_DIR, 
        config.TEST_RGB_DIR, 
        config.TEST_COLOR_DIR,
        config.TEST_Intensity_DIR,
        config.TEST_Incidence_DIR,
        config.TEST_Binary_DIR,
        config.TEST_LABEL_DIR,



        config.BATCH_SIZE,
        rgb_transform,
        lidar_transform,
        incidence_transform,
        intensity_transform,
        binary_transform,
        color_transform,
        label_transform,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    # Test the model after training is done
    test_loss = test_fn(test_loader, model, loss_fn) #Masked L2 Loss
    print(f"Test loss L2: {test_loss}")

    test_loss = test_fn(test_loader, model, loss_fn_Masked_L1) #Masked L1 Loss
    print(f"Test loss Masked L1: {test_loss}")

    test_loss = test_fn(test_loader, model, loss_fn_L1) #Masked L1 Loss
    print(f"Test loss L1: {test_loss}")

main()
