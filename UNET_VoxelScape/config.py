import torch

#Things to Change
    #in_channels
    #Loss_Plot_Path
    #Loss_CSV_Path
    #Checkpoint_Path 
    #Concatenated image in dataset.py 

#Voxelscape Dataset

# Hyperparameters 
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 2

out_channels = 1
in_channels = 5 #Change Accordingly

folder = "T1"  #Change Accordingly

Loss_Plot_Path = f"Objective 2/UNET/Output_2.0/{folder}/{folder}_loss_plot.png"
Loss_CSV_Path = f"Objective 2/UNET/Output_2.0/{folder}/{folder}_loss.csv"
Checkpoint_Path = f"Objective 2/UNET/Output_2.0/{folder}/{folder}_model.pth.tar"



PIN_MEMORY = True
LOAD_MODEL = False

base_path = "Voxelscape_Data/"

TRAIN_Lidar_DIR = base_path + "Train/train_lidar_depth"
#TRAIN_RGB_DIR = base_path + "Train/train_lidar_rgb"
#TRAIN_COLOR_DIR = base_path + "Train/train_color_mask"
TRAIN_Intensity_DIR = base_path + "Train/train_lidar_intensity"
TRAIN_Incidence_DIR = base_path + "Train/train_incidence_mask"
TRAIN_Binary_DIR = base_path + "Train/train_binary_mask"
TRAIN_LABEL_DIR = base_path + "Train/train_lidar_label"
TRAIN_REFLECTANCE_DIR = base_path + "Train/train_lidar_reflectance"

VAL_Lidar_DIR = base_path + "Val/val_lidar_depth"
#VAL_RGB_DIR = base_path + "Val/val_lidar_rgb"
#VAL_COLOR_DIR = base_path + "Val/val_color_mask"
VAL_Intensity_DIR = base_path + "Val/val_lidar_intensity"
VAL_Incidence_DIR = base_path + "Val/val_incidence_mask"
VAL_Binary_DIR = base_path + "Val/val_binary_mask"
VAL_LABEL_DIR = base_path + "Val/val_lidar_label"
VAL_REFLECTANCE_DIR = base_path + "Val/val_lidar_reflectance"

TEST_Lidar_DIR = base_path + "Test/test_lidar_depth"
#TEST_RGB_DIR = base_path + "Test/test_lidar_rgb"
#TEST_COLOR_DIR = base_path + "Test/test_color_mask"
TEST_Intensity_DIR = base_path + "Test/test_lidar_intensity"
TEST_Incidence_DIR = base_path + "Test/test_incidence_mask"
TEST_Binary_DIR = base_path + "Test/test_binary_mask"
TEST_LABEL_DIR = base_path + "Test/test_lidar_label"
TEST_REFLECTANCE_DIR = base_path + "Test/test_lidar_reflectance"

