import torch

Trial_Num = "T3" #Change
Trial_Path = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 2/GAN/{Trial_Num}/Model" #Change
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 5 #Change
OUT_CHANNELS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False #Check this
SAVE_MODEL = True
CHECKPOINT_DISC = f"{Trial_Path}/disc.pth.tar_{Trial_Num}"
CHECKPOINT_GEN = f"{Trial_Path}/gen.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_LOAD = "disc.pth.tar_epoch_100"
CHECKPOINT_GEN_LOAD = "gen.pth.tar_epoch_100"
OUTPUT_FOLDER = f"/home/viveka21/projects/def-rkmishra-ab/viveka21/Objective 2/GAN/{Trial_Num}/Output" #Change

#Input Directory

base_path = "/home/viveka21/projects/def-rkmishra-ab/viveka21/Kitti_Data/"

TRAIN_Lidar_DIR = base_path + "Train/train_lidar_depth"
TRAIN_RGB_DIR = base_path + "Train/train_lidar_rgb"
TRAIN_COLOR_DIR = base_path + "Train/train_color_mask"
TRAIN_Intensity_DIR = base_path + "Train/train_lidar_intensity"
TRAIN_Incidence_DIR = base_path + "Train/train_incidence_mask"
TRAIN_Binary_DIR = base_path + "Train/train_binary_mask"
TRAIN_LABEL_DIR = base_path + "Train/train_lidar_label"
TRAIN_REFLECTANCE_DIR = base_path + "Train/train_lidar_reflectance"

VAL_Lidar_DIR = base_path + "Val/val_lidar_depth"
VAL_RGB_DIR = base_path + "Val/val_lidar_rgb"
VAL_COLOR_DIR = base_path + "Val/val_color_mask"
VAL_Intensity_DIR = base_path + "Val/val_lidar_intensity"
VAL_Incidence_DIR = base_path + "Val/val_incidence_mask"
VAL_Binary_DIR = base_path + "Val/val_binary_mask"
VAL_LABEL_DIR = base_path + "Val/val_lidar_label"
#VAL_REFLECTANCE_DIR = base_path + "VAL/val_lidar_reflectance"
VAL_REFLECTANCE_DIR = "/home/viveka21/projects/def-rkmishra-ab/viveka21/Kitti_Data/Val/val_lidar_reflectance"

