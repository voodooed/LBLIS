import time
import torch
from utils import save_checkpoint, load_checkpoint, save_outputs
import torch.nn as nn
import torch.optim as optim
import config
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn
import torch.optim as optim
from train import train_fn
from transform_utils import rgb_transform,lidar_transform,intensity_transform,incidence_transform,binary_transform,color_transform,label_transform,reflectance_transform
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
)




def main():

    disc = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    gen = Generator(in_channels=config.IN_CHANNELS, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()


    if config.LOAD_MODEL:
        checkpoint_gen = load_checkpoint( config.CHECKPOINT_GEN_LOAD, gen, opt_gen, config.LEARNING_RATE,)
        checkpoint_disc = load_checkpoint( config.CHECKPOINT_DISC_LOAD, disc, opt_disc, config.LEARNING_RATE,)

    train_loader, val_loader= get_loaders(
        config.TRAIN_Lidar_DIR, 
        #config.TRAIN_RGB_DIR,
        #config.TRAIN_COLOR_DIR, 
        config.TRAIN_Intensity_DIR,
        config.TRAIN_Incidence_DIR, 
        config.TRAIN_Binary_DIR,
        config.TRAIN_LABEL_DIR,
        config.TRAIN_REFLECTANCE_DIR,

        config.VAL_Lidar_DIR, 
        #config.VAL_RGB_DIR, 
        #config.VAL_COLOR_DIR,
        config.VAL_Intensity_DIR,
        config.VAL_Incidence_DIR,
        config.VAL_Binary_DIR,
        config.VAL_LABEL_DIR,
        config.VAL_REFLECTANCE_DIR,


        config.BATCH_SIZE,
        rgb_transform,
        lidar_transform,
        incidence_transform,
        intensity_transform,
        binary_transform,
        color_transform,
        label_transform,
        reflectance_transform,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    total_epochs = config.NUM_EPOCHS

    for epoch in range(total_epochs):
        epoch_start_time = time.time()

        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        remaining_epochs = total_epochs - epoch - 1
        estimated_time_remaining = remaining_epochs * elapsed_time
        print(f'Estimated time remaining for training: {estimated_time_remaining / 3600} hours')

        if config.SAVE_MODEL and epoch in [0, 48, 49, 99, 149, 199]:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN + f"_epoch_{epoch + 1}")
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC + f"_epoch_{epoch + 1}")

        save_outputs(gen, val_loader, epoch, folder=config.OUTPUT_FOLDER)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time for training: {total_time / 3600} hours')


if __name__ == "__main__":
    
    main()