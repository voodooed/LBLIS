import os
import torch
import torchvision
from dataset import KittiDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch
import config
from torchvision.utils import save_image
import math

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(
    train__lidar_dir,
    #train_rgb_dir,
    #train_color_dir,
    train_intensity_dir,
    train_incidence_dir,
    train_binary_dir,
    train_label_dir,
    train_reflectance_dir,

    val_lidar_dir,
    #val_rgb_dir,
    #val_color_dir,
    val_intensity_dir,
    val_incidence_dir,
    val_binary_dir,
    val_label_dir,
    val_reflectance_dir,

    batch_size,
    rgb_transform,
    lidar_transform,
    incidence_transform,
    intensity_transform,
    binary_transform,
    color_transform,
    label_transform,
    reflectance_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = KittiDataset(
        lidar_dir = train__lidar_dir,
        #rgb_dir=train_rgb_dir,
        #color_dir=train_color_dir,
        intensity_dir=train_intensity_dir,
        incidence_dir=train_incidence_dir,
        binary_dir = train_binary_dir,
        label_dir = train_label_dir,
        reflectance_dir = train_reflectance_dir,

        rgb_transform=rgb_transform,
        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        intensity_transform=intensity_transform,
        binary_transform=binary_transform,
        color_transform=color_transform,
        label_transform=label_transform,
        reflectance_transform=reflectance_transform,
       
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KittiDataset(
        lidar_dir = val_lidar_dir,
        #rgb_dir=val_rgb_dir,
        #color_dir=val_color_dir,
        intensity_dir=val_intensity_dir,
        incidence_dir=val_incidence_dir,
        binary_dir = val_binary_dir,
        label_dir = val_label_dir,
        reflectance_dir = val_reflectance_dir,

        rgb_transform=rgb_transform,
        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        intensity_transform=intensity_transform,
        binary_transform=binary_transform,
        color_transform=color_transform,
        label_transform=label_transform,
        reflectance_transform=reflectance_transform,

    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )


    return train_loader, val_loader



def save_outputs(gen, val_loader, epoch, folder, num_images=25):
    gen.eval()

    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    with torch.no_grad():
        y_fake = gen(x)
        y_fake_denormalized = (y_fake * 0.2276) + 0.4257  # Denormalize generated fake images
        y_denormalized = (y * 0.2276) + 0.4257  # Denormalize ground truth images

    # Create a new directory for this epoch
    epoch_dir = os.path.join(folder, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image individually
    for i in range(min(num_images, x.size(0))):

        # Get the image names from the val_loader
        image_name = os.path.splitext(val_loader.dataset.lidar_images[i])[0]

        # Create the filenames for fake and true images
        fake_image_name = f"{image_name}_fake.png"
        real_image_name = f"{image_name}_real.png"

        # Save the denormalized fake and true images using torchvision's save_image
        save_image(y_fake_denormalized[i], os.path.join(epoch_dir, fake_image_name))
        save_image(y_denormalized[i], os.path.join(epoch_dir, real_image_name))

    gen.train()


def save_image_grid(images, filename):
    save_image(images, filename, nrow=int(math.sqrt(images.size(0))), normalize=True, range=(0, 1))



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr