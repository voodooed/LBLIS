import torch
import random
import config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import UNET
from loss import MaskedL2Loss
from train import train_fn,val_fn,test_fn
from transform_utils import rgb_transform,lidar_transform,intensity_transform,incidence_transform,binary_transform,color_transform,label_transform, reflectance_transform
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    plot_losses
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)



def main():
  
    model = UNET(in_channels=config.in_channels , out_channels=config.out_channels).to(config.DEVICE)

    loss_fn = MaskedL2Loss()
    #loss_fn = nn.MSELoss()
    #loss_fn = MSE_SSIM_Loss()

    # Create the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    train_loader, val_loader, test_loader = get_loaders(
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
        #config.VAL_COLOR_DIR ,
        config.VAL_Intensity_DIR,
        config.VAL_Incidence_DIR,
        config.VAL_Binary_DIR,
        config.VAL_LABEL_DIR,
        config.VAL_REFLECTANCE_DIR,

        config.TEST_Lidar_DIR, 
        #config.TEST_RGB_DIR, 
        #config.TEST_COLOR_DIR ,
        config.TEST_Intensity_DIR,
        config.TEST_Incidence_DIR,
        config.TEST_Binary_DIR,
        config.TEST_LABEL_DIR,
        config.TEST_REFLECTANCE_DIR,

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

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("/DATA2/Vivek/Code/Implementation/UNET_T3/Phase_3/my_checkpoint.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []

    # Create a DataFrame to hold your data
    df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Validation Loss'])
    best_val_loss = float('inf')  # start with a high loss

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1} of {config.NUM_EPOCHS}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        val_loss = val_fn(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_losses[-1]}")
        print(f"Validation loss: {val_losses[-1]}")

        # Append the losses for this epoch to the DataFrame
        new_row = pd.DataFrame({'Epoch': [epoch+1], 'Train Loss': [train_loss], 'Validation Loss': [val_loss]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(config.Loss_CSV_Path, index=False) #Change Accordingly


        # save model if validation loss has decreased
        if val_loss < best_val_loss:
            print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(best_val_loss, val_loss))
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, config.Checkpoint_Path)
            best_val_loss = val_loss

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        #save_checkpoint(checkpoint,'my_checkpoint.pth.tar')

      
    # Test the model after training is done
    test_loss = test_fn(test_loader, model, loss_fn)
    print(f"Test loss: {test_loss}")

    # After all epochs are completed
    plot_losses(train_losses, val_losses, config.Loss_Plot_Path)  # Call the function to plot the losses
    



if __name__ == "__main__":
    main()
