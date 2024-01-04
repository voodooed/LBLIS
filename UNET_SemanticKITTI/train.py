import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):  # train_fn is going to do one epoch of training
    loop = tqdm(loader) # For progress bar
    losses = []
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):

        data = data.to(device=DEVICE)
        #targets = targets.unsqueeze(1).to(device=DEVICE)  #unsqueeze(1) for adding channel dimension
        targets = targets.to(device=DEVICE)

        # The mask is included in the inputs, so we need to separate it out
        mask = data[:, 0:1, :, :]
        
        
        # forward
        with torch.cuda.amp.autocast(): #Float 16 training
            predictions = model(data)
            loss = loss_fn(predictions, targets, mask=mask)
            #loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop - Shows the loss function so far
        #losses.append(loss.item())
        #loop.set_postfix(loss=loss.item())
        
        # calculate running loss
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

        loop.set_postfix(loss=running_loss)

    return running_loss
    #return sum(losses) / len(losses)

def val_fn(loader, model, loss_fn):  
    loop = tqdm(loader) 
    losses = []

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            #targets = targets.unsqueeze(1).to(device=DEVICE)
            mask = data[:, 0:1, :, :]

            predictions = model(data)
            loss = loss_fn(predictions, targets, mask=mask)
            #loss = loss_fn(predictions, targets)
            #losses.append(loss.item())
            #loop.set_postfix(loss=loss.item())

            running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

            loop.set_postfix(loss=running_loss)
    
    model.train()  # Set the model back to training mode

    return running_loss
    #return sum(losses) / len(losses) 



def test_fn(loader, model, loss_fn):
    loop = tqdm(loader)
    losses = []
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            mask = data[:, 0:1, :, :]

            predictions = model(data)
            loss = loss_fn(predictions, targets, mask=mask)

            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

           

            loop.set_postfix(loss=running_loss)

            

    # No need to switch back to train mode since there will be no more training after testing
    # model.train() 

    return (sum(losses) / len(losses))
