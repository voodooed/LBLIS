import torch
import torch.nn as nn
import torch
import torch.nn.functional as F




class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, predicted, target, mask=None, color_mask=None):
        # Compute element-wise squared difference
        squared_diff = torch.pow(predicted - target, 2)

        # Default to a mask of ones if no mask is provided
        if mask is None:
            mask = torch.ones_like(target)

       
        # Convert mask to boolean
        #mask = mask >= 0
        mask = mask.bool()

        # Error checking: ensure that mask only contains 0s and 1s
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Mask must only contain 0s and 1s")
        

        # If rgb_mask is provided, convert it to boolean and combine with mask
        if color_mask is not None:
            color_mask = color_mask >= 0
            full_mask = mask & color_mask
        else:
            full_mask = mask

        # Apply the mask
        masked_squared_diff = squared_diff * full_mask

        # Compute the sum of the squared differences
        sum_squared_diff = torch.sum(masked_squared_diff)

        # Compute the mean of the loss function
        loss = sum_squared_diff / torch.sum(full_mask)

        return loss
      

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, predicted, target):
        # Compute element-wise squared difference
        squared_diff = torch.pow(predicted - target, 2)

        # Compute the sum of the squared differences
        sum_squared_diff = torch.sum(squared_diff)

        # Compute the mean of the loss function
        loss = sum_squared_diff / predicted.numel()

        return loss

class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, predicted, target, mask=None, color_mask=None):
        # Compute element-wise absolute difference
        abs_diff = torch.abs(predicted - target)

        # Default to a mask of ones if no mask is provided
        if mask is None:
            mask = torch.ones_like(target)

        # Convert mask to boolean
        mask = mask.bool()

        # Error checking: ensure that mask only contains 0s and 1s
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Mask must only contain 0s and 1s")

        # If rgb_mask is provided, convert it to boolean and combine with mask
        if color_mask is not None:
            color_mask = color_mask.bool()  # Convert color_mask to boolean
            full_mask = mask & color_mask
        else:
            full_mask = mask

        # Apply the mask
        masked_abs_diff = abs_diff * full_mask

        # Compute the sum of the absolute differences
        sum_abs_diff = torch.sum(masked_abs_diff)

        # Compute the mean of the loss function
        loss = sum_abs_diff / torch.sum(full_mask)

        return loss

    
class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, predicted, target,mask=None):
        # Compute element-wise absolute difference
        abs_diff = torch.abs(predicted - target)

        # Compute the mean of the absolute differences
        loss = torch.mean(abs_diff)

        return loss
