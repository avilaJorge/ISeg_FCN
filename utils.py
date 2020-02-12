import torch
from dataloader import *
import numpy as np
def iou(pred, target):
    ious = []
    cls_ious = []
    for cls in valid_ids:
        for i in range(pred.size()[0]):
        # Complete this function
            intersection = ((pred[i] == cls) & (target[i] == cls)).sum().float()
            union = ((pred[i] == cls) | (target[i] == cls)).sum().float()
            
            if union == 0:
                cls_ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
            # Append the calculated IoU to the list ious
                cls_ious.append(float(intersection / union))
        
        ious.append(np.nanmean(cls_ious))
    return ious


def pixel_acc(pred, target):
    #Complete this function
    acc_arr = []
    for i in range(pred.size()[0]):
        acc_arr.append(float((pred[i] == target[i]).sum().float() / pred[i].nelement()))
    return np.nanmean(acc_arr)