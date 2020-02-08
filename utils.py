def iou(pred, target):
    ious = []
    for cls in range(34):
        # Complete this function
        intersection = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            # Append the calculated IoU to the list ious
            ious.append(float(intersection / union))
    return ious


def pixel_acc(pred, target):
    #Complete this function
    return float((pred == target).sum() / pred.nelement())