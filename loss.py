import torch

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    classification_loss *= 0.01
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - torch.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    return torch.mean(L_g * y_true_cls * training_mask) + classification_loss