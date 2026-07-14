import math
import torch
import torch.nn as nn
import numpy as np


def Icesfp_loss(input_vec, target_vec, confidence_map, aolp, mask_tensor=None, reduction='sum', device=None):
    new_input = input_vec
    cosine_loss = loss_fn_cosine(new_input, target_vec, mask_tensor=mask_tensor, reduction=reduction, device=device)
    aolp_loss = loss_aolp(new_input, aolp, mask_tensor, confidence_map)
    loss = cosine_loss + aolp_loss * 0.05
    return loss


def loss_fn_cosine(input_vec, target_vec, mask_tensor, reduction='sum', device=None):
    mask_invalid_pixels = torch.all(mask_tensor < 255, dim=1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-06)
    loss_cos = 1.0 - cos(input_vec, target_vec)
    loss_cos[mask_invalid_pixels] = 0.0
    loss_cos_sum = loss_cos.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum()
    error_sum = loss_cos_sum
    error_output = error_sum / total_valid_pixels
    if reduction == 'elementwise_mean':
        loss_cos = error_output
    elif reduction == 'sum':
        loss_cos = error_sum
    elif reduction == 'none':
        loss_cos = loss_cos
    else:
        raise Exception("Invalid value for reduction  parameter passed. Please use 'elementwise_mean' or 'none'".format())
    return loss_cos


def loss_aolp(input_vec, aolp, mask_tensor, confidence_map):
    confidence_map = confidence_map.squeeze(1)
    aolp = aolp.squeeze(1) * math.pi
    aolp_0 = aolp + math.pi / 2
    aolp_1 = aolp - math.pi / 2
    aolp_0 = torch.remainder(aolp_0, math.pi * 2)
    aolp_1 = torch.remainder(aolp_1, math.pi * 2)
    mask_invalid_pixels = torch.all(mask_tensor < 255, dim=1)
    y = input_vec[:, 1, :, :]
    x = input_vec[:, 0, :, :]
    phi = torch.atan2(y, x)
    phi = torch.remainder(phi, math.pi * 2)
    error_0 = torch.min(torch.abs(phi - aolp_0), math.pi * 2 - torch.abs(phi - aolp_0))
    error_1 = torch.min(torch.abs(phi - aolp_1), math.pi * 2 - torch.abs(phi - aolp_1))
    error = torch.min(error_0, error_1)
    error = error * confidence_map
    error[mask_invalid_pixels] = 0.0
    loss = torch.sum(error)
    total_valid_pixels = (~mask_invalid_pixels).sum()
    return loss


def metric_calculator_batch(input_vec, target_vec, mask=None):
    if len(input_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))
    INVALID_PIXEL_VALUE = -1 / np.sqrt(3) + 0.0001
    mask_valid_pixels = ~torch.all(target_vec < INVALID_PIXEL_VALUE, dim=1)
    total_valid_pixels = mask_valid_pixels.sum()
    if total_valid_pixels == 0:
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return (torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels)
    cos = nn.CosineSimilarity(dim=1, eps=1e-06)
    loss_cos = cos(input_vec, target_vec)
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, -1.0 + eps, 1.0 - eps)
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)
    loss_deg = loss_deg[mask_valid_pixels.bool()]
    temp = torch.min(loss_deg)
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()
    percentage_1 = (loss_deg < 11.25).sum().float() / total_valid_pixels * 100
    percentage_2 = (loss_deg < 22.5).sum().float() / total_valid_pixels * 100
    percentage_3 = (loss_deg < 30).sum().float() / total_valid_pixels * 100
    return (loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3)


def metric_calculator(input_vec, target_vec, mask=None):
    if len(input_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(target_vec.shape))
    INVALID_PIXEL_VALUE = 0
    mask_valid_pixels = ~torch.all(target_vec == INVALID_PIXEL_VALUE, dim=0)
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    if total_valid_pixels == 0:
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return (torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels)
    cos = nn.CosineSimilarity(dim=0, eps=1e-06)
    loss_cos = cos(input_vec, target_vec)
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, -1.0 + eps, 1.0 - eps)
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)
    temp = loss_deg[0, :, :, :]
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()
    percentage_1 = (loss_deg < 11.25).sum().float() / total_valid_pixels * 100
    percentage_2 = (loss_deg < 22.5).sum().float() / total_valid_pixels * 100
    percentage_3 = (loss_deg < 30).sum().float() / total_valid_pixels * 100
    return (loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels)


def loss_fn_radians(input_vec, target_vec, reduction='sum'):
    cos = nn.CosineSimilarity(dim=1, eps=1e-06)
    loss_cos = cos(input_vec, target_vec)
    loss_rad = torch.acos(loss_cos)
    if reduction == 'elementwise_mean':
        loss_rad = torch.mean(loss_rad)
    elif reduction == 'sum':
        loss_rad = torch.sum(loss_rad)
    elif reduction == 'none':
        pass
    else:
        raise Exception("Invalid value for reduction  parameter passed. Please use 'elementwise_mean' or 'none'".format())
    return loss_rad


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    n, c, h, w = logit.shape
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32), ignore_index=ignore_index, reduction='sum')
    loss = criterion(logit, target.long())
    if batch_average:
        loss /= n
    return loss
