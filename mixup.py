import  numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    # '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def again_mixup_data(inputs_1, inputs_2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    inputs = lam * inputs_1 + (1 - lam) * inputs_2
    return inputs, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion_twomixup(criterion, pred, y_a, y_b, y_a_2, y_b_2, lam1, lam2, lam):
    return ((lam1 * criterion(pred, y_a) + (1 - lam1) * criterion(pred, y_b))*lam)+\
     ((lam2 * criterion(pred, y_a_2) + (1 - lam2) * criterion(pred, y_b_2))*(1 - lam))
