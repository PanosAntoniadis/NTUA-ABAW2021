import math
import torch
import numpy as np
import itertools
import torch.nn.functional as F

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target.long())

def ccc(output, target):
    mean_target = torch.mean(target, axis=0)
    mean_output = torch.mean(output, axis=0)
    var_target = torch.var(target, axis=0)
    var_output = torch.var(output, axis=0)
    cor = torch.mean((output - mean_output) * (target - mean_target), axis=0)
    r = 2*cor / (var_target + var_output + (mean_target-mean_output)**2)
    ccc = sum(r)/2
    return 1 - ccc

def mean_squared_error(output, target):
    loss = F.mse_loss(output, target)
    return loss

