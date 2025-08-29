import torch
import torch.nn as nn
from setting import get_setting



args = get_setting()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        if hasattr(m, 'weight_ih_l0') and m.weight_ih_l0 is not None:
            nn.init.orthogonal_(m.weight_ih_l0)
        if hasattr(m, 'weight_hh_l0') and m.weight_hh_l0 is not None:
            nn.init.orthogonal_(m.weight_hh_l0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


def get_learning_rate(sub_network_name, args):
    lr_dict = {
        'Ens': args.lr_Ens,
        'Ent': args.lr_Ent,
        'Dec': args.lr_Dec,
        'Cls': args.lr_Cls,
        'Dis': args.lr_Dis,
        'Fsh': args.lr_Dis
    }
    return lr_dict.get(sub_network_name, 0.001)


def exp_lr_scheduler(optimizer, epoch, num_epochs, alpha=10, beta=0.75):
 
    """Applies Adaptive Learning Rate based on training progress"""
    
    p = epoch / num_epochs
    initial_lr = optimizer.defaults['lr']
    lr = initial_lr * (1 + alpha * p) ** -beta
    #lr = max(lr, args.min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Beta0_scheduler(epoch, num_epochs, u=0.1):
    p = epoch / num_epochs
    p_tensor = torch.tensor(p)
    beta0 = (2 * u) / (1 + torch.exp(-p_tensor)) - (u)
    return beta0.item()

def Beta2_scheduler(epoch, N, beta2_min=0.01):
    if epoch <= N/2:
        return 0.5 - (epoch / (N/2)) * (0.5 - beta2_min)
    else:
        return beta2_min
