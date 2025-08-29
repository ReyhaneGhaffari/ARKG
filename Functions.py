import torch
import torch.nn as nn
from torch.autograd import Function
from setting import get_setting
import torch.nn.functional as F


args = get_setting()


def inverseDecayScheduler(step, initial_lambda=1.0, gamma=10, power=0.75, max_iter=1000):
    step = min(step, max_iter)
    return initial_lambda * ((1 + gamma * step / max_iter) ** (-power))
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(torch.tensor(lambd))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0].item()
        return grad_output.neg() * lambd, None
     

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)



class GradientReverseModule(nn.Module):  
    def __init__(self, args):  
        super(GradientReverseModule, self).__init__()
        self.max_steps = args.max_steps  
        self.initial_lambda = 1.0
        self.gamma = 10
        self.power = 0.75

    def forward(self, x, global_step):  
     
        lambd = inverseDecayScheduler(global_step, self.initial_lambda, self.gamma, self.power, self.max_steps)
     
        return grad_reverse(x, lambd)




def kl_divergence(mu_p, var_p, mu_q, var_q):
    """
    Computes the KL divergence between two multivariate diagonal Gaussian distributions:
        KL( N(mu_p, var_p) || N(mu_q, var_q) )

    Parameters:
        mu_p (Tensor): Mean of the first distribution (typically encoder output)
        var_p (Tensor): Variance of the first distribution
        mu_q (Tensor): Mean of the second distribution (typically standard normal or prior)
        var_q (Tensor): Variance of the second distribution

    Returns:
        Tensor: KL divergence per element (same shape as input tensors)
    """
    sigma_p = torch.sqrt(var_p)
    sigma_q = torch.sqrt(var_q)
    epsilon = 1e-10
    kl = torch.log(sigma_q + epsilon) - torch.log(sigma_p + epsilon) + (var_p + (mu_p - mu_q) ** 2) / (2 * var_q) - 0.5
 
    return kl.sum(dim=1)




class NLLNormal(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, var=1, weight=1):
        if not torch.is_tensor(var):
            var = torch.tensor(var, dtype=pred.dtype, device=pred.device)
        var = torch.clamp(var, min=self.eps)
        const_term = torch.log(torch.tensor(2.0 * torch.pi, dtype=pred.dtype, device=pred.device))
        if isinstance(weight, torch.Tensor):
            while weight.dim() < pred.dim():
                weight = weight.unsqueeze(-1)
        nll = (0.5 * (const_term + torch.log(var)) + 0.5 * ((pred - target) ** 2) / var) * weight
        if self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'mean':
            return nll.mean()
        else:
            return nll

def weight_loglike(pred, target, weight, args):
    if args.recon_loss_type == 'mse':
        loss = F.mse_loss(pred, target, reduction='none').to(args.device)
        loss = loss.view(loss.size(0), -1).mean(dim=1) * weight
        recon_loss = loss.mean().to(args.device)
    elif args.recon_loss_type == 'nll':
        loss_fn = NLLNormal().to(args.device)
        recon_loss = loss_fn(pred, target, var=1, weight=weight)
    return recon_loss


def threshold_selection(y_true, probabilities, rho):  
    if len(y_true) == 0:
        return 0   
    predicted_labels, max_indices = torch.max(probabilities, dim=1)
    y_true_indices = torch.argmax(y_true, dim=1)
    if max_indices.size(0) != y_true_indices.size(0):
        raise ValueError(f"Size mismatch: max_indices size {max_indices.size(0)} vs y_true size {y_true_indices.size(0)}")
    correct_predictions = (max_indices == y_true_indices).sum().item() 
    A = correct_predictions / (y_true_indices.size(0))
    A = torch.tensor([A],   dtype=torch.float64)
    rho=torch.tensor([rho], dtype=torch.float64)
    ex = torch.exp((-(rho) * A))
    TC = 1 / (1 +ex )  
    return TC




def softmax_to_hard_labels_with_one_topk(soft_output):
    _, indices = torch.topk(soft_output, 1, dim=1)
    hard_labels = torch.zeros_like(soft_output)
    hard_labels.scatter_(1, indices, 1)
    return hard_labels



def HCS(zt, rimgt, predicted_probabilities, TC, args):  
    num_classes = args.num_class
    mask = predicted_probabilities >= TC
    valid_indices = mask.any(dim=1)
    selected_samples = rimgt[valid_indices]
    selected_features = zt[valid_indices]
    selected_probabilities = predicted_probabilities[valid_indices]
    predicted_classes = torch.argmax(selected_probabilities, dim=1)
    known_mask = predicted_classes < (num_classes - 1)
    known_samples = selected_samples[known_mask]
    known_features = selected_features[known_mask]
    known_labels = selected_probabilities[known_mask]  # <== soft label
    unknown_mask = predicted_classes == (num_classes - 1)
    unknown_samples = selected_samples[unknown_mask]
    unknown_features = selected_features[unknown_mask]
    unknown_labels = selected_probabilities[unknown_mask]  # <== soft label
    return known_samples, known_features, known_labels, unknown_samples, unknown_features, unknown_labels
