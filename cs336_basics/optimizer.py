"""
Optimizers and learning rate schedulers.
"""

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer implementation following Loshchilov & Hutter (2019).
    
    Implements Algorithm 2 from "Decoupled Weight Decay Regularization".
    Key difference from Adam: weight decay is applied directly to parameters
    after the gradient-based update, rather than being added to the gradient.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get or initialize state
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # Update biased first moment estimate
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias correction terms
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute bias-corrected first moment estimate
                # m_hat_t = m_t / (1 - beta1^t)
                corrected_exp_avg = exp_avg / bias_correction1
                
                # Compute bias-corrected second raw moment estimate
                # v_hat_t = v_t / (1 - beta2^t)
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Update parameters (gradient-based update)
                # theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                p.addcdiv_(
                    corrected_exp_avg,
                    corrected_exp_avg_sq.sqrt().add_(eps),
                    value=-lr
                )
                
                # Apply weight decay (decoupled from gradient-based update)
                # theta_t = theta_t - lr * lambda * theta_t
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
        
        return loss


def get_cosine_schedule_with_warmup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute learning rate for cosine annealing schedule with linear warmup.
    
    Schedule:
        - For it < warmup_iters: linearly increase from 0 to max_learning_rate
        - For warmup_iters <= it <= cosine_cycle_iters: 
          cosine decay from max_learning_rate to min_learning_rate
        - For it > cosine_cycle_iters: constant at min_learning_rate
    
    Args:
        it: Current iteration number
        max_learning_rate: Maximum learning rate (alpha_max)
        min_learning_rate: Minimum learning rate (alpha_min)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total iterations for cosine cycle (T_c)
    
    Returns:
        Learning rate for the given iteration
    """
    # Warmup phase: linear increase from 0 to max_learning_rate
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # After cosine cycle: constant at min_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # Cosine annealing phase
    # Progress through cosine cycle (0 to 1)
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # Cosine decay from max to min
    # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
