from torch.optim.optimizer import Optimizer, required
import torch
from typing import List, Optional
from torch import Tensor
import numpy as np
from copy import deepcopy


def snnsgd(params: List[Tensor],
           d_p_list: List[Tensor],
           lr: float,
           dampening: float,
           with_ls: bool,
           ls_temp: float,
           ls_list: List[Optional[Tensor]],
           with_baseline: bool,
           baseline_num_list: List[Optional[Tensor]],
           baseline_den_list: List[Optional[Tensor]]):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]

        if with_ls:
            ls = ls_list[i]
            if ls is None:
                ls = ls_temp
            else:
                ls.mul_(dampening).add_(ls_temp, alpha=1 - dampening)
            ls_list[i] = ls

            if with_baseline:
                baseline_num = baseline_num_list[i]
                if baseline_num is None:
                    baseline_num = d_p.pow(2).mul(ls)
                else:
                    baseline_num.mul_(dampening).add_(d_p.pow(2).mul(ls), alpha=1 - dampening)
                baseline_num_list[i] = baseline_num

                baseline_den = baseline_den_list[i]
                if baseline_den is None:
                    baseline_den = d_p.pow(2)
                else:
                    baseline_den.mul_(dampening).add_(d_p.pow(2), alpha=1 - dampening)
                baseline_den_list[i] = baseline_den

                baseline = baseline_num_list[i] / (1e-7 + baseline_den_list[i])
            else:
                baseline = 0

            d_p = d_p.mul(ls - baseline)

        param.add_(d_p, alpha=-lr)




class SNNSGD(Optimizer):
    def __init__(self, params, lr=required, dampening=0.01, ls=True, baseline=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, ls=ls, dampening=dampening, baseline=baseline)
        super(SNNSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SNNSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, ls_temp=None, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            dampening = group['dampening']
            lr = group['lr']
            with_ls = group['ls']
            with_baseline = group['baseline']
            baseline_num_list = []
            baseline_den_list = []
            ls_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if with_baseline:
                        if 'baseline_num' not in state:
                            baseline_num_list.append(None)
                        else:
                            baseline_num_list.append(state['baseline_num'])

                        if 'baseline_den' not in state:
                            baseline_den_list.append(None)
                        else:
                            baseline_den_list.append(state['baseline_den'])

                    if with_ls:
                        if 'ls' not in state:
                            ls_list.append(None)
                        else:
                            ls_list.append(state['ls'])

            snnsgd(params_with_grad,
                   d_p_list,
                   lr,
                   dampening,
                   with_ls,
                   ls_temp,
                   ls_list,
                   with_baseline,
                   baseline_num_list,
                   baseline_den_list)

            # update baselines and ls in state
            for p, ls, baseline_num, baseline_den in zip(params_with_grad, ls_list, baseline_num_list, baseline_den_list):
                state = self.state[p]
                state['ls'] = ls
                state['baseline_num'] = baseline_num
                state['baseline_den'] = baseline_den

        return loss


def snnadam(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            exp_avg_sqs: List[Tensor],
            max_exp_avg_sqs: List[Tensor],
            state_steps: List[int],
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
            dampening: float,
            with_ls: bool,
            ls_temp: float,
            ls_list: List[Optional[Tensor]],
            with_baseline: bool,
            baseline_num_list: List[Optional[Tensor]],
            baseline_den_list: List[Optional[Tensor]]):

    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if with_ls:
            ls = ls_list[i]
            ls.mul_(dampening).add_(ls_temp, alpha=1 - dampening)

            if with_baseline:
                baseline_num = baseline_num_list[i]
                baseline_num.mul_(dampening).add_(grad.pow(2).mul(ls), alpha=1 - dampening)

                baseline_den = baseline_den_list[i]
                baseline_den.mul_(dampening).add_(grad.pow(2), alpha=1 - dampening)

                baseline = baseline_num / (1e-7 + baseline_den)
            else:
                baseline = 0

            grad.mul_(ls - baseline)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class SNNAdam(Optimizer):
    def __init__(self, params, dampening=0.01, ls=True, baseline=True, weight_decay=0., lr=required, betas=(0.9, 0.999), eps=1e-8, device=required, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, dampening=dampening, ls=ls, baseline=baseline, weight_decay=weight_decay, amsgrad=amsgrad)
        super(GaussianBayesAdam, self).__init__(params, defaults, device)

    @torch.no_grad()
    def step(self, ls_temp=None):
        ## Only for mean for now
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            baseline_num_list = []
            baseline_den_list = []
            ls_list = []

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Moving average of learning signal
                        state['ls'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Baseline to reduce variance of gradients
                        state['baseline_num'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['baseline_den'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if group['baseline']:
                        baseline_num_list.append(state['baseline_num'])
                        baseline_den_list.append(state['baseline_den'])
                    if group['ls']:
                        ls_list.append(state['ls'])
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            snnadam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group['amsgrad'],
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    dampening=group['dampening'],
                    with_ls=group['ls'],
                    ls_temp=ls_temp,
                    ls_list=ls_list,
                    with_baseline=group['baseline'],
                    baseline_num_list=baseline_num_list,
                    baseline_den_list=baseline_den_list)
