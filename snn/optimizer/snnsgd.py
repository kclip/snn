from torch.optim.optimizer import Optimizer, required
import torch
from typing import List, Optional
from torch import Tensor


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
                    baseline_den = d_p.pow(2).mul(ls)
                else:
                    baseline_den.mul_(dampening).add_(d_p.pow(2).mul(ls), alpha=1 - dampening)
                baseline_den_list[i] = baseline_den

                baseline = baseline_num_list[i] / (1e-7 + baseline_den_list[i])
            else:
                baseline = 0

        else:
            ls = 1
            baseline = 0

        if ls is not None:
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
    def step(self, ls_temp, closure=None):
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
