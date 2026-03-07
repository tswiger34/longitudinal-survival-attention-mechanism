"""Loss functions intended to be used when performing survival analysis research using the LTSA model

Credit to
[mahmoodlab](https://github.com/mahmoodlab/MCAT/blob/b9cca63be83c67de7f95308d54a58f80b78b0da1/utils/utils.py) for
creating the python implementations of these loss functions and
[bionlplab](https://github.com/bionlplab/longitudinal_transformer_for_survival_analysis/blob/main/src/losses.py)
for identifying and aggregating the required functions to make this model work.

## Functions

- **nll_loss**:
- **ce_loss**:

## Classes

- **CrossEntropySurvLoss**:
- **NLLSurvLoss**:
- **CoxSurvLoss**:

"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from ty_extensions import Unknown


def nll_loss(
    hazards: Tensor,
    S: Tensor | None,
    Y: Tensor,
    c: Tensor,
    obs_times,
    beta: float = 0.15,
    eps: float = 1e-7,
):
    """_summary_

    S(-1) = 0, all patients are alive from (-inf, 0) by definition

    Args:
        hazards (Tensor): Tensor of hazard scores for each obs 1,2,...,k
        S (Tensor): Tensor of survival scores for obs 1,2,...,k, should be the cumulative product of `1 - hazards`
        Y (Tensor): The ground truth tensor of obs 1,2,...,k
        c (Tensor): Tensor of censorship statuses for obs 1,2,...,k, values should be either 0 or 1
        obs_times (_type_): _description_
        beta (float, optional): _description_. Defaults to 0.15.
        eps (float, optional): _description_. Defaults to 1e-7.

    Returns:
        Tensor: _description_
    """
    batch_size: int = len(Y)
    Y: Tensor = Y.view(batch_size, 1)
    c: Tensor = c.view(batch_size, 1).float()
    if S is None:
        S: Tensor = torch.cumprod(input=1 - hazards, dim=1)

    S_padded: Tensor = torch.cat(tensors=[torch.ones_like(input=c), S], dim=1)

    uncensored: Tensor = 1 - c
    uncensored_loss: Tensor = uncensored.neg() * (
        torch.log(input=torch.gather(input=S_padded, dim=1, index=Y).clamp(min=eps))
        + torch.log(input=torch.gather(input=hazards, dim=1, index=Y).clamp(min=eps))
    )
    censored_loss: Tensor = c.neg() * torch.log(
        input=torch.gather(input=S_padded, dim=1, index=Y + 1).clamp(min=eps)
    )
    neg_l: Tensor = censored_loss + uncensored_loss
    loss: Tensor = (1 - beta) * neg_l + beta * uncensored_loss
    loss: Tensor = loss.mean()
    return loss


def ce_loss(
    hazards: Tensor,
    S: Tensor | None,
    Y: Tensor,
    c: Tensor,
    obs_times=None,
    beta: float = 0.15,
    eps: float = 1e-7,
):
    """_summary_

    Args:
        hazards (Tensor): _description_
        S (Tensor): _description_
        Y (Tensor): _description_
        c (Tensor): _description_
        obs_times (_type_, optional): _description_. Defaults to None.
        beta (float, optional): _description_. Defaults to 0.15.
        eps (float, optional): _description_. Defaults to 1e-7.

    Returns:
        _type_: _description_
    """
    batch_size: int = len(Y)
    Y: Tensor = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c: Tensor = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S: Tensor = torch.cumprod(input=1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards

    S_padded: Tensor = torch.cat(tensors=[torch.ones_like(input=c), S], dim=1)

    reg: Unknown | Tensor = -(1 - c) * (
        torch.log(input=torch.gather(input=S_padded, dim=1, index=Y) + eps)
        + torch.log(input=torch.gather(input=hazards, dim=1, index=Y).clamp(min=eps))
    )
    ce_l: Unknown | Tensor = -c * torch.log(input=torch.gather(input=S, dim=1, index=Y).clamp(min=eps)) - (
        1 - c
    ) * torch.log(input=1 - torch.gather(input=S, dim=1, index=Y).clamp(min=eps))
    loss: Unknown | Tensor = (1 - beta) * ce_l + beta * reg
    loss: Unknown | Tensor = loss.mean()

    return loss


class CrossEntropySurvLoss(object):
    """Cross entropy survival loss object"""

    def __init__(self, beta=0.15):
        self.beta: Unknown | float = beta

    def __call__(self, hazards, S, Y, c, obs_times, beta=None):
        if beta is None:
            return ce_loss(hazards, S, Y, c, obs_times, beta=self.beta)
        else:
            return ce_loss(hazards, S, Y, c, obs_times, beta=beta)


class NLLSurvLoss(object):
    def __init__(self, beta: float = 0.15):
        self.beta: float = beta

    def __call__(self, hazards, S, Y, c, obs_times, beta=None):
        if beta is None:
            return nll_loss(hazards, S, Y, c, obs_times, beta=self.beta)
        else:
            return nll_loss(hazards, S, Y, c, obs_times, beta=beta)


class CoxSurvLoss(object):
    def __call__(hazards: torch.Tensor, S: torch.Tensor, Y, c, beta, device: torch.device | None, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data

        current_batch_len: int = len(S)
        R_mat: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.zeros(
            shape=[current_batch_len, current_batch_len], dtype=int
        )
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat: Tensor = torch.FloatTensor(R_mat).to(device)
        theta: Tensor = hazards.reshape(-1)
        exp_theta: Tensor = torch.exp(theta)
        loss_cox: Unknown | Tensor = -torch.mean(
            input=(theta - torch.log(input=torch.sum(input=exp_theta * R_mat, dim=1))) * (1 - c)
        )

        return loss_cox
