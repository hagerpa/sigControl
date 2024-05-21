import sys

import torch.optim

from server.fbm_tracking.run import build_strategy
from server.run_environment import run
from src.strategies import *

SERVER = True

KAPPA = 10 ** -3
KAPPA_T = 10 ** -1
CONSTRAINT = None  # (-1,1)
INITIAL = 1.0
SIGMA = 0.2

result_path = "./results_server/" if SERVER else "./results_local/"

jobs = dict()

learning_rate = 1

if SERVER:
    batch_size = 1024 * 16 * 8
    validation_size = 1024 * 16 * 16

    epochs = 2
    MC_ = validation_size
    MAX_GB = 128
else:
    batch_size = 1024
    validation_size = 1024 * 8

    epochs = 2
    MC_ = validation_size
    MAX_GB = 8

RESTARTS = 5

params = {
    "kappa": KAPPA,
    "kappa_T": KAPPA_T,
    "sigma": SIGMA,
    "initial": INITIAL,
    "constraint": CONSTRAINT,

    "restarts": RESTARTS,
    "steps_per_restart": 1,
    "epochs": epochs,
    "batch_size": batch_size,
    "validation_size": validation_size,

    "optim": "LBFGS",
    "learning_rate": learning_rate,
    "optim_history_size": 40,
    "optim_line_search_fn": "strong_wolfe",
    "optim_tolerance_change": 0.0,
    "optim_max_iter": 100,

    "max_gb": MAX_GB,
    "MC_": MC_,
    "sig_comp": 'tX',
}


def update_params(restarts_, epochs_, optim, best_model):
    if epochs_ == 30:
        return {
            "batch_size": batch_size * 1,
        }
    elif epochs_ == 60:
        return {
            "batch_size": batch_size * 1,
        }
    elif epochs_ == 70:
        return {
            "batch_size": batch_size * 1,
        }
    else:
        return {}


JOBS = [
    {"dscrt_train": dscrt, "dscrt": dscrt, "N": N, "H": H, "space": space}
    for dscrt in [100]
    for N in [1, 2, 3, 4]
    # for sig_comp in MODES
    for H in [1 / 4, 1 / 3, 1 / 2, 0.7, 0.9, 1.0]
    for space in ['log', 'sig']
]

pfs = {
    "time": lambda time_, path_: time_,
    "path": lambda time_, path_: path_,
}


def rde_model(Y, U, dX):
    return - U * dX[:, 0]


from examples import new_batch

TWAP = 1.0 * INITIAL - INITIAL ** 2 * KAPPA * KAPPA_T / (KAPPA + 1.0 * KAPPA_T)


# alpha = INITIAL * 1.0 * KAPPA_T / (KAPPA + 1.0 * KAPPA_T)

# noinspection PyUnreachableCode
def loss_fn(time, Y, U, X, **kwargs):
    if False:
        dP = SIGMA * (X[:, 1:, 1] - X[:, :-1, 1])
        gains = 1.0 + torch.sum(Y[:, :-1] * dP, dim=1)
    else:
        B = 1.0 + SIGMA * X[:, :, 1]
        gains = torch.mean(U[:, :-1] * B[:, :-1], dim=1) * (time[-1] - time[0]) + Y[:, -1] * B[:, -1]

    costs = KAPPA * torch.mean(U[:, :-1] ** 2, dim=1) * (time[-1] - time[0]) + KAPPA_T * Y[:, -1] ** 2
    loss = - 100 * (gains - costs - TWAP) / TWAP
    return torch.mean(loss), torch.var(loss)


if __name__ == "__main__":
    run(sys.argv, JOBS, build_strategy, rde_model, loss_fn, new_batch, params, update_params, result_path)
