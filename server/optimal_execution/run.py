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
SIGMA = 0.02

result_path = "./results_server/" if SERVER else "./results_local/"

jobs = dict()

if SERVER:
    validation_size = 1024 * 16 * 8

    MC_ = 2**22
    MAX_GB = 128
else:
    validation_size = 1024 * 8

    MC_ = validation_size
    MAX_GB = 8

RESTARTS = 2

params = {
    "kappa": KAPPA,
    "kappa_T": KAPPA_T,
    "sigma": SIGMA,
    "initial": INITIAL,
    "constraint": CONSTRAINT,

    "restarts": RESTARTS,
    "validation_size": validation_size,

    "optim": "Adam",
    "learning_rate": 0.1,
    "batch_size": 512,
    "n_batches": 2 ** 10,
    "steps_per_restart": 3,
    "epochs": 31,

    #"optim": "LBFGS",
    #"optim_history_size": 40,
    #"optim_line_search_fn": "strong_wolfe",
    #"optim_tolerance_change": 0.0,
    #"optim_max_iter": 100,
    #"learning_rate": 1.0,
    #"n_batches": 1,
    #"batch_size": 1024 * 16 * 8,
    #"steps_per_restart": 1,
    #"epochs": 2,

    "max_gb": MAX_GB,
    "MC_": MC_,
    "sig_comp": 'tX',

    "save_model": True,
}


def update_params(restarts_, epochs_, optim, best_model):
    return {}


JOBS = [
    {"dscrt_train": dscrt, "dscrt": dscrt, "N": N, "H": H, **mod_,
     "epochs": 31 + 5*N #Remove when using LBFGS
     }
    for dscrt in [100]
    for N in [1, 2, 3, 4, 5]
    for H in [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 3, 1.0 / 2, 3.0 / 4, 7.0 / 8, 1.0]
    for mod_ in [
        {"space": 'log', "nn_hidden": 2},
        {"space": 'sig', "nn_hidden": 0}
    ]
]

pfs = {
    "time": lambda time_, path_: time_,
    "path": lambda time_, path_: path_,
}


def rde_model(Y, U, dX):
    return - U * dX[:, 0]


from server.fbm_tracking.run import new_batch

TWAP = 1.0 * INITIAL - INITIAL ** 2 * KAPPA * KAPPA_T / (KAPPA + 1.0 * KAPPA_T)


# alpha = INITIAL * 1.0 * KAPPA_T / (KAPPA + 1.0 * KAPPA_T)

# noinspection PyUnreachableCode
def loss_fn(time, Y, U, X, **kwargs):
    if False:  # alternative expression of cost functional
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
