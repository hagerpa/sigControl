"""
This code is used to train and evaluate several test cases on a server.
The numerical results from the paper were generated with this code.
It is very technical and not necessary for understanding the rest of the code.
Therefore, it is also not documented.
"""

import pickle
import time as pytime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch.optim

from src.rde import RDE
from src.strategies import *

def build_rde(rde_model, strat, N, sig_comp, space, **kwargs):
    return RDE(N=N, rde_model=rde_model, strat=strat, sig_comp=sig_comp, space=space)


def create_optimizer_(strat__, optim, learning_rate, **kwargs):
    if optim == "Adam":
        optim_class = torch.optim.Adam
    elif optim == "Adagrad":
        optim_class = torch.optim.Adagrad
    elif optim == "LBFGS":
        optim_class = torch.optim.LBFGS
    else:
        raise RuntimeError("Unknown optimizer {}".format(optim))

    opt_args = optim_class.__init__.__code__.co_varnames
    opt_kwargs = {k_: kwargs["optim_" + k_] for k_ in opt_args if ("optim_" + k_) in kwargs}
    return optim_class(strat__.parameters(), lr=learning_rate, **opt_kwargs)


def calculate_sizes(sig_dim, dscrt, dscrt_train, batch_size, n_batches, validation_size, max_gb, MC_, **params):
    # Set DATA_MAX so that arrays will not be larger than MAX_GB GB
    data_max_ = max_gb * 1024 * 1024 * 1024 // 8  # GB to Bytes, divided by 8 Bytes for a float64

    data_max_val_ = data_max_ // ((5 * dscrt + sig_dim) * 4)  # [X, Y, U, payoff]_{t=0,...T} + Sig_t
    val_size_ = min(data_max_val_, validation_size)

    batch_size_ = batch_size
    # [X, Y, U, payoff, Sig_t]_{t=0,...T}
    n_batches_max_ = data_max_ // (batch_size_ * (5 * sig_dim) * dscrt_train)
    n_batches_ = min(n_batches_max_, n_batches)

    data_max_test_ = min(data_max_val_, MC_)

    return batch_size_, n_batches_, val_size_, data_max_test_


def run(sys_args, JOBS, build_strategy, rde_model, loss_fn, new_batch, params, update_params, result_path,
        model_break_points=()):
    if len(sys_args) == 1:
        print("Choose a job number from:", flush=True)
        for (i, j) in enumerate(JOBS):
            print(i + 1, j)

    else:
        print(sys_args, flush=True)

        params = JOBS[int(sys_args[1]) - 1] | params

        results = []

        print("######  {}  ######".format(params), flush=True)

        # ## TRAINING ##

        time_train = 0
        time_val = 0
        best_loss = 1E42
        best_model = None

        for restart in range(params["restarts"]):  # Best of #RESTARTS training

            strat = build_strategy(**params)
            rde = build_rde(rde_model, strat, **params)
            optim = create_optimizer_(strat, **params)

            best_model = {
                "strat": strat.state_dict(),
                "optim": optim.state_dict()
            } if (restart == 0) else best_model

            BATCH_SIZE, N_BATCHES, VAL_SIZE, _ = calculate_sizes(rde.sig_dim, **params)
            T_train, X_train = new_batch(MC=BATCH_SIZE * N_BATCHES, steps=params["dscrt_train"], **params)
            T_val, X_val = new_batch(MC=VAL_SIZE, steps=params["dscrt_train"], **params)

            for epoch in range(params['epochs']):

                params = params | update_params(restart, epoch, optim, best_model)

                if ("sample_inside" in params) and params["sample_inside"]:
                    BATCH_SIZE, N_BATCHES, VAL_SIZE, _ = calculate_sizes(rde.sig_dim, **params)
                    T_train, X_train = new_batch(MC=BATCH_SIZE * N_BATCHES, steps=params["dscrt_train"], **params)
                    T_val, X_val = new_batch(MC=VAL_SIZE, steps=params["dscrt"], **params)

                if epoch in model_break_points:
                    print(" -/> \n -\> {} \n -/>".format(params))

                    strat = build_strategy(**params)
                    strat.load_state_dict(best_model["strat"])
                    rde = build_rde(rde_model, strat, **params)
                    optim = create_optimizer_(strat, **params)

                if True:  # (epoch % 1000 == 0) or (epoch == params['steps_per_restart']):
                    with torch.no_grad():
                        strat.eval()
                        time_check = pytime.time()

                        Y, U = rde.forward(X_val, initial=params["initial"])
                        loss, var = loss_fn(T_val, Y, U, X_val, **params)

                        time_check = pytime.time() - time_check
                        time_val += time_check

                        print("{}: {} ({}, {}) <= validation".format(epoch, loss.item(),
                                                                     np.sqrt(var.item() / params["validation_size"]),
                                                                     VAL_SIZE), flush=True)

                    if best_loss > loss:
                        best_loss = loss.detach()
                        best_model = {
                            "strat": deepcopy(strat.state_dict()),
                            "optim": deepcopy(optim.state_dict())
                        }

                    if epoch == params["steps_per_restart"]:
                        if restart == params["restarts"] - 1:
                            strat = build_strategy(**params)
                            strat.load_state_dict(best_model["strat"])

                            optim = create_optimizer_(strat, **params)
                            optim.load_state_dict(best_model["optim"])

                            rde = build_rde(rde_model, strat, **params)
                        else:
                            break
                    elif epoch == params['epochs'] - 1:
                        break

                strat.train()
                time_check = pytime.time()

                for i in range(N_BATCHES):
                    #T_batch = T_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                    X_batch = X_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

                    def closure():
                        if torch.is_grad_enabled():
                            optim.zero_grad()
                        __Y, __U = rde.forward(X_batch, initial=params["initial"])
                        __loss, _ = loss_fn(T_train, __Y, __U, X_batch, **params)
                        if __loss.requires_grad:
                            __loss.backward()
                        return __loss

                    optim.step(closure)

                time_check = pytime.time() - time_check
                time_train += time_check

                # with torch.no_grad():
                #    Y, U = rde.forward(X, initial=params["initial"])
                #    loss, var = loss_fn(time, Y, U, X, **params)

                # print("{}: {} ({}, {})".format(epoch, loss.item(), np.sqrt(var.item() / BATCH_SIZE),
                #                               BATCH_SIZE), flush=True)

        strat = build_strategy(**params)
        strat.load_state_dict(best_model["strat"])
        strat.eval()
        rde = build_rde(rde_model, strat, **params)

        print("-- training ended ({}s for training, {}s for validation)--".format(time_train, time_val), flush=True)

        # ## RESAMPLING ##

        _, _, _, DATA_MAX_TEST = calculate_sizes(rde.sig_dim, **params)
        M = params["MC_"] // DATA_MAX_TEST
        eval_time = 0
        r_ = 0
        v_ = 0

        print("-- begin resampling in batches --", flush=True)
        print("{} / {}".format(0, M), flush=True)
        for k in range(M):
            time_check = pytime.time()

            with torch.no_grad():
                time, X = new_batch(MC=DATA_MAX_TEST, steps=params["dscrt"], **params)
                Y, U = rde.forward(X, initial=params["initial"])
                l__, v__ = loss_fn(time, Y, U, X, **params)
                r_ += l__.detach().numpy() / M
                v_ += v__.detach().numpy() / M

            time_check = pytime.time() - time_check
            eval_time += time_check

            print("{} / {} ({}s)".format(k + 1, M, time_check), flush=True)

        new_res = params | {
            "res": r_,
            "err": np.sqrt(v_ / (M * DATA_MAX_TEST)),
            "time": pd.Timestamp(pytime.ctime(pytime.time())),
            "train_time": time_train,
            "eval_time": eval_time
        }
        print(new_res, flush=True)
        results += [new_res]

        pd.DataFrame(results).to_pickle(result_path + "benchmarks/{}.pkl".format(int(sys_args[1])))

        if "save_model" in params and params["save_model"]:
            with open("./saved_models/" + "{}.pkl".format(int(sys_args[1])), 'wb') as file:
                mod = {
                    "meta": params,
                    "model": best_model["strat"]
                }
                pickle.dump(mod, file)
