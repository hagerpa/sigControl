import sys

from server.run_environment import run

result_path += "twap_"

RESTARTS = 1

params = params | {
    "epochs": 1,
    "batch_size": 1,
}

JOBS = [
    {"dscrt_train": dscrt, "dscrt": dscrt, "N": N, "H": H, "space": space}
    for dscrt in [100, 1000, 10_000]
    for N in [1]
    for H in [1 / 4, 1 / 3, 1 / 2, 0.7, 0.9, 1.0]
    for space in ['log']
]


def build_strategy(N, space, sig_comp, **kwargs):
    return TimeWeightedAverage(y0=kwargs['initial'], T=1.0, kappa=kwargs['kappa'], kappa_T=kwargs["kappa_T"])


if __name__ == "__main__":
    run(sys.argv, JOBS, build_strategy, rde_model, loss_fn, new_batch, params, update_params, result_path)
