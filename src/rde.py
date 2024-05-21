from typing import Callable

import torch
import iisignature

from src.join_signatures import generate_joinsig_function

relu = torch.nn.functional.relu


class RDE(torch.nn.Module):
    def __init__(self, N, rde_model: Callable, strat: torch.nn.Module, sig_comp='tX', space='sig'):
        """
        A model layer that approximates trajectories of a controlled RDE
            dY_t = model(U_t, Y_t, dX_t)
        where the control U is a function of the truncated (log-) signature of the driver X or possibly also of (X,Y).
        :param N: Truncation level of the signature.
        :param rde_model: Vector filed of the RDE, i.e., dY_t = model(Y_t, U_t, dX_t).
        :param strat: Strategy as function of the signature U_t = strat(Sig(X)_t).
        :param sig_comp: Which components of the path (t,X,Y) to include in the signature.
        :param space: 'log' if strategy is a function of the log-signature. Default is space='sig'.
        """
        super().__init__()
        self.d = len(sig_comp)
        self.sig_comp = sig_comp
        self.sig_join, self.sig_dim = generate_joinsig_function(self.d, N, space)
        self.strat = strat
        self.s = iisignature.prepare(2, N, 'O')
        self.rde_model = rde_model

    def forward(self, X: torch.Tensor, initial=0.0):
        m, n, d = X.shape

        Y0 = initial * torch.ones(m, dtype=torch.float64)

        d = {
            't': torch.zeros(m, dtype=torch.float64),
            'X': torch.zeros((m, d), dtype=torch.float64),
            'Y': torch.zeros((m, 1), dtype=torch.float64)
        }

        dZ = torch.zeros(m, self.d, dtype=torch.float64, requires_grad=False)
        ZZ = torch.zeros(m, self.sig_dim, dtype=torch.float64, requires_grad=False)  # Hidden state log-signature
        Y = [Y0]
        U = []

        for i in range(n - 1):
            dX = X[:, i + 1] - X[:, i]
            d['t'] = dX[:, 0]
            d['X'] = dX[:, 1]
            u = torch.flatten(self.strat(ZZ))
            d['Y'] = self.rde_model(Y[-1], u, dX)

            Y += [Y[-1] + d['Y']]
            U += [u]

            grads = torch.is_grad_enabled()
            with torch.no_grad():
                torch.stack([d[k] for k in self.sig_comp], dim=1, out=dZ)
                if grads:
                    ZZ = self.sig_join.apply(ZZ, dZ)
                else:
                    ZZ[...] = self.sig_join.apply(ZZ, dZ)  # in re-simulation this saves memory allocation time

        U += [torch.flatten(self.strat(ZZ))]

        return torch.stack(Y, dim=1), torch.stack(U, dim=1)
