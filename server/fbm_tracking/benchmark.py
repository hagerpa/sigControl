import sys
import numpy as np
from numpy import cosh, sinh
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma, betainc
import pandas as pd

T = 1.0
beta_inc = betainc

t_span = np.concatenate([
    np.linspace(0, 0.1, 100, endpoint=False),
    np.linspace(0.1, 0.9, 100, endpoint=False),
    np.linspace(0.9, 1.0, 100),
])


def benchmark(H, kappa=0.1):
    tau = lambda t: (T - t) / np.sqrt(kappa)

    c2_H = 2 * H * gamma(3 / 2 - H) / (gamma(H + 1 / 2) * gamma(2 - 2 * H))
    c_H = np.sqrt(c2_H)

    def z(t, s):
        r = s ** (H - 1 / 2) * H / c_H
        r *= 1 - beta_inc(1 - 2 * H, H + 1 / 2, s / t)
        r += c_H * (t / s) ** (H - 1 / 2) * (t - s) ** (H - 1 / 2)
        return r

    def z_prime(t, s):
        return c_H * (H - 0.5) * s ** (0.5 - H) * t ** (H - 0.5) * (t - s) ** (H - 1.5)

    def z_(t, s):
        f = lambda u: u ** (H - 3 / 2) * (u - s) ** (H - 0.5)
        res = quad(f, s, t, points=[s, t], full_output=1)[0]
        res *= - (H - 0.5) * s ** (0.5 - H)
        res += (t / s) ** (H - 0.5) * (t - s) ** (H - 1 / 2)
        res *= c_H
        return res

    if H > 0.5:
        z = z_
    else:
        for _ in range(100):
            t = np.random.uniform()
            s = np.random.uniform(0, t)
            assert np.allclose(z(t, s), z_(t, s)), "{}, {}".format(z(t, s), z_(t, s))

    def alpha_raw(t, s):
        f = lambda u: z(u, s) * cosh(tau(u))
        return quad(f, t, T, points=[t, T], full_output=1)[0]

    def overline_alpha_raw(t, s):
        f = lambda u: z_prime(u, s) * sinh(tau(u))
        return quad(f, t, T, points=[t, T], full_output=1)[0]

    def beta_raw(t):
        if t == 0:
            return 0
        else:
            f = lambda s: overline_alpha_raw(t, s) ** 2
            return quad(f, 0, t, points=[0, t], full_output=1)[0]

    vs = np.array([beta_raw(t) for t in t_span])
    beta = PchipInterpolator(t_span, vs)

    # Squared distance term
    sq_f = lambda t: beta(t) / sinh(tau(t)) ** 2
    sq_dist = 0.5 * quad(sq_f, 0, T, points=[0, T], full_output=1)[0]

    # QV term
    qv_f = lambda s: alpha_raw(s, s) ** 2 / (np.sqrt(kappa) * sinh(tau(s)) * cosh(tau(s)))
    qv_term = 0.5 * quad(qv_f, 0, T)[0]

    return qv_term, sq_dist


H_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [1 / 16, 1 / 8, 1 / 4, 1 / 3, 3 / 4]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Choose a job number from:", flush=True)
        for (i, j) in enumerate(H_space):
            print(i + 1, j)
    else:
        H_ = H_space[int(sys.argv[1]) - 1]
        qv, sq = benchmark(H_)
        res = [{"H": H_, "res": qv + sq, "qv": qv, "sq": sq}]
        pd.DataFrame(res).to_pickle("bench_{}.pkl".format(int(sys.argv[1])))
