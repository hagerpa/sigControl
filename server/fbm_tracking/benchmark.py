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
            t_ = np.random.uniform()
            s_ = np.random.uniform(0, t_)
            assert np.allclose(z(t_, s_), z_(t_, s_)), "{}, {}".format(z(t_, s_), z_(t_, s_))

    def alpha_raw(t, s):
        f = lambda u: z(u, s) * cosh(tau(u))
        return quad(f, t, T, points=[t, T], full_output=1)[0]

    def alpha_diag_raw(s):
        def f1(u):
            r = (H - 1 / 2) * (u / s) ** (H - 3 / 2) * cosh(tau(u)) / s
            r += - (u / s) ** (H - 1 / 2) * sinh(tau(u)) / np.sqrt(kappa)
            r *= c_H * (u - s) ** (H + 1 / 2) / (H + 1 / 2)
            return r

        a = c_H * (T / s) ** (H - 1 / 2) * (T - s) ** (H + 1 / 2) / (H + 1 / 2)
        def f2(u):
            r = s ** (H - 1 / 2) * H / c_H
            r *= 1 - beta_inc(1 - 2 * H, H + 1 / 2, s / u)
            return r * cosh(tau(u))
        def f(u):
            return - f1(u) + f2(u)
        return a + quad(f, s, T, points=[s, T], full_output=1)[0]

    def alpha_diag(s):
        if H >= 0.5:
            return alpha_raw(s,s)
        if H < 0.5:
            return alpha_diag_raw(s)

    def overline_alpha_raw(t, s):
        f = lambda u: z_prime(u, s) * sinh(tau(u))
        return quad(f, t, T, points=[t, T], full_output=1)[0]

    def beta_raw(t):
        if t == 0:
            return 0
        else:
            f = lambda s: overline_alpha_raw(t, s) ** 2
            return quad(f, 0, t, points=[0, t], full_output=1)[0]

    vs = np.array([beta_raw(t_) for t_ in t_span])
    beta = PchipInterpolator(t_span, vs)

    # Squared distance term
    sq_f = lambda t: beta(t) / sinh(tau(t)) ** 2
    sq_dist = 0.5 * quad(sq_f, 0, T, points=[0, T], full_output=1)[0]

    # QV term
    eps = 0.0000001
    alpha_const = alpha_diag(eps) / eps**(H-1/2)
    eps_int = (1/(2*H)) * alpha_const**2 * eps**(2*H) / (np.sqrt(kappa) * sinh(tau(0)) * cosh(tau(0)))
    qv_f = lambda s: alpha_diag(s) ** 2 / (np.sqrt(kappa) * sinh(tau(s)) * cosh(tau(s)))
    qv_term = 0.5 * ( eps_int + quad(qv_f, eps, T)[0] )

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
        print(res)
        pd.DataFrame(res).to_pickle("benchmarks/bench_{}.pkl".format(int(sys.argv[1])))
