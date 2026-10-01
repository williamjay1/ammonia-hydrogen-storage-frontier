"""Exact elimination of directional flow variables from the cavern LP.

For net production v = eta*e-h*q, a feasible schedule always admits injection
max(v,0) and withdrawal max(-v,0). Simultaneous flows cannot improve the stock
balance and only increase nonnegative compression demand. The site-power
constraint is represented by its two linear pieces. Capacity and service
objectives are unchanged. This is a numerical reformulation, not new physics.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog


def solve_compact(c, *, A_ub, b_ub, A_eq, b_eq, bounds, method, **kwargs):
    c = np.asarray(c, dtype=float)
    n, remainder = divmod(len(c) - 2, 7)
    if remainder or n < 1 or A_eq.shape[0] < 3 * n:
        raise ValueError("Unexpected seven-block cavern LP")
    if np.any(c[4*n:6*n] != 0):
        raise ValueError("Flow objectives are not supported by this reformulation")
    keep = np.r_[np.arange(4*n), np.arange(6*n, 7*n+2)]
    # For a single cyclic period, summing all NH3 stock balances already
    # enforces the annual service row exactly. Removing that redundant dense
    # row preserves the feasible set and avoids an unnecessary factorization
    # bottleneck. Chronological multi-period and zero-boundary rows are kept.
    cyclic_single = (A_eq.shape[0] == 3*n + 1
                     and float(A_eq[n, 4*n-1]) == -1.0)
    eq_rows = (np.arange(2*n) if cyclic_single else
               np.r_[np.arange(2*n), np.arange(3*n, A_eq.shape[0])])
    # Infer the exact hydrogen coefficients from the original feed balance.
    eta = float(A_eq[2*n, 0])
    h = -float(A_eq[2*n, n])
    if eta <= 0 or h <= 0:
        raise ValueError("Unexpected hydrogen-feed coefficients")
    net = sparse.coo_matrix((np.r_[np.full(n, eta), np.full(n, -h)],
                            (np.r_[np.arange(n), np.arange(n)],
                             np.r_[np.arange(n), np.arange(n, 2*n)])),
                           shape=(n, len(keep))).tocsr()
    compact_eq = A_eq[eq_rows][:, keep].tolil()
    compact_eq[:n] = compact_eq[:n].tocsr() - net
    compact_eq = compact_eq.tocsr()
    compact_ub = A_ub[:, keep].tocsr()
    charge_block = A_ub[:, 4*n:5*n].tocsr()
    discharge_block = A_ub[:, 5*n:6*n].tocsr()
    if discharge_block.nnz:
        raise ValueError("Unexpected withdrawal coefficient in inequality matrix")
    power_rows = np.unique(charge_block.nonzero()[0])
    # Keep e+aux*q-g <= R and append its positive-net-injection piece.
    extra_a, extra_b = [], []
    if len(power_rows):
        if np.any(charge_block.data < 0):
            raise ValueError("Compression coefficients must be nonnegative")
        extra_a.append(compact_ub[power_rows] + charge_block[power_rows] @ net)
        extra_b.append(np.asarray(b_ub)[power_rows])
    for start, sign in ((4*n, 1.0), (5*n, -1.0)):
        limits = np.asarray([np.inf if bounds[start+t][1] is None
                             else bounds[start+t][1] for t in range(n)])
        if any(bounds[start+t][0] not in (0, 0.0, None) for t in range(n)):
            raise ValueError("Positive minimum flow is not supported")
        finite = np.flatnonzero(np.isfinite(limits))
        if len(finite):
            extra_a.append(sign * net[finite])
            extra_b.append(limits[finite])
    if extra_a:
        compact_ub = sparse.vstack([compact_ub, *extra_a], format="csr")
        compact_b = np.concatenate([np.asarray(b_ub), *extra_b])
    else:
        compact_b = np.asarray(b_ub)
    options = {"threads": 1, **kwargs.pop("options", {})}
    # SciPy forwards the native HiGHS threads option but warns it is not one
    # of its own documented options. Record the option in the run manifest.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
                                message="Unrecognized options detected.*")
        result = linprog(c[keep], A_ub=compact_ub, b_ub=compact_b,
                         A_eq=compact_eq, b_eq=np.asarray(b_eq)[eq_rows],
                         bounds=[bounds[int(j)] for j in keep], method=method,
                         options=options, **kwargs)
    if result.success:
        full = np.zeros_like(c)
        full[keep] = result.x
        production = eta * full[:n] - h * full[n:2*n]
        full[4*n:5*n] = np.maximum(production, 0)
        full[5*n:6*n] = np.maximum(-production, 0)
        result.x = full
    return result
