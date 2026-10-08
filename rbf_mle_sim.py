"""
Simulation study for fixed-domain maximum likelihood estimation of an RBF
Gaussian process with a nugget,

    Y_i = F(x_i) + eps_i,   F ~ GP(0, s * exp(-|x - x'|^2 / (2 l^2))),
    eps_i ~ N(0, t),        theta = (s, l, t) = (sigma^2, ell, tau^2),

on the unit cube [0, 1]^p, p in {1, 2, 3}, with regular-grid designs.
Companion code for A. Qaqish and D. Li, "Sharp Asymptotic Theory of Maximum
Likelihood Estimation for Gaussian Processes with an RBF Kernel".

Theory being checked (main theorem):
    s_hat - s0 = O_P(b_n^{-p/2}),
    l_hat - l0 = O_P(b_n^{-(p+2)/2}),
    sqrt(n) (t_hat - t0) => N(0, 2 t0^2),           b_n = log n / log log n,
together with MSE_j ~ [I_n(theta0)^{-1}]_jj and I_n^{1/2}(theta_hat - theta0)
=> N(0, I_3).

Computation
-----------
The Taylor (power-series) features

    phi_k(x) = exp(-u^2/2) u^k / sqrt(k!),   u = (x - 1/2) / l,

satisfy exp(-(x-x')^2/(2l^2)) = sum_k phi_k(x) phi_k(x'), and truncating the
sum at M = M(l) terms reproduces the kernel to ~1e-14 and its l-derivative
to ~1e-12 (max norm, l >= 0.05). This is the same power series that drives
the proof. Hence R_n(l) = Phi Phi^T with Phi of size n x M, and the Gaussian
log-likelihood, data generation and Fisher information all cost O(n M^2)
instead of O(n^3). For p = 2, 3 on a G^p grid the kernel factorises,
R_n = R_G (x) ... (x) R_G, so everything reduces to 1-D quantities on G
points. Eigenvalues of R_n below RTOL * max are treated as zero. This is
negligible near the maximum of the likelihood, but at extreme parameter
values (large s and l together with a tiny t) the computed log-likelihood
can be inaccurate.

The MLE maximises the likelihood over the box BOUNDS. Because the profile
likelihood in l can have several local maxima, it is evaluated on a grid of
l, every local maximum on the grid is refined by a Brent search, and the
inner (s, t) problem is solved from several starts (see fit_mle, _profile).

Usage
-----
    python rbf_mle_sim.py              # full study, p = 1, 2, 3 (results/, figures/)
    python rbf_mle_sim.py --quick      # small, fast version (results_quick/, figures_quick/)
    python rbf_mle_sim.py --plot-only  # redraw figures and Table 1 from saved results
"""

import argparse
import os
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy import stats

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

THETA0 = dict(s=1.0, l=0.25, t=0.01)            # true (sigma^2, ell, tau^2)
BOUNDS = dict(s=(1e-2, 1e2), l=(0.05, 2.0), t=(1e-4, 1.0))   # compact Theta

# sample sizes (p = 2, 3 use G^p grids, n = G^p)
N_LIST = {1: [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000],
          2: [g * g for g in (10, 18, 32, 56, 100, 178, 316, 562, 1000)],
          3: [g ** 3 for g in (5, 7, 10, 14, 22, 32, 46, 68, 100)]}
N_REP = 1000
SEED = 20261003

# sample sizes for the (cheap, simulation-free) Fisher-information curves
N_FISHER = {1: np.unique(np.round(np.logspace(2, 8, 25))).astype(int),
            2: np.unique(np.round(np.logspace(1, 6, 21)).astype(int)) ** 2,
            3: np.unique(np.round(np.logspace(np.log10(5), 4, 19)).astype(int)) ** 3}

QUICK = dict(N_LIST={1: [100, 300, 1000, 3000, 10000],
                     2: [g * g for g in (10, 18, 32, 56, 100)],
                     3: [g ** 3 for g in (5, 7, 10, 14, 22)]},
             N_REP=200,
             N_FISHER={1: np.unique(np.round(np.logspace(2, 6, 13))).astype(int),
                       2: np.unique(np.round(np.logspace(1, 4, 13)).astype(int)) ** 2,
                       3: np.unique(np.round(np.logspace(np.log10(5), 3, 9)).astype(int)) ** 3})

RTOL = 1e-12     # eigenvalues of R below RTOL * max are treated as zero
CHUNK = 4096     # rows per block (small blocks stay in cache)


def b_n(n):
    n = np.asarray(n, dtype=float)
    return np.log(n) / np.log(np.log(n))


# --------------------------------------------------------------------------
# Exact low-rank representation of the RBF kernel matrix on a 1-D grid
# --------------------------------------------------------------------------

def side(n, p):
    """Grid points per axis for n = G^p."""
    return n if p == 1 else int(round(n ** (1.0 / p)))


def grid(n):
    """Regular (midpoint) grid of n points in [0, 1]."""
    return (np.arange(n) + 0.5) / n


def n_features(l):
    """Number of Taylor features giving machine-precision kernel accuracy on
    [0, 1] (the series is a Poisson(u) sum with u <= (0.5/l)^2)."""
    u = (0.5 / l) ** 2
    return int(np.ceil(u + 10.0 * np.sqrt(u) + 25))


def _powers(u, K):
    """u^k / sqrt(k!) for k < K, by a running product (no exp/log)."""
    r = np.empty((u.size, K))
    r[:, 0] = 1.0
    if K > 1:
        r[:, 1:] = u[:, None] / np.sqrt(np.arange(1, K))
        np.cumprod(r, axis=1, out=r)
    return r


def features(x, l, M, deriv=False):
    """phi_k(x) = exp(-u^2/2) u^k / sqrt(k!), u = (x - 1/2)/l, k < M;
    optionally also d phi_k / d l."""
    u = (x - 0.5) / l
    P = np.exp(-0.5 * u ** 2)[:, None] * _powers(u, M)
    if deriv:
        return P, P * (u[:, None] ** 2 - np.arange(M)) / l
    return P


def gram(n, l, Y=None, deriv=False):
    """Gram matrix W^T W of the features on grid(n) (W = Phi, or [Phi, dPhi]
    if deriv), and optionally Phi^T Y, accumulated in blocks of rows."""
    M = n_features(l)
    G = np.zeros((2 * M, 2 * M) if deriv else (M, M))
    B = None if Y is None else np.zeros((M,) + Y.shape[1:])
    for a in range(0, n, CHUNK):
        xa = (np.arange(a, min(a + CHUNK, n)) + 0.5) / n    # = grid(n)[a:a + CHUNK]
        if deriv:
            W = np.hstack(features(xa, l, M, deriv=True))
        else:
            W = features(xa, l, M)
        G += W.T @ W
        if Y is not None:
            B += W[:, :M].T @ Y[a:a + CHUNK]
    return G, B, M


def gram_hankel(n, l, Y=None):
    """Same as gram(n, l, Y) without derivatives, in O(nM) instead of O(nM^2).

    phi_j phi_k = exp(-u^2) u^(j+k) / sqrt(j! k!), so Phi^T Phi is a scaled
    Hankel matrix: [Phi^T Phi]_jk = h_(j+k) sqrt(binom(j+k, j)) with the
    normalised moments h_m = sum_i exp(-u_i^2) u_i^m / sqrt(m!). The midpoint
    grid is symmetric about 1/2 (u_i = -u_(n-1-i)), so odd moments vanish and
    each sum runs over half of the points; Phi^T Y uses the even/odd parts
    of Y. The summands are exactly those of the direct product, regrouped."""
    M = n_features(l)
    half = n // 2
    x = grid(n)[:half]
    he = np.zeros(M)                                  # h_(2q), q < M
    B = None if Y is None else np.zeros((M,) + Y.shape[1:])
    if Y is not None:
        Ys, Yd = Y[:half] + Y[::-1][:half], Y[:half] - Y[::-1][:half]
        ev = np.arange(M) % 2 == 0
    q = np.arange(1, M)
    for a in range(0, half, CHUNK):
        u = (x[a:a + CHUNK] - 0.5) / l
        e = np.exp(-0.5 * u ** 2)
        V = np.empty((u.size, M))                     # u^(2q) / sqrt((2q)!)
        V[:, 0] = 1.0
        V[:, 1:] = (u * u)[:, None] / np.sqrt(2 * q * (2 * q - 1))
        np.cumprod(V, axis=1, out=V)
        he += 2.0 * ((e * e) @ V)
        if Y is not None:
            P = e[:, None] * _powers(u, M)
            B[ev] += P[:, ev].T @ Ys[a:a + CHUNK]
            B[~ev] += P[:, ~ev].T @ Yd[a:a + CHUNK]
    if n % 2 == 1:                                    # centre point, u = 0
        he[0] += 1.0
        if Y is not None:
            B[0] += Y[half]
    h = np.zeros(2 * M - 1)
    h[::2] = he
    j = np.arange(M)
    jk = j[:, None] + j[None, :]
    G = h[jk] * np.exp(0.5 * (gammaln(jk + 1) - gammaln(j + 1)[:, None]
                              - gammaln(j + 1)[None, :]))
    return G, B, M


def spectrum(n, l, Y=None, rtol=RTOL):
    """Nonzero eigenvalues lam of R_n(l) = Phi Phi^T and projections U^T Y
    (U = Phi V lam^{-1/2}); if Y is None, return U itself."""
    G, B, M = gram_hankel(n, l, Y)
    w, V = np.linalg.eigh(G)
    keep = w > rtol * w.max()
    w, V = w[keep], V[:, keep]
    if Y is None:
        return w, (features(grid(n), l, M) @ V) / np.sqrt(w)
    Z = V.T @ B
    return w, Z / (np.sqrt(w) if Z.ndim == 1 else np.sqrt(w)[:, None])


# --------------------------------------------------------------------------
# Data generation (exact draws from N(0, s R_n(l) + t I))
# --------------------------------------------------------------------------

def simulate(p, n, theta, rng):
    """One data set on the regular grid: a length-n vector (p = 1) or a
    G^p array (p = 2, 3; n = G^p). Since R = Phi Phi^T (Kronecker powers for
    p > 1), F = sqrt(s) Phi xi with xi ~ N(0, I) has exactly covariance s R."""
    s, l, t = theta["s"], theta["l"], theta["t"]
    M = n_features(l)
    if p == 1:
        F = np.sqrt(s) * (features(grid(n), l, M) @ rng.standard_normal(M))
        return F + np.sqrt(t) * rng.standard_normal(n)
    G = side(n, p)
    P = features(grid(G), l, M)
    if p == 2:
        F = np.sqrt(s) * (P @ rng.standard_normal((M, M)) @ P.T)
    else:
        F = np.sqrt(s) * np.einsum("ia,jb,kc,abc->ijk", P, P, P,
                                   rng.standard_normal((M, M, M)), optimize=True)
    return F + np.sqrt(t) * rng.standard_normal((G,) * p)


# --------------------------------------------------------------------------
# Exact maximum likelihood
# --------------------------------------------------------------------------

def _suffstats(p, Y, l):
    """For lengthscale l: nonzero eigenvalues of R_n(l) and the squared
    projections of the data onto the corresponding eigenvectors."""
    if p == 1:
        lam, z = spectrum(Y.size, l, Y)
        return lam, z ** 2
    if p == 2:
        lam1, Z1 = spectrum(Y.shape[0], l, Y)       # Z1 = U^T Y   (r x G)
        _, Z = spectrum(Y.shape[0], l, Z1.T)        # Z  = U^T Y U (r x r)
        return np.outer(lam1, lam1).ravel(), Z.ravel() ** 2
    lam1, U = spectrum(Y.shape[0], l)               # p = 3: U is G x r
    Z = np.einsum("ia,jb,kc,ijk->abc", U, U, U, Y, optimize=True)
    L = lam1[:, None, None] * lam1[None, :, None] * lam1[None, None, :]
    return L.ravel(), Z.ravel() ** 2


def _nll_st(logs, logt, lam, z2, n, E):
    """Negative log-likelihood (up to constants) and gradient in
    (log s, log t) for fixed l; C = s R + t I has eigenvalues s*lam + t on
    span(U) and t on its orthogonal complement."""
    s, t = np.exp(logs), np.exp(logt)
    d = s * lam + t
    r = lam.size
    res = E - z2.sum()
    f = 0.5 * (np.log(d).sum() + (n - r) * np.log(t) + (z2 / d).sum() + res / t)
    gs = 0.5 * s * ((lam / d).sum() - (z2 * lam / d ** 2).sum())
    gt = 0.5 * t * ((1 / d).sum() + (n - r) / t - (z2 / d ** 2).sum() - res / t ** 2)
    return f, np.array([gs, gt])


def _profile(l, p, Y, E, n, start=None):
    """Profile negative log-likelihood over (s, t) in the box, for fixed l.
    The (s, t) problem can have two local minima (the field explaining a
    small or a large share of the variance), so L-BFGS-B is run from s at
    its lower bound, from a moment estimate of s on the leading
    eigen-directions (z_k^2 ~ s lam_k there), and from `start` if given;
    the best solution is kept."""
    lam, z2 = _suffstats(p, Y, l)
    lb = np.log([BOUNDS["s"][0], BOUNDS["t"][0]])
    ub = np.log([BOUNDS["s"][1], BOUNDS["t"][1]])
    top = np.argsort(lam)[-5:]                     # leading eigen-directions
    t_init = max((E - z2.sum()) / max(n - lam.size, 1), 2e-4)
    s_init = max(np.mean(z2[top] / lam[top]), BOUNDS["s"][0])
    starts = [np.log([BOUNDS["s"][0], t_init]), np.log([s_init, t_init])]
    if start is not None:
        starts.append(np.asarray(start, dtype=float))
    best = None
    for st in starts:
        res = minimize(lambda v: _nll_st(v[0], v[1], lam, z2, n, E),
                       np.clip(st, lb + 1e-9, ub - 1e-9), jac=True,
                       method="L-BFGS-B", bounds=list(zip(lb, ub)),
                       options=dict(ftol=1e-15, gtol=1e-10, maxiter=500))
        if best is None or res.fun < best.fun:
            best = res
    return best.fun, best.x


def fit_mle(p, Y, n_grid=25):
    """MLE of (s, l, t) over the box BOUNDS. The profile likelihood in l
    can have more than one local maximum, so it is first evaluated on a
    log-spaced grid of l; every local minimum of the grid profile (negative
    log-likelihood) is then refined by a bounded Brent search in log l
    between its grid neighbours, and the best point found is returned."""
    n = Y.size
    E = float((Y ** 2).sum())
    lg = np.exp(np.linspace(*np.log(BOUNDS["l"]), n_grid))
    prof = [_profile(l, p, Y, E, n) for l in lg]
    vals = np.array([v for v, _ in prof])
    pad = np.r_[np.inf, vals, np.inf]
    basins = np.flatnonzero((vals <= pad[:-2]) & (vals <= pad[2:]))
    i = int(np.argmin(vals))
    f_best, l_hat, (ls, lt) = vals[i], lg[i], prof[i][1]
    for i in basins:
        lo, hi = np.log(lg[max(i - 1, 0)]), np.log(lg[min(i + 1, n_grid - 1)])
        st0 = prof[i][1]
        res = minimize_scalar(lambda v: _profile(np.exp(v), p, Y, E, n, st0)[0],
                              bounds=(lo, hi), method="bounded",
                              options=dict(xatol=1e-7))
        l_i = float(np.exp(res.x))
        f, x = _profile(l_i, p, Y, E, n, st0)
        if f < f_best:                           # else the grid point stays
            f_best, l_hat, (ls, lt) = f, l_i, x
    return np.array([np.exp(ls), l_hat, np.exp(lt)])


# --------------------------------------------------------------------------
# Exact Fisher information
# --------------------------------------------------------------------------

def _basis(G, l):
    """Eigenvalues lam of R_G(l) and the matrix B = U^T (dR/dl) U, both in
    an orthonormal basis U of span(Phi, dPhi/dl); R and dR/dl vanish on its
    orthogonal complement."""
    W, _, M = gram(G, l, deriv=True)
    w, V = np.linalg.eigh(W)
    keep = w > 1e-15 * w.max()
    w, V = w[keep], V[:, keep]
    T = V / np.sqrt(w)
    QP, QdP = T.T @ W[:, :M], T.T @ W[:, M:]
    A = QP @ QP.T
    D = QdP @ QP.T
    lam, U = np.linalg.eigh(A)
    return lam, U.T @ (D + D.T) @ U


def fisher_information(p, n, theta):
    """I_n(theta)_{ab} = tr(C^{-1} C_a C^{-1} C_b) / 2 for (s, l, t) on the
    regular grid with n points (p = 2, 3: n = G^p)."""
    s, l, t = theta["s"], theta["l"], theta["t"]
    G = side(n, p)
    lam, B = _basis(G, l)
    dB = np.diag(B)
    if p == 1:
        L, Rd = lam, dB                                   # eig(R), diag(U^T dR U)
        g = 1 / (s * L + t)
        Ill = 0.5 * s * s * np.einsum("i,k,ik->", g, g, B ** 2)
    elif p == 2:
        L = np.outer(lam, lam)
        Rd = np.outer(dB, lam) + np.outer(lam, dB)
        g = 1 / (s * L + t)
        B2 = B ** 2
        Ill = 0.5 * s * s * (np.einsum("j,ij,kj,ik->", lam ** 2, g, g, B2)
                             + np.einsum("i,ij,il,jl->", lam ** 2, g, g, B2)
                             + 2 * np.sum(g ** 2 * np.outer(lam * dB, lam * dB)))
    else:
        # p = 3: dR = B(x)L(x)L + L(x)B(x)L + L(x)L(x)B in the eigenbasis; the
        # three terms overlap only on the diagonal.
        o = lambda a, b, c: a[:, None, None] * b[None, :, None] * c[None, None, :]
        L = o(lam, lam, lam)
        a1, a2, a3 = o(dB, lam, lam), o(lam, dB, lam), o(lam, lam, dB)
        Rd = a1 + a2 + a3
        g = 1 / (s * L + t)
        B2, l2 = B ** 2, lam ** 2
        sq = (np.einsum("ijk,ajk,ia,j,k->", g, g, B2, l2, l2, optimize=True)
              + np.einsum("ijk,iak,ja,i,k->", g, g, B2, l2, l2, optimize=True)
              + np.einsum("ijk,ija,ka,i,j->", g, g, B2, l2, l2, optimize=True))
        cross = 2 * np.sum(g ** 2 * (a1 * a2 + a1 * a3 + a2 * a3))
        Ill = 0.5 * s * s * (sq + cross)
    Iss = 0.5 * np.sum(g ** 2 * L ** 2)
    Ist = 0.5 * np.sum(g ** 2 * L)
    Itt = 0.5 * (np.sum(g ** 2) + (n - L.size) / t ** 2)
    Isl = 0.5 * s * np.sum(g ** 2 * L * Rd)
    Ilt = 0.5 * s * np.sum(g ** 2 * Rd)
    return np.array([[Iss, Isl, Ist], [Isl, Ill, Ilt], [Ist, Ilt, Itt]])


def fisher_dense(X, theta):
    """O(n^3) reference implementation (used only for testing)."""
    s, l, t = theta["s"], theta["l"], theta["t"]
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    R = np.exp(-D2 / (2 * l * l))
    Ci = np.linalg.inv(s * R + t * np.eye(len(X)))
    A = [Ci @ R, Ci @ (s * R * D2 / l ** 3), Ci]       # C^{-1} dC/d(s, l, t)
    return np.array([[0.5 * np.sum(a * b.T) for b in A] for a in A])


def nll_dense(X, Y, theta):
    """O(n^3) reference negative log-likelihood (used only for testing)."""
    s, l, t = theta
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    C = s * np.exp(-D2 / (2 * l * l)) + t * np.eye(len(X))
    c = np.linalg.cholesky(C)
    a = np.linalg.solve(c, Y)
    return np.log(np.diag(c)).sum() + 0.5 * a @ a


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------

def replicate_seeds(p, n_sizes, n_rep, seed=SEED):
    """Seeds of the Monte Carlo replicates: entry [i, r] seeds replicate r
    at the i-th of n_sizes sample sizes."""
    root = np.random.SeedSequence([seed, p])
    return root.generate_state(n_sizes * n_rep).reshape(n_sizes, n_rep)


def _one_rep(p, n, seed):
    rng = np.random.default_rng(seed)
    Y = simulate(p, n, THETA0, rng)
    return fit_mle(p, Y)


def run_simulation(p, n_list, n_rep, seed=SEED, n_jobs=-1, verbose=True,
                   checkpoint=None):
    """Monte Carlo MLEs: array of shape (len(n_list), n_rep, 3). Replicate r
    at the i-th sample size always uses the same seed, so results do not
    depend on n_jobs. If checkpoint is a directory, each sample size is
    saved there when finished and reloaded (not recomputed) on a rerun."""
    from joblib import Parallel, delayed
    seeds = replicate_seeds(p, len(n_list), n_rep, seed)
    est = np.empty((len(n_list), n_rep, 3))
    for i, n in enumerate(n_list):
        ck = (None if checkpoint is None else
              os.path.join(checkpoint, f"p{p}_n{n}_r{n_rep}_s{seed}.npy"))
        t0 = time.time()
        if ck is not None and os.path.exists(ck):
            est[i] = np.load(ck)
        else:
            out = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(p, n, int(sd))
                                          for sd in seeds[i])
            est[i] = np.array(out)
            if ck is not None:
                np.save(ck, est[i])
        if verbose:
            e = est[i] - np.array([THETA0[k] for k in "slt"])
            print(f"  p={p} n={n:>8d}  rmse(s,l,t) = "
                  + " ".join(f"{v:.3e}" for v in np.sqrt((e ** 2).mean(0)))
                  + f"   [{time.time() - t0:.0f}s]", flush=True)
    return est


def fisher_curve(p, n_list):
    """Inverse Fisher information I_n(theta0)^{-1} (3 x 3) for each n; its
    diagonal gives the asymptotic variances."""
    Vs = np.array([np.linalg.inv(fisher_information(p, int(n), THETA0))
                   for n in n_list])
    return Vs


def run_all(p, cfg, outdir="results", n_jobs=-1):
    """Fisher curves and Monte Carlo for one dimension p. The results file is
    rewritten after every sample size, so figures can be drawn from the
    completed sizes while the larger ones are still running."""
    ckdir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    print(f"Fisher information curve, p={p} ...", flush=True)
    t0 = time.time()
    nF = np.asarray(cfg["N_FISHER"][p])
    VF = fisher_curve(p, nF)
    nS = np.asarray(cfg["N_LIST"][p])
    VS = fisher_curve(p, nS)
    print(f"  done [{time.time() - t0:.0f}s]", flush=True)
    path = os.path.join(outdir, f"sim_p{p}.npz")
    print(f"Monte Carlo, p={p}, {cfg['N_REP']} replicates per n ...", flush=True)
    for k in range(1, len(nS) + 1):
        t0 = time.time()
        est = run_simulation(p, nS[:k], cfg["N_REP"], n_jobs=n_jobs,
                             verbose=False, checkpoint=ckdir)
        e = est[-1] - np.array([THETA0[c] for c in "slt"])
        print(f"  p={p} n={nS[k-1]:>8d}  rmse(s,l,t) = "
              + " ".join(f"{v:.3e}" for v in np.sqrt((e ** 2).mean(0)))
              + f"   [{time.time() - t0:.0f}s]", flush=True)
        np.savez(path, p=p, n=nS[:k], est=est, V=VS[:k], n_fisher=nF,
                 V_fisher=VF, theta0=np.array([THETA0[c] for c in "slt"]),
                 bounds=np.array([BOUNDS[c] for c in "slt"]))
    print(f"saved {path}", flush=True)
    return path


# --------------------------------------------------------------------------
# Summaries and figures
# --------------------------------------------------------------------------

def _iqr_sd(e, axis):
    """Robust scale: interquartile range / 1.349 (= SD for a normal law)."""
    q1, q3 = np.percentile(e, [25, 75], axis=axis)
    return (q3 - q1) / 1.349


def summarize(res, n_boot=2000, seed=0):
    """RMSE and robust scale (IQR/1.349) with bootstrap 95% intervals, bias,
    SD and Fisher prediction."""
    th0 = res["theta0"]
    err = res["est"] - th0
    rmse = np.sqrt((err ** 2).mean(1))
    rng = np.random.default_rng(seed)
    R = err.shape[1]
    iqr = _iqr_sd(err, 1)
    boot = np.empty((n_boot,) + rmse.shape)
    boot_q = np.empty((n_boot,) + rmse.shape)
    for b in range(n_boot):
        idx = rng.integers(0, R, R)
        boot[b] = np.sqrt((err[:, idx] ** 2).mean(1))
        boot_q[b] = _iqr_sd(err[:, idx], 1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    qlo, qhi = np.percentile(boot_q, [2.5, 97.5], axis=0)
    sd_fisher = np.sqrt(np.diagonal(res["V"], axis1=1, axis2=2))
    lo_b, hi_b = res["bounds"][:, 0], res["bounds"][:, 1]
    at_bound = ((np.isclose(res["est"], lo_b, rtol=1e-3)
                 | np.isclose(res["est"], hi_b, rtol=1e-3)).mean(1))
    return dict(n=res["n"], rmse=rmse, lo=lo, hi=hi, iqr=iqr, qlo=qlo, qhi=qhi,
                bias=err.mean(1),
                sd=err.std(1, ddof=1), sd_fisher=sd_fisher, at_bound=at_bound)


def loglog_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def standardized(res, i):
    """Coordinatewise standardized errors (theta_hat_j - theta0_j) /
    sqrt([I_n^{-1}]_jj) and the joint statistic |I_n^{1/2}(theta_hat -
    theta0)|^2, which is chi^2_3 under the normal limit."""
    err = res["est"][i] - res["theta0"]
    V = res["V"][i]
    z = err / np.sqrt(np.diag(V))
    q = np.einsum("ri,ij,rj->r", err, np.linalg.inv(V), err)
    return z, q


PAR_LABELS = [r"$\sigma^2$", r"$\ell$", r"$\tau^2$"]
LOG_LABELS = [r"$\log\sigma^2$", r"$\log\ell$", r"$\log\tau^2$"]
COL = ["#1f5fa8", "#c2410c", "#15803d"]


def to_log(res):
    """Results for the log-parameters (log s, log l, log t). By the delta
    method the asymptotic covariance becomes D I_n^{-1} D, D = diag(1/theta0),
    so the rates are unchanged; e.g. sqrt(n)(log t_hat - log t0) => N(0, 2)."""
    D = 1.0 / res["theta0"]
    out = dict(res)
    out["est"] = np.log(res["est"])
    out["theta0"] = np.log(res["theta0"])
    out["bounds"] = np.log(res["bounds"])
    out["V"] = res["V"] * np.outer(D, D)
    out["V_fisher"] = res["V_fisher"] * np.outer(D, D)
    out["log"] = True
    return out


def _shade(color, f):
    """Lighten (f < 0, towards white) or darken (f > 0, towards black)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(color))
    return tuple(c + (1 - c) * (-f) if f < 0 else c * (1 - f))


def _decade_ticks(axis, nmax, p):
    """Top n-axis: a tick at every decade and no minor ticks (same in every
    figure); labels on every decade for p = 1, every other decade for p > 1."""
    from matplotlib.ticker import FixedLocator, NullLocator
    ks = list(range(2, int(np.log10(nmax)) + 1))
    axis.xaxis.set_major_locator(FixedLocator([10.0 ** k for k in ks]))
    axis.xaxis.set_minor_locator(NullLocator())
    step = 1 if p == 1 else 2
    axis.set_xticklabels([f"$10^{{{k}}}$" if (k - 2) % step == 0 else "" for k in ks])


def _num(x, fmt=".2f"):
    """Number for a legend, with a true minus sign in the text font."""
    return format(x, fmt).replace("-", "\u2212")


def make_figure(res, path=None, log=False):
    """Two-row figure for one dimension p: log-log rate plots (top) and
    normal Q-Q plots (bottom). With log=True, everything refers to
    (log sigma^2, log ell, log tau^2)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "mathtext.fontset": "cm"})
    p = int(res["p"])
    th_raw = res["theta0"]
    if log:
        res = to_log(res)
    labels = LOG_LABELS if log else PAR_LABELS
    summ = summarize(res)
    n, nF = summ["n"], res["n_fisher"]
    sdF = np.sqrt(np.diagonal(res["V_fisher"], axis1=1, axis2=2))
    th0 = res["theta0"]
    expo = [p / 2, (p + 2) / 2]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9.0),
                             gridspec_kw=dict(height_ratios=[1.15, 1],
                                              hspace=0.7))
    # ---------------- rates ----------------
    for j in range(3):
        ax = axes[0, j]
        if j < 2:
            x, xF = b_n(n), b_n(nF)
        else:
            x, xF = n.astype(float), nF.astype(float)
        yerr = np.vstack([summ["rmse"][:, j] - summ["lo"][:, j],
                          summ["hi"][:, j] - summ["rmse"][:, j]])
        # slope of the asymptotic SD over the simulated n only, so that it is
        # comparable with the slopes fitted to the Monte Carlo errors
        sa = loglog_slope(x, summ["sd_fisher"][:, j])
        ax.plot(xF, sdF[:, j], "-", color="0.55", lw=1.6,
                label=f"Asymptotic SD (fitted slope = {_num(sa)})")
        if j < 2:
            # theoretical rate, anchored at the largest Fisher-curve point
            ref = sdF[-1, j] * (xF / xF[-1]) ** (-expo[j])
            lab = (r"Theoretical rate $\propto b_n^{-%s}$" % ("p/2" if j == 0 else "(p+2)/2")
                   + f" (slope = {_num(-expo[j], 'g')})")
            ax.plot(xF, ref, "--", color="k", lw=1.1, label=lab)
        else:
            if log:
                ax.plot(xF, np.sqrt(2 / xF), "--", color="k", lw=1.1,
                        label=r"Theoretical rate $\sqrt{2}\, n^{-1/2}$" + f" (slope = {_num(-0.5, 'g')})")
            else:
                ax.plot(xF, np.sqrt(2) * th0[2] / np.sqrt(xF), "--", color="k",
                        lw=1.1, label=r"Theoretical rate $\sqrt{2}\,\tau_0^2\, n^{-1/2}$" + f" (slope = {_num(-0.5, 'g')})")
        sq = loglog_slope(x, summ["iqr"][:, j])
        ax.errorbar(x * 1.012, summ["iqr"][:, j],
                    yerr=np.vstack([summ["iqr"][:, j] - summ["qlo"][:, j],
                                    summ["qhi"][:, j] - summ["iqr"][:, j]]),
                    fmt="s", ms=4.5, mfc="white", color=COL[j], ecolor=COL[j],
                    capsize=2.5, lw=1.0,
                    label=f"IQR/1.349 (fitted slope = {_num(sq)})", zorder=3)
        sl = loglog_slope(x, summ["rmse"][:, j])
        ax.errorbar(x, summ["rmse"][:, j], yerr=yerr, fmt="o", ms=4.5,
                    color=COL[j], ecolor=COL[j], capsize=2.5, lw=1.2,
                    label=f"RMSE (fitted slope = {_num(sl)})", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        x0, x1 = x[-1] * 1.03, xF[-1] * 1.05
        ax.axvspan(x0, x1, color="0.93", lw=0, zorder=0)
        # shaded: beyond the largest simulated n (asymptotic SD and theory only)
        ax.set_xlim(None, xF[-1] * 1.05)
        ax.set_ylabel("RMSE of " + labels[j])
        if j < 2:
            ax.set_xlabel(r"$b_n=\log n/\log\log n$  (log scale)")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            top = ax.secondary_xaxis("top", functions=(_b_to_n, _n_to_b))
            top.set_xscale("log")
            _decade_ticks(top, nF[-1], p)
            top.set_xlabel("$n$", labelpad=4)
        else:
            ax.set_xlabel("$n$  (log scale)")
            _decade_ticks(ax, nF[-1], p)
            top = ax.secondary_xaxis("top")
            _decade_ticks(top, nF[-1], p)
            top.set_xlabel("$n$", labelpad=4)
        # same order as the slope table: RMSE, IQR/1.349, asymptotic SD, theory
        h, lb = ax.get_legend_handles_labels()
        order = [3, 2, 0, 1]
        ax.legend([h[i] for i in order], [lb[i] for i in order], fontsize=7.8,
                  frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        ax.set_title(f"({'abc'[j]}) " + labels[j], loc="left", fontsize=11)

    # ---------------- normality ----------------
    # One colour per sample size (shared by the three panels); the grey band
    # is the pointwise 95% range of the k-th order statistic of R draws from
    # an exact N(0, 1). Points outside the frame are drawn as triangles on
    # its top or bottom edge.
    k_show = [0, len(n) // 2, len(n) - 1]
    n_shade = [-0.6, 0.0, 0.55]                 # light -> dark with n
    R = res["est"].shape[1]
    i = np.arange(1, R + 1)
    qq = stats.norm.ppf((i - 0.5) / R)
    band = stats.norm.ppf(stats.beta.ppf([[0.025], [0.975]], i, R - i + 1))
    lim, ytop, ybot = 3.6, 8.0, -4.6
    for j in range(3):
        ax = axes[1, j]
        ax.fill_between(qq, band[0], band[1], color="0.85", lw=0,
                        label="95% band for exact $N(0,1)$")
        ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1)
        for k, f in zip(k_show, n_shade):
            c = _shade(COL[j], f)
            zs = np.sort(standardized(res, k)[0][:, j])
            inside = (zs <= ytop) & (zs >= ybot)
            ax.plot(qq[inside], zs[inside], "o", ms=2.2, mew=0, color=c,
                    alpha=0.85, label=f"$n={n[k]:,}$")
            for m, edge, mk in [(zs > ytop, ytop - 0.12, "^"),
                                (zs < ybot, ybot + 0.12, "v")]:
                if m.any():
                    ax.plot(qq[m], np.full(m.sum(), edge), mk, ms=4, mew=0,
                            color=c, clip_on=False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(ybot, ytop)
        ax.set_xlabel("standard normal quantiles")
        if log:
            ax.set_ylabel(r"$(\log\hat\theta_j-\log\theta_{0j})\,\theta_{0j}/[\mathcal{I}_n(\theta_0)^{-1}]_{jj}^{1/2}$")
        else:
            ax.set_ylabel(r"$(\hat\theta_j-\theta_{0j})/[\mathcal{I}_n(\theta_0)^{-1}]_{jj}^{1/2}$")
        ax.set_title(f"({'def'[j]}) normal Q–Q, " + labels[j], loc="left",
                     fontsize=11)
        ax.legend(fontsize=7.8, frameon=False, loc="upper left", markerscale=2.2)
    fig.suptitle(f"$p={p}$: fixed-domain MLE of the RBF kernel parameters"
                 + (" (log scale)" if log else "")
                 + f" ($\\sigma_0^2={th_raw[0]:g},\\ \\ell_0={th_raw[1]:g},\\ "
                 f"\\tau_0^2={th_raw[2]:g}$; {res['est'].shape[1]} replicates per $n$)",
                 fontsize=11.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07, wspace=0.32)
    if path:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200, bbox_inches="tight")
    return fig


def _n_to_b(n):
    return b_n(np.clip(n, 16.0, None))


def _b_to_n(b):
    """Inverse of b_n = log n / log log n (Newton on log n)."""
    b = np.clip(np.atleast_1d(np.asarray(b, dtype=float)), 2.0, 40.0)
    L = np.maximum(b * np.log(np.maximum(b, 1.5)), 1.1)
    for _ in range(60):
        L = np.maximum(L - (L / np.log(L) - b) / ((np.log(L) - 1) / np.log(L) ** 2), 1.0001)
    return np.exp(L)


def load(path):
    d = dict(np.load(path))
    d["p"] = int(d["p"])
    return d


def print_slope_table(results):
    """Table 1 of the paper: least-squares slopes of log(error) against
    log b_n (log sigma^2, log ell) or log n (log tau^2) over the simulated
    sample sizes, for the RMSE, IQR/1.349 and asymptotic SD of the
    log-parameters. results: dict p -> loaded results."""
    print("\nTable 1 (log scale): fitted slopes over the simulated n")
    print(" " * 16 + "".join(f"{h:>10s}" for h in ["RMSE", "IQR/1.349", "asy.SD", "theory"]))
    for j, name in enumerate(["log sigma^2", "log ell", "log tau^2"]):
        for p in sorted(results):
            r = to_log(results[p])
            err = r["est"] - r["theta0"]
            n = r["n"]
            x = b_n(n) if j < 2 else n.astype(float)
            ys = [np.sqrt((err[:, :, j] ** 2).mean(1)), _iqr_sd(err[:, :, j], 1),
                  np.sqrt(r["V"][:, j, j])]
            theory = [-p / 2, -(p + 2) / 2, -0.5][j]
            print(f"{name:>11s} p={p} " + "".join(f"{loglog_slope(x, y):10.2f}" for y in ys)
                  + f"{theory:10.2g}")


def print_table(res):
    s = summarize(res)
    p = res["p"]
    print(f"\np = {p}")
    print("        n    b_n |  RMSE(s)  asy.SD | RMSE(l)  asy.SD | "
          "sqrt(n)RMSE(t)  sqrt2 t0 | at-bound(s,l,t)")
    for i, n in enumerate(s["n"]):
        print(f"{n:9d} {b_n(n):6.3f} | {s['rmse'][i,0]:.4f}  {s['sd_fisher'][i,0]:.4f} |"
              f" {s['rmse'][i,1]:.5f} {s['sd_fisher'][i,1]:.5f} |"
              f"   {np.sqrt(n)*s['rmse'][i,2]:.5f}     {np.sqrt(2)*res['theta0'][2]:.5f} |"
              f" {s['at_bound'][i,0]:.3f} {s['at_bound'][i,1]:.3f} {s['at_bound'][i,2]:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small, fast run")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw figures and tables from saved results")
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--outdir", default=None,
                    help="default: results (results_quick with --quick)")
    ap.add_argument("--figdir", default=None,
                    help="default: figures (figures_quick with --quick)")
    args = ap.parse_args()
    # figures are only saved to files, never shown
    import matplotlib
    matplotlib.use("Agg")
    cfg = QUICK if args.quick else dict(N_LIST=N_LIST, N_REP=N_REP, N_FISHER=N_FISHER)
    suffix = "_quick" if args.quick else ""
    outdir = args.outdir or "results" + suffix
    figdir = args.figdir or "figures" + suffix
    os.makedirs(figdir, exist_ok=True)
    results = {}
    for p in args.p:
        path = os.path.join(outdir, f"sim_p{p}.npz")
        if not args.plot_only:
            run_all(p, cfg, outdir, args.jobs)
        res = results[p] = load(path)
        print_table(res)
        make_figure(res, os.path.join(figdir, f"rbf_mle_p{p}.pdf"))
        make_figure(res, os.path.join(figdir, f"rbf_mle_log_p{p}.pdf"), log=True)
    print_slope_table(results)


if __name__ == "__main__":
    main()
