# Simulations: MLE for Gaussian processes with an RBF kernel

Code for the simulation study (Section 4 and Appendix D) of

> Ameer Qaqish and Didong Li. *Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel.* 2026.

The paper shows that, under fixed-domain asymptotics, the maximum likelihood estimators of the spatial variance $\sigma^2$, lengthscale $\ell$ and nugget variance $\tau^2$ of a Gaussian process with the RBF kernel $\sigma^2\exp\{-\lVert x-x'\rVert^2/(2\ell^2)\}$ converge at the minimax-optimal rates $b_n^{-p/2}$, $b_n^{-(p+2)/2}$ and $n^{-1/2}$, where $b_n=\log n/\log\log n$, and are jointly asymptotically normal. The simulations check this on $[0,1]^p$ for $p=1,2,3$, with $\theta_0=(\sigma_0^2,\ell_0,\tau_0^2)=(1,0.25,0.01)$, regular grids of $n\approx10^2$ to $10^6$ points, and 1000 replicates per $(p,n)$.

## Contents

| File | Contents |
|---|---|
| `rbf_mle_sim.py` | All computation: data generation, the global MLE, the exact Fisher information, the Monte Carlo driver, the figures and Table 1. |
| `rbf_mle_simulation.ipynb` | Checks the fast code against dense linear algebra, draws the figures, and computes every number quoted in Section 4 and Appendix D. |
| `results/sim_p{1,2,3}.npz` | The Monte Carlo estimates (9 sample sizes × 1000 replicates × 3 parameters) and Fisher-information curves behind the paper. |
| `figures/` | Figures 1–3 (`rbf_mle_log_p{1,2,3}.pdf`) and A1–A3 (`rbf_mle_p{1,2,3}.pdf`) of the paper, with PNG copies. |

## Setup

The results were computed with Python 3.13 and the package versions pinned in `requirements.txt`:

```
pip install -r requirements.txt
pip install notebook          # only needed for the notebook
```

## Reproducing the paper

Redraw all figures and print Table 1 from the saved results (about a minute):

```
python rbf_mle_sim.py --plot-only
```

Rerun the whole study, overwriting `results/` and `figures/`:

```
python rbf_mle_sim.py
```

This takes about 8–9 hours on a 4-core laptop, almost all of it for $p=1$ with $n\ge3\times10^5$ ($p=2$ and $p=3$ take about 15 and 20 minutes). Each finished sample size is saved in `results/checkpoints/`, so an interrupted run picks up where it stopped when started again. `--p` selects dimensions (for example `--p 2 3`) and `--jobs` sets the number of parallel workers.

A quick end-to-end check with five sample sizes per dimension and 200 replicates takes about 3 minutes and writes to `results_quick/` and `figures_quick/`:

```
python rbf_mle_sim.py --quick
```

The notebook loads `results/` by default and runs in about ten minutes.

## How the computation works

- **Likelihood at large $n$.** The feature expansion $\exp\{-(x-x')^2/(2\ell^2)\}=\sum_k\phi_k(x)\phi_k(x')$, with $\phi_k(x)=e^{-u^2/2}u^k/\sqrt{k!}$ and $u=(x-1/2)/\ell$, is the power series used in the proof. Truncated at $M\approx50$–$225$ terms it reproduces the kernel to about $10^{-14}$, so $R_n=\Phi\Phi^\top$ has rank $M\ll n$, and the likelihood, exact draws from the model and the Fisher information cost $O(nM^2)$ instead of $O(n^3)$. On the regular grids used for $p=2,3$ the kernel matrix is a Kronecker product of one-dimensional matrices, so only one-dimensional quantities are formed. The notebook checks all of this against dense $O(n^3)$ computations.
- **Global MLE.** For each fixed $\ell$, the likelihood is maximised over $(\sigma^2,\tau^2)$ with L-BFGS-B from several starting values. Because the profile likelihood in $\ell$ can have more than one local maximum, it is first evaluated at 25 log-spaced lengthscales in $[0.05,2]$; every local maximum on this grid is then refined by a bounded Brent search in $\log\ell$, and the best point is returned. The parameter space is $[10^{-2},10^2]\times[0.05,2]\times[10^{-4},1]$.
- **Random seeds.** Replicate $r$ at the $i$-th sample size of dimension $p$ is seeded with entry $(i,r)$ of `replicate_seeds(p, 9, 1000)`, which is derived from `numpy.random.SeedSequence((20261003, p))`. Results therefore do not depend on the number of workers, and any replicate can be regenerated on its own.

## Citation

```bibtex
@misc{qaqish2026rbf,
  title  = {Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an {RBF} Kernel},
  author = {Qaqish, Ameer and Li, Didong},
  year   = {2026}
}
```

## License

MIT; see `LICENSE`.
