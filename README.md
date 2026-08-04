# phase_iso

# Graphical Models for Multivariate Phase Relationships

Python implementations of the interaction-screening method introduced in:

> Andrew S. Perley and Todd P. Coleman
> **Graphical Models and Efficient Inference Methods for Multivariate Phase Probability Distributions**
> [arXiv:2504.00459](https://arxiv.org/abs/2504.00459)

This repository provides:

* a reference CVXPY implementation;
* a faster JAX implementation with a compatible API;
* regularized graph-structure estimation;
* two-stage support selection and unregularized refitting;
* Gibbs sampling;
* likelihood utilities;
* synthetic benchmarks comparing CVXPY and JAX.

## Model

Let

$$
Y = (Y_1,\ldots,Y_p)
$$

be a vector of phases, where each $Y_i$ is represented in radians.

The pairwise circular graphical model has density

$$
p(y) =
\frac{1}{Z(\theta)}
\exp\left[
\sum_{(i,j)\in E}
\kappa_{ij}
\cos\left(y_j-y_i-\mu_{ij}\right)
\right],
$$

where:

* $\kappa_{ij} \geq 0$ is the coupling magnitude;
* $\mu_{ij}$ is the preferred phase difference;
* $E$ is the graph edge set;
* $Z(\theta)$ is the partition function.

The implementation uses the natural parameters

$$
\theta_{c,ij} =
\kappa_{ij}\cos(\mu_{ij})
$$

and

$$
\theta_{s,ij} =
\kappa_{ij}\sin(\mu_{ij}).
$$

The coupling magnitude and preferred phase difference can be recovered using

$$
\kappa_{ij} =
\sqrt{
\theta_{c,ij}^2
+
\theta_{s,ij}^2
}
$$

and

$$
\mu_{ij} =
\operatorname{atan2}
\left(
\theta_{s,ij},
\theta_{c,ij}
\right).
$$

The parameter matrices satisfy

$$
\theta_c = \theta_c^\mathsf{T}
$$

and

$$
\theta_s = -\theta_s^\mathsf{T}.
$$

Thus, `theta_c` is symmetric and `theta_s` is skew-symmetric.

## Interaction Screening

For each node $u$, the interaction-screening estimator minimizes a nodewise objective.

The regularized objective has the form

$$
\widehat{\theta}_u =
\operatorname*{arg,min}*{\theta_u}
\left[
\frac{1}{n}
\sum*{k=1}^{n}
\exp\left(-z_{k,u}^{\mathsf{T}}\theta_u\right)
+
\lambda
\sum_{j\neq u}
\sqrt{
\theta_{c,uj}^2
+
\theta_{s,uj}^2
}
\right].
$$

The penalty is a group-lasso penalty. The cosine and sine coefficients associated with one edge are treated as a single group.

The default regularization strength is

$$
\lambda =
4
\sqrt{
\frac{
\log\left(8p^2/\epsilon\right)
}{
n
}
},
$$

with $\epsilon=0.1$ in the current implementation.

## Repository Layout

```text
.
├── iso.py
├── iso_jax.py
├── benchmark_iso.py
├── plot_iso_comparisons.py
├── README.md
└── LICENSE
```

### `iso.py`

Reference implementation using:

* NumPy;
* SciPy;
* CVXPY;
* MOSEK when available;
* SCS as a fallback;
* joblib for nodewise parallelization.

### `iso_jax.py`

JAX implementation with the same primary function names and argument order as `iso.py`.

To switch implementations, change the import:

```python
# Reference CVXPY implementation
import iso
```

```python
# JAX implementation
import iso_jax as iso
```

Existing calls can remain unchanged:

```python
theta_c, theta_s = iso.fit(Y)
```

```python
theta_c, theta_s = iso.fit_sparse(Y)
```

## Installation

Python 3.10 or newer is recommended.

### Install with `uv`

```bash
uv venv
source .venv/bin/activate
```

```bash
uv pip install \
    numpy \
    scipy \
    cvxpy \
    joblib \
    jax \
    matplotlib
```

MOSEK is optional:

```bash
uv pip install mosek
```

Without MOSEK, the CVXPY implementation falls back to SCS.

### Run without creating a persistent environment

```bash
uv run \
    --with numpy \
    --with scipy \
    --with cvxpy \
    --with joblib \
    --with jax \
    --with matplotlib \
    python benchmark_iso.py
```

Add MOSEK when available:

```bash
uv run \
    --with numpy \
    --with scipy \
    --with cvxpy \
    --with mosek \
    --with joblib \
    --with jax \
    --with matplotlib \
    python benchmark_iso.py
```

## Input Data

The estimators expect a NumPy array with shape

```text
(n_samples, n_nodes)
```

Each entry should contain a phase in radians, usually wrapped onto $[-\pi,\pi)$.

```python
import numpy as np

Y = np.load("phases.npy")

if Y.ndim != 2:
    raise ValueError("Y must have shape (n_samples, n_nodes).")

Y = np.angle(np.exp(1j * Y))

print(Y.shape)
```

Rows are treated as independent observations.

## Regularized Estimation

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=True,
)

kappa = np.hypot(theta_c, theta_s)
mu = np.arctan2(theta_s, theta_c)

np.savez(
    "regularized_model.npz",
    theta_c=theta_c,
    theta_s=theta_s,
    kappa=kappa,
    mu=mu,
)
```

The output should satisfy

```python
assert np.allclose(theta_c, theta_c.T)
assert np.allclose(theta_s, -theta_s.T)
```

## Two-Stage Sparse Estimation

The two-stage estimator performs:

1. group-lasso regularized structure selection;
2. unregularized refitting restricted to the selected edge set.

Use `fit_sparse` to run both stages:

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    beta=None,
    alpha=2e-6,
    lam=None,
    parallel=False,
    verbose=True,
)

kappa = np.hypot(theta_c, theta_s)
mu = np.arctan2(theta_s, theta_c)

np.savez(
    "refit_model.npz",
    theta_c=theta_c,
    theta_s=theta_s,
    kappa=kappa,
    mu=mu,
)
```

The first-stage coefficients are shrunk by the group-lasso penalty.

The second-stage coefficients are estimated without the group-lasso penalty, but only for edges selected during the first stage. Therefore, the regularized and refitted coefficient values are generally different.

## Hard Thresholding with `alpha`

The optional `alpha` argument applies a hard threshold after the regularized fit.

For each edge, the estimated magnitude is

$$
\widehat{\kappa}_{ij} =
\sqrt{
\widehat{\theta}*{c,ij}^2
+
\widehat{\theta}*{s,ij}^2
}.
$$

The implementation sets the edge to zero when

$$
\widehat{\kappa}_{ij}
<
\frac{\alpha}{2}.
$$

For example:

```python
theta_c, theta_s = iso.fit_sparse(
    Y,
    alpha=2e-6,
)
```

This corresponds to an effective magnitude threshold of $10^{-6}$.

Disable explicit hard thresholding with:

```python
theta_c, theta_s = iso.fit_sparse(
    Y,
    alpha=None,
)
```

A small positive `alpha` may be useful with CVXPY because conic solvers can return small numerical residuals rather than exact zeros.

## Choosing `lambda`

Use the default theoretically motivated value with:

```python
theta_c, theta_s = iso.fit(
    Y,
    lam=None,
)
```

Supply a custom value with:

```python
theta_c, theta_s = iso.fit(
    Y,
    lam=0.1,
)
```

In general:

```text
larger lambda  -> fewer selected edges
smaller lambda -> more selected edges
```

## CVXPY Implementation

```python
import numpy as np
import iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    beta=None,
    alpha=2e-6,
    lam=None,
    parallel=True,
    verbose=False,
)
```

When `parallel=True`, the nodewise problems are distributed using joblib.

The implementation attempts to use MOSEK first and falls back to SCS if MOSEK fails.

## JAX Implementation

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    beta=None,
    alpha=2e-6,
    lam=None,
    parallel=False,
    verbose=True,
)
```

The regularized JAX estimator uses proximal-gradient optimization.

For an edge group $v_j$, the group-lasso proximal update is

$$
\operatorname{prox}_{\eta\lambda}(v_j) =
\left(
1-
\frac{\eta\lambda}{\lVert v_j\rVert_2}
\right)_+
v_j,
$$

where

$$
(a)_+ = \max(a,0).
$$

This update can set both coefficients associated with an edge exactly to zero.

The first call for a new input shape includes JAX compilation time. Subsequent calls with the same shapes can reuse the compiled executable.

## Gibbs Sampling

```python
import numpy as np
import iso_jax as iso

p = 6

theta_c = np.zeros((p, p))
theta_s = np.zeros((p, p))

kappa = 0.8
mu = 0.3

theta_c[0, 1] = kappa * np.cos(mu)
theta_c[1, 0] = theta_c[0, 1]

theta_s[0, 1] = kappa * np.sin(mu)
theta_s[1, 0] = -theta_s[0, 1]

samples = iso.gibbs(
    [theta_c, theta_s],
    n_samples=5000,
    burn_in=2000,
    Q=2,
)

print(samples.shape)
```

## Synthetic Benchmark

Run the default benchmark:

```bash
uv run benchmark_iso.py
```

Run a larger benchmark:

```bash
uv run benchmark_iso.py \
    --n-nodes 24 \
    --average-degree 3 \
    --n-samples 10000 \
    --burn-in 5000 \
    --thin 2 \
    --alpha 2e-6 \
    --jax-warm-repeats 3 \
    --original-parallel \
    --output-dir ./iso_benchmark_results
```

The benchmark compares:

* CVXPY regularized estimates against JAX regularized estimates;
* CVXPY refit estimates against JAX refit estimates;
* all four estimates against the true model;
* parameter-recovery error;
* support precision, recall, and F1;
* cold and warm JAX runtimes;
* CVXPY runtime.

## Comparison Plots

After running the benchmark:

```bash
uv run \
    --with numpy \
    --with matplotlib \
    python plot_iso_comparisons.py \
    --results-dir ./iso_benchmark_results
```

The plotting script generates $y=x$ comparison plots for:

```text
CVXPY regularized vs JAX regularized
CVXPY refit       vs JAX refit
```

It also compares each of the following against the true model:

```text
CVXPY regularized
CVXPY refit
JAX regularized
JAX refit
```

## Likelihood Utilities

```python
scores = iso.unnormalized_log_likelihood(
    Y,
    [theta_c, theta_s],
)
```

```python
likelihoods = iso.unnormalized_likelihood(
    Y,
    [theta_c, theta_s],
)
```

```python
log_z = iso.log_partition(
    [theta_c, theta_s],
    n_samples=100000,
)
```

```python
total_log_likelihood = iso.log_likelihood(
    Y,
    [theta_c, theta_s],
    log_Z=log_z,
)
```

The partition function for a general graph is approximated using Monte Carlo integration.

## Public API

Both implementations expose the following primary functions:

```python
iso.iso(...)
iso.fit(...)
iso.iso_sparse(...)
iso.fit_sparse(...)
iso.gibbs(...)
iso.unnormalized_likelihood(...)
iso.unnormalized_log_likelihood(...)
iso.log_likelihood(...)
iso.log_partition(...)
```

### Regularized fit

```python
theta_c, theta_s = iso.fit(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
)
```

### Two-stage fit

```python
theta_c, theta_s = iso.fit_sparse(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
)
```

### Gibbs sampler

```python
samples = iso.gibbs(
    [theta_c, theta_s],
    n_samples=1000,
    burn_in=100,
    Q=1,
    fsave=None,
)
```

## Implementation Notes

* Input phases must be in radians.
* Rows of `Y` are treated as independent observations.
* The group penalty acts on edge magnitudes rather than on cosine and sine coefficients separately.
* `fit_sparse` performs regularized support selection followed by an unregularized active-set refit.
* The CVXPY and JAX implementations may differ slightly because of solver tolerances and stopping criteria.
* The first JAX call includes compilation overhead.
* General-model partition-function estimation may be computationally expensive.
* This is research software and should be validated for the intended application.

## Scope

The paper introduces:

1. a Chow–Liu tree approximation;
2. interaction screening for general graphical models.

The current Python files focus primarily on the interaction-screening estimator.

## Citation

```bibtex
@article{perley2025graphical,
  title   = {Graphical Models and Efficient Inference Methods for Multivariate Phase Probability Distributions},
  author  = {Perley, Andrew S. and Coleman, Todd P.},
  journal = {arXiv preprint arXiv:2504.00459},
  year    = {2025}
}
```

## License

Add the desired open-source license in `LICENSE` before distributing the repository.


