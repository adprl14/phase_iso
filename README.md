# phase_iso

# Graphical Models for Multivariate Phase Relationships

Research code for fitting graphical models to multivariate phase data using the interaction screening objective described in:

> Andrew S. Perley and Todd P. Coleman,
> **“Graphical Models and Efficient Inference Methods for Multivariate Phase Probability Distributions.”**
> arXiv:2504.00459
> https://arxiv.org/abs/2504.00459

The repository provides:

* a reference implementation using CVXPY;
* a faster JAX implementation with the same public API;
* regularized graph-structure estimation;
* two-stage support selection and unregularized parameter refitting;
* Gibbs sampling from the fitted circular graphical model;
* likelihood and approximate partition-function utilities;
* scripts for comparing the CVXPY and JAX implementations.

## Model

Let

[
Y=(Y_1,\ldots,Y_p)\in[-\pi,\pi)^p
]

be a vector of phases. The model has density

[
p(y)
====

\frac{1}{Z}
\exp\left[
\sum_{(i,j)\in E}
\kappa_{ij}
\cos(y_j-y_i-\mu_{ij})
\right],
]

where:

* (\kappa_{ij}\geq 0) is the coupling magnitude between nodes (i) and (j);
* (\mu_{ij}) is the preferred phase difference;
* (E) is the graphical-model edge set;
* (Z) is the partition function.

The implementation uses the natural-parameter representation

[
\theta_{ij,c}=\kappa_{ij}\cos(\mu_{ij}),
\qquad
\theta_{ij,s}=\kappa_{ij}\sin(\mu_{ij}).
]

The original parameters can be recovered using

[
\kappa_{ij}
===========

\sqrt{\theta_{ij,c}^2+\theta_{ij,s}^2},
\qquad
\mu_{ij}
========

\operatorname{atan2}(\theta_{ij,s},\theta_{ij,c}).
]

An edge is absent when (\kappa_{ij}=0).

## Interaction Screening

For each node (u), the interaction screening objective estimates the parameters of edges incident to that node without evaluating the model’s high-dimensional partition function.

The regularized estimator minimizes

[
S_n(\theta_u)
+
\lambda
\sum_{j\neq u}
\sqrt{\theta_{uj,c}^2+\theta_{uj,s}^2}.
]

The penalty is a group-lasso penalty: the cosine and sine parameters for each edge are treated as one group. This allows an entire edge to be selected or removed.

The default regularization parameter is

[
\lambda
=======

4\sqrt{
\frac{\log(8p^2/\epsilon)}{n}
},
]

with (\epsilon=0.1) in the current implementation.

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
* joblib for nodewise CPU parallelization.

### `iso_jax.py`

JAX implementation with the same primary function names and argument order as `iso.py`.

To switch implementations, change only the import:

```python
# Reference CVXPY implementation
import iso

# JAX implementation
import iso_jax as iso
```

Existing calls such as `iso.fit(...)` and `iso.fit_sparse(...)` can remain unchanged.

### `benchmark_iso.py`

Generates a synthetic circular graphical model, samples data, and compares:

* CVXPY regularized estimates;
* JAX regularized estimates;
* CVXPY two-stage refit estimates;
* JAX two-stage refit estimates;
* estimation error relative to the true parameters;
* runtime, including cold and warm JAX timings.

### `plot_iso_comparisons.py`

Generates (y=x) comparison plots from the benchmark outputs, including:

* every estimate against the true parameters;
* CVXPY regularized against JAX regularized;
* CVXPY refit against JAX refit.

## Installation

Python 3.10 or newer is recommended.

### Using `uv`

Create an environment and install the dependencies:

```bash
uv venv
source .venv/bin/activate

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

Without MOSEK, the reference implementation falls back to SCS.

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
    numpy \
    scipy \
    cvxpy \
    joblib \
    jax \
    matplotlib
```

## Input Data

The fitting functions expect a NumPy array with shape

```text
(n_samples, n_nodes)
```

Each entry must be a phase in radians, typically represented on

```text
[-π, π)
```

For example:

```python
import numpy as np

Y = np.load("phases.npy")

print(Y.shape)
# (n_samples, n_nodes)

Y = np.angle(np.exp(1j * Y))
```

The current estimator treats rows as independent observations.

## Basic Usage

### Regularized structure estimation

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit(
    Y,
    lam=None,
    alpha=None,
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

The returned matrices satisfy approximately

```python
np.allclose(theta_c, theta_c.T)
np.allclose(theta_s, -theta_s.T)
```

Thus:

* `theta_c` is symmetric;
* `theta_s` is skew-symmetric;
* `kappa` is symmetric;
* `mu[j, i] = -mu[i, j]`, modulo (2\pi).

## Two-Stage Sparse Estimation

The paper uses a two-stage procedure for improved parameter estimation:

1. fit the group-lasso regularized model to identify an edge set;
2. refit an unregularized model using only the selected edges.

Use:

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    lam=None,
    alpha=2e-6,
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

The first stage includes the group-lasso penalty. The second stage has no group-lasso penalty, but coefficients outside the selected support are not estimated and remain zero.

Therefore, the regularized and refitted coefficients are not expected to be identical. The refit removes shrinkage bias from the selected coefficients.

## Thresholding with `alpha`

The optional `alpha` argument performs hard thresholding after the regularized fit.

For each candidate edge, the implementation computes

[
\widehat{\kappa}_{ij}
=====================

\sqrt{
\widehat{\theta}*{ij,c}^2
+
\widehat{\theta}*{ij,s}^2
}.
]

The edge is set to zero when

[
\widehat{\kappa}_{ij}<\frac{\alpha}{2}.
]

For example:

```python
theta_c, theta_s = iso.fit_sparse(
    Y,
    alpha=2e-6,
)
```

uses an effective first-stage threshold of (10^{-6}).

A small positive threshold can prevent numerical solver residuals from being interpreted as selected edges in the CVXPY implementation.

Setting

```python
alpha=None
```

disables this explicit hard-thresholding step.

## Choosing the Regularization Parameter

By default, `lam=None` uses the theoretical scaling implemented in the code:

```python
epsilon = 0.1
lam = 4 * np.sqrt(
    np.log(8 * n_nodes**2 / epsilon) / n_samples
)
```

A custom value can be supplied:

```python
theta_c, theta_s = iso.fit(
    Y,
    lam=0.1,
)
```

Larger values generally produce sparser estimated graphs. Smaller values generally retain more candidate edges.

## Using the Reference CVXPY Implementation

```python
import numpy as np
import iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    lam=None,
    alpha=2e-6,
    parallel=True,
    verbose=False,
)
```

When `parallel=True`, the reference implementation fits nodewise problems using joblib.

MOSEK is attempted first. If MOSEK is unavailable or fails, the implementation falls back to SCS.

## Using the JAX Implementation

```python
import numpy as np
import iso_jax as iso

Y = np.load("phases.npy")

theta_c, theta_s = iso.fit_sparse(
    Y,
    lam=None,
    alpha=2e-6,
    parallel=False,
    verbose=True,
)
```

The JAX implementation uses a specialized proximal-gradient solver for the regularized objective.

The first call for a new input shape includes JAX compilation time. Subsequent calls with the same shapes reuse the compiled executable.

The `parallel` argument is accepted for API compatibility but is not currently used to parallelize the nodewise Python loop.

## Synthetic Example

The repository can also generate samples from a known model.

```python
import numpy as np
import iso_jax as iso

p = 6

theta_c_true = np.zeros((p, p))
theta_s_true = np.zeros((p, p))

edges = [
    (0, 1, 0.8, 0.2),
    (1, 2, 0.7, -0.4),
    (2, 3, 0.9, 0.6),
    (3, 4, 0.6, -0.3),
    (4, 5, 0.8, 0.1),
    (0, 5, 0.7, -0.5),
]

for i, j, kappa, mu in edges:
    theta_c = kappa * np.cos(mu)
    theta_s = kappa * np.sin(mu)

    theta_c_true[i, j] = theta_c
    theta_c_true[j, i] = theta_c

    theta_s_true[i, j] = theta_s
    theta_s_true[j, i] = -theta_s

Y = iso.gibbs(
    [theta_c_true, theta_s_true],
    n_samples=5000,
    burn_in=2000,
    Q=2,
)

theta_c_hat, theta_s_hat = iso.fit_sparse(
    Y,
    lam=None,
    alpha=2e-6,
)

kappa_hat = np.hypot(theta_c_hat, theta_s_hat)
mu_hat = np.arctan2(theta_s_hat, theta_c_hat)

print("True coupling magnitudes")
print(np.round(np.hypot(theta_c_true, theta_s_true), 3))

print("Estimated coupling magnitudes")
print(np.round(kappa_hat, 3))
```

## Benchmarking CVXPY and JAX

A typical benchmark command is:

```bash
uv run \
    --with numpy \
    --with scipy \
    --with cvxpy \
    --with mosek \
    --with joblib \
    --with jax \
    --with matplotlib \
    python benchmark_iso.py \
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

The benchmark reports:

* regularized estimation error;
* refit estimation error;
* support precision, recall, and F1;
* coupling-magnitude error;
* phase-offset error;
* CVXPY runtime;
* JAX cold runtime;
* JAX warm runtime.

## Generating Comparison Plots

After running the benchmark:

```bash
uv run \
    --with numpy \
    --with matplotlib \
    python plot_iso_comparisons.py \
        --results-dir ./iso_benchmark_results
```

The plotting script compares all four estimates against truth and performs matched cross-implementation comparisons:

```text
CVXPY regularized vs JAX regularized
CVXPY refit       vs JAX refit
```

The figures use identical axis limits and include a dashed (y=x) reference line.

## Likelihood Utilities

The modules expose:

```python
iso.unnormalized_log_likelihood(Y, thetas)
iso.unnormalized_likelihood(Y, thetas)
iso.log_partition(thetas, n_samples=None)
iso.log_likelihood(Y, thetas, log_Z=None)
```

The partition function for a general graph is difficult to evaluate exactly. The current implementation estimates it using Monte Carlo integration with uniformly sampled phase vectors.

For comparing models on the same data, the unnormalized log likelihood can be computed using:

```python
scores = iso.unnormalized_log_likelihood(
    Y,
    [theta_c, theta_s],
)
```

## Public API

Both implementations expose:

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

### `fit`

```python
fit(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
)
```

Returns:

```python
[theta_c, theta_s]
```

for the regularized estimator.

### `fit_sparse`

```python
fit_sparse(
    Y,
    beta=None,
    alpha=None,
    lam=None,
    parallel=False,
    verbose=False,
)
```

Returns:

```python
[theta_c, theta_s]
```

after regularized support selection and unregularized active-set refitting.

### `gibbs`

```python
gibbs(
    thetas,
    n_samples=1000,
    burn_in=100,
    Q=1,
    fsave=None,
)
```

where `Q` is the number of Gibbs sweeps between retained samples.

## Implementation Notes

* Input phases must be in radians.
* The current model assumes independent samples.
* `fit()` and `fit_sparse()` solve separate nodewise objectives and then combine the nodewise estimates.
* The regularization is applied to edge magnitudes, not separately to cosine and sine coefficients.
* `fit_sparse()` selects edges using the first-stage coupling magnitudes and then removes the penalty during refitting.
* The current JAX implementation preserves the original API but may not produce bitwise-identical numerical results to MOSEK or SCS.
* The `beta` argument is retained for compatibility but is not enforced by the current implementations.
* General-model partition-function estimation can be computationally expensive.
* This is research code and should be validated for the intended application.

## Scope

The paper describes two inference approaches:

1. a Chow–Liu dependence-tree approximation;
2. interaction screening for arbitrary graphical-model structures.

The Python files in this repository currently focus on the interaction-screening estimator for the full graphical model. A Chow–Liu implementation is not included unless added separately.

## Citation

Please cite the accompanying paper when using this code:

```bibtex
@article{perley2025graphical,
  title   = {Graphical Models and Efficient Inference Methods for Multivariate Phase Probability Distributions},
  author  = {Perley, Andrew S. and Coleman, Todd P.},
  journal = {arXiv preprint arXiv:2504.00459},
  year    = {2025}
}
```

## License

Add the appropriate open-source license before distributing the repository. See the `LICENSE` file for the selected terms.
