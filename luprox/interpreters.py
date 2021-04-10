from typing import Any, Callable, Dict
from jax import numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, random


def linear_uncertainty(fun: Callable):
    """See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming"""

    def fun_with_uncertainty(mean, covariance):
        mean = mean.real
        covariance = covariance.real

        # Getting output meand and covariance
        out_mean = fun(mean)

        if out_mean.size < mean.size:
            J = jacrev(fun)(mean)
        else:
            J = jacfwd(fun)(mean)

        out_cov = jnp.matmul(J, jnp.matmul(covariance,J.T)) # this factor of 4 is odd
        return out_mean, out_cov

    return jit(fun_with_uncertainty)


def mc_uncertainty(fun: Callable, trials):

    def _sample(mean, L, key):
        noisy_x = mean + jnp.dot(L, random.normal(key, mean.shape))
        return fun(noisy_x)

    get_samples = vmap(_sample, in_axes=(None, None, 0))

    def fun_with_uncertainty(mean, covariance, key):
        mean = mean.real
        covariance = covariance.real
        keys = random.split(key, trials)

        L = jnp.linalg.cholesky(covariance)
        samples = get_samples(mean, L, keys)
        return jnp.mean(samples, axis=0), jnp.cov(samples.T)

    return jit(fun_with_uncertainty)
