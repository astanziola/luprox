from typing import Any, Callable, Dict
from jax import numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, random


def linear_uncertainty(fun: Callable):
    """See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming"""

    def fun_with_uncertainty(mean, covariance, *args, **kwargs):
        mean = mean.real
        covariance = covariance.real

        def f(x):
            return fun(x, *args, **kwargs)

        # Getting output meand and covariance
        out_mean = f(mean)

        if out_mean.size < mean.size:
            J = jacrev(f)(mean)
        else:
            J = jacfwd(f)(mean)

        out_cov = jnp.matmul(J, jnp.matmul(covariance, J.T))  # this factor of 4 is odd
        return out_mean, out_cov

    return jit(fun_with_uncertainty)


def monte_carlo(fun: Callable, trials):
    def sampling_function(mean, covariance, key):
        def _sample(mean, L, key, *args, **kwargs):
            noisy_x = mean + jnp.dot(L, random.normal(key, mean.shape))
            return fun(noisy_x, *args, **kwargs)

        mean = mean.real
        covariance = covariance.real
        keys = random.split(key, trials)

        L = jnp.linalg.cholesky(covariance)
        samples = vmap(_sample, in_axes=(None, None, 0))(mean, L, keys)

        return samples

    return jit(sampling_function)


def mc_uncertainty(fun: Callable, trials):
    def fun_with_uncertainty(mean, covariance, key):
        samples = monte_carlo(fun, trials)(mean, covariance, key)
        return jnp.mean(samples, axis=0), jnp.cov(samples.T)

    return jit(fun_with_uncertainty)
