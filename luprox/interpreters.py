from typing import Any, Callable, Dict
from jax import numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, random, eval_shape


def linear_uncertainty(fun: Callable):
    """See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming"""

    def fun_with_uncertainty(mean, covariance, *args, **kwargs):
        mean = mean.real
        covariance = covariance.real

        out_shape = eval_shape(fun, mean, *args, **kwargs).shape

        def f(x):
            y = fun(x, *args, **kwargs)
            return jnp.ravel(y)

        # Getting output meand and covariance
        out_mean = f(mean)
        J = jacfwd(f)(mean)

        out_cov = (jnp.abs(J)**2) @ jnp.sqrt(covariance)# this factor of 4 is odd
        del J
        out_cov = jnp.reshape(out_cov, out_shape)
        out_mean = jnp.reshape(out_mean, out_shape)

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
        meanval = 0
        var = 0
        for i in range(trials):
            sample = _sample(mean, L, keys[i])
            meanval = meanval + sample/trials
            del sample

        for i in range(trials):
            sample = _sample(mean, L, keys[i])
            var = var + jnp.abs(sample-meanval)**2/trials
            del sample
        return meanval, var

    return sampling_function


def mc_uncertainty(fun: Callable, trials):
    def fun_with_uncertainty(mean, covariance, key):
        return monte_carlo(fun, trials)(mean, covariance, key)
    return fun_with_uncertainty
