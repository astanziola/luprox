from setuptools import setup, find_packages

setup(
    name="luprox",
    version="0.0.1",
    description="Linear uncertainty propagation using JAX transformations",
    author="Antonio Stanziola",
    author_email="a.stanziola@ucl.ac.uk",
    packages=["luprox"]
    install_requires=["jax", "jaxlib"]
)