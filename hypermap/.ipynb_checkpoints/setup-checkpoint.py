from setuptools import setup, find_packages

setup(
    name="hypermap",
    version="0.1.0",
    description="Meta-learning model for single-cell perturbation prediction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12",
        "higher>=0.2.1",
        "numpy",
        "pandas",
        "scanpy",
        "scipy",
        "tqdm",
    ],
)
