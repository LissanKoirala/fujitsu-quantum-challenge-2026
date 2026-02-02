"""Setup configuration for Fujitsu Quantum Challenge project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fujitsu-quantum-challenge-2026",
    version="0.1.0",
    author="Fujitsu Quantum Challenge 2026 Team",
    description="Quantum algorithms for the Fujitsu Quantum Challenge 2026",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LissanKoirala/fujitsu-quantum-challenge-2026",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "qiskit>=0.46.0",
        "qiskit-aer>=0.13.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pylint>=2.18.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
)
