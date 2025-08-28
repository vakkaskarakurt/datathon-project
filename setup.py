from setuptools import setup, find_packages

setup(
    name="ecommerce-session-prediction",
    version="1.0.0",
    description="E-commerce session value prediction using advanced ML techniques",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.1.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.6.0",
        "catboost>=1.1.0",
        "polars>=0.15.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)