from setuptools import setup, find_packages

setup(
    name="neuron_explainer",
    packages=find_packages(),
    version="0.0.1",
    author="OpenAI",
    install_requires=[
        "httpx>=0.22",
        "scikit-learn",
        "boostedblob>=0.13.0",
        "tiktoken",
        "blobfile",
        "numpy",
        "pytest",
        "orjson",
    ],
    url="",
    description="",
    python_requires='>=3.9',
)
