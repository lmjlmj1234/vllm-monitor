from setuptools import setup, find_packages

setup(
    name="vllm-monitor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "psutil>=5.9.0",
        "torch>=2.1.0",
    ],
)