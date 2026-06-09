from setuptools import setup, find_packages

setup(
    name="vllm-monitor",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "nvidia-ml-py>=13.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.22"],
        "fallback": ["torch>=2.0.0"],
    },
    python_requires=">=3.10",
)
