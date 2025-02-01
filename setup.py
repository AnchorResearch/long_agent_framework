from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="long_agent_framework",
    version="0.1.0",
    author="Anchor Research",
    author_email="",  # TODO: Add your email
    description="A framework for studying long-running AI agent behavior and constraint maintenance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anchor-research/long-agent-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "aideml",  # Assuming this is published or available
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
) 