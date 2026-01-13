"""
Setup configuration for the classical music recommendation system.

This allows installation as a package:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="classical-music-recommender",
    version="1.0.0",
    description="Production-grade content-based recommendation system for classical music works",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/classical-music-recommender",
    packages=find_packages(include=["recommender", "recommender.*"]),
    python_requires=">=3.11",
    install_requires=[
        "musicbrainzngs>=0.7.1",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
        ],
        "scale": [
            "faiss-cpu>=1.7.4",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="music recommendation classical-music machine-learning content-based-filtering",
    project_urls={
        "Documentation": "https://github.com/yourusername/classical-music-recommender#readme",
        "Source": "https://github.com/yourusername/classical-music-recommender",
        "Bug Reports": "https://github.com/yourusername/classical-music-recommender/issues",
    },
)
