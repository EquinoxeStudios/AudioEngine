"""
Audio Engine Setup Script

Professional therapeutic noise generator setup for PyPI distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

setup(
    name="audio-engine",
    version="1.0.0",
    author="Audio Engine Team",
    author_email="contact@audioengine.com",
    description="Professional therapeutic noise generator for YouTube content creation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/AudioEngine",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/AudioEngine/issues",
        "Source": "https://github.com/your-repo/AudioEngine",
        "Documentation": "https://github.com/your-repo/AudioEngine/wiki"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Content Creators",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Multimedia :: Sound/Audio :: Generators",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Console"
    ],
    keywords=[
        "audio", "noise", "therapeutic", "white-noise", "pink-noise", "brown-noise",
        "youtube", "lufs", "infant", "sleep", "relaxation", "cuda", "gpu",
        "professional-audio", "audio-generation", "therapeutic-audio"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0"
        ],
        "performance": [
            "numba>=0.56.0",
            "cupy-cuda11x>=11.0.0"
        ],
        "colab": [
            "google-colab",
            "ipywidgets"
        ]
    },
    entry_points={
        "console_scripts": [
            "audio-engine=audio_engine.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "audio_engine": [
            "*.md",
            "examples/*.py",
            "tests/*.py"
        ]
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests"
)