from setuptools import setup, find_packages
import os

setup(
    name="codex",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "tenacity"
    ],
    entry_points={
        "console_scripts": [
            "codex=codex.main:main",
        ],
    },
    author="Muzaffar Mohamed",
    author_email="exrienz@gmail.com",
    description="CLI wrapper for Ollama GPT API",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codex",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
