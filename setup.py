from setuptools import setup, find_packages

setup(
    name="llm-formal-reasoning",
    version="0.1.0",
    description="Testing the Limits of LLMs on Regular Languages",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "trl>=0.8.6",
        "accelerate>=0.28.0",
        "datasets>=2.18.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "frontier": [
            "openai>=1.20.0",
            "anthropic>=0.25.0",
            "google-generativeai>=0.5.0",
        ],
        "dev": ["pytest>=7.0", "black>=23.0", "isort>=5.0"],
    },
)
