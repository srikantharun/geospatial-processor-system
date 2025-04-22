from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="geoprocessor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive geospatial data processing toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/geoprocessor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.5b2",
            "flake8>=3.9",
            "mypy>=0.812",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5.2",
            "pre-commit>=2.13",
        ],
        "aws": [
            "boto3>=1.18.0",
            "s3fs>=2021.8.0",
        ],
        "postgis": [
            "psycopg2-binary>=2.9.1",
            "SQLAlchemy>=1.4.0",
            "GeoAlchemy2>=0.9.0",
        ],
        "viz": [
            "folium>=0.12.1",
            "plotly>=5.1.0",
            "holoviews>=1.14.5",
            "geoviews>=1.9.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "geoprocessor=geoprocessor.cli:main",
        ],
    },
)
