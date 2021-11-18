import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="csaf",
    version="0.0.1",
    author=["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"],
    author_email=["ethanlew16@gmail.com", "mpodhradsky@galois.com", "aditya.zutshi@galois.com"],
    description="Control System Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GaloisInc/csaf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./", "f16lib": "f16lib", "csaf_examples": "csaf_examples"},
    install_requires = [
        'numpy>=1.20.0',
        'numba>=0.54.0',
        'matplotlib>=3.4.3',
        'scipy>=1.7.1',
        'GPyOpt>=1.2.6',
        'tqdm>=4.62.2'
    ],
    packages=setuptools.find_packages(where="./"),
    python_requires=">=3.9",
)
