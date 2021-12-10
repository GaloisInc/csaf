import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="csaf",
    version="0.0.1",
    author=["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"],
    author_email=["ethanlew16@gmail.com", "mpodhradsky@galois.com", "aditya.zutshi@galois.com"],
    description="Control Systems Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GaloisInc/csaf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ".", "csaf_f16": "csaf_f16", "csaf_examples": "csaf_examples"},
    install_requires = [
        'numpy>=1.20.0',
        'numba>=0.54.0',
        'matplotlib>=3.4.3',
        'scipy>=1.7.1',
        'GPyOpt>=1.2.6',
        'tqdm>=4.62.2'
    ],
    include_package_data=True,
    package_data={'csaf_f16': ['./csaf_f16/models/trained_models/*.onnx', './csaf_f16/models/trained_models/np/*.npz']},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
)
