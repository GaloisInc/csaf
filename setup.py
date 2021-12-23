import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="csaf-controls",
    version="0.2",
    author="Ethan Lew, Michal Podhradsky, Aditya Zutshi",
    author_email="ethanlew16@gmail.com, mpodhradsky@galois.com, aditya.zutshi@galois.com",
    description="Control Systems Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GaloisInc/csaf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ".", "csaf_f16": "csaf_f16", "csaf_examples": "csaf_examples"},
    install_requires = [
        'numpy>=1.20.0',
        'numba>=0.54.0',
        'matplotlib>=3.4.3',
        'scipy>=1.7.1',
        'GPyOpt>=1.2.6',
        'tqdm>=4.62.2',
        'onnxruntime>=1.7.0',
        'pydot>=1.4.1',
        'svgpath2mpl>=1.0.0'
    ],
    # TODO: remove this
    include_package_data=True,
    package_data={'csaf_f16': ['models/trained_models/*.onnx', 'models/trained_models/np/*.npz']},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
)
