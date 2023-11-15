<div align="center">
    <img src="https://github.com/aksell/test-pytorch-modelhub/assets/22766894/e4e223f1-66d8-4a85-ad4e-6e5849917a2c" alt="signaturizer3D" width="40%"/>
    <center><h1>signaturizer3D</h1></center>
</div>

<br/>

<div align="center">
<h3>Infer the bioactivity of molecules using models trained on the 3D structure of molecules</h3>
</div>


---
## Bioactivity signaturizers

This package builds on the original [signaturizers](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer) package ([paper](https://www.nature.com/articles/s41467-021-24150-4)) by applying a state of the art 3D transformer model for molecular representation ([Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)) to the task of infering
bioactivity.

Bioactivity signatures are multi-dimensional vectors that capture biological
traits of a molecule (fex its target profile) in a numerical vector
format that is akin to the structural descriptors or fingerprints used in the
field of chemoinformatics.

The **signaturizer3D** models infer bioactivity in 25 bioactivity types (including
target profiles, cellular response and clinical outcomes) and can be used as
drop-in replacements for chemical descriptors in day-to-day chemoinformatics
tasks.

For and overview of the different bioctivity types available see the original Chemical
 Checker [paper](https://www.nature.com/articles/s41587-020-0502-7) or [website](https://chemicalchecker.com/).
## Get started

### Install
Create a virtual environment with Python 3.9 or higher.
```shell
conda create -n sign3D-env python=3.10
conda activate sign3D-env
```
Install the package with pip (use PyPi test for now, as the package isn't published on PyPi yet)
```shell
pip install tqdm pandas rdkit-pypi # Install package not published to test.pypi from pypi first
pip install -i https://test.pypi.org/simple/ signaturizer3d
# python -m pip install signaturizer3d # This will be the only command once the package is published to pypi
```
Install pytorch. Pytorch needs to be installed separately. Find the correct install
command for your compute platform (CPU, GPU, ...) and install tool (pip, conda) [on this page](https://pytorch.org/get-started/locally/).
Fex, if you want to install with Conda for the CPU only you would run.
```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

If you're using singularity containers there is an example image definition file [singularity.def](https://gitlabsbnb.irbbarcelona.org/alenes/signaturizer3d/-/blob/main/singularity.def) that shows how you can install
the package with cuda and pytorch.
### Infer signatures for molecules
Instantiate a signaturizer for one of the 25 bioactivity spaces in the chemical checker:
```python
from signaturizer3d import Signaturizer, CCSpace

CCSpace.print_spaces() # Prints a description of the 25 available spaces

signaturizer = Signaturizer(CCSpace.B4)
```
The first time you load a space it will download and cache the model weights
locally.

Infer signaturers from a list of SMILES.
```python
smiles_list = ['C', 'CCC', "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1" ]
signatures = signaturizer.infer_from_smiles(smiles_list)
print(signatures.shape) # -> (3, 128) a 128D vector per molecule
```

Infer signatures from an SDF file or a directory of SDF files by specifying a path.
```python
signatures = signaturizers.infer_from_sdf("/path/to/file.sdf")
```

See this [notebook](https://gitlabsbnb.irbbarcelona.org/alenes/signaturizer3d/-/blob/main/notebooks/infer_signatures.ipynb) for more detailed examples of signaturizer usage.

For a more comprehensive example of using infered bioactivity signatures for analysing similarity between a set of compounds 
have a look at the [example notebook](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer/-/blob/master/notebook/signaturizer.ipynb) in the original signaturizers package.

## Development
Guidelines on how to set up the development environment and run tests.

### Install dependencies locally
Dependencies are managed via [poetry](https://python-poetry.org/). Install them with poetry by running this inside
the project directory:
```shell
poetry install
```

Install pytorch (the project has been tested with pytorch 2.1). Pytorch needs to be installed outside poetry for now. Find the correct install
command for your compute platform (CPU, GPU, ...) and install tool [on this page](https://pytorch.org/get-started/locally/).

### Run tests
Run all tests.
```shell
poetry run pytest
```

By default tests marked with `performance` are excluded. These tests test runtime of different
components. This will be very system dependent, therefore they are excluded by default.
If you want to use these tests to monitor changes in performance as you change the code you
should run the tests on your system before making any changes and update the time threshold to
that before making any changes. 
Run the `performance` tests with:
```shell
poetry run pytest -m 'performance'
```

### Documentation
For more information about the package and unimol check out the [docs](docs/index.md)