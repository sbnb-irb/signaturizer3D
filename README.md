# Signaturizer3D

Signaturizers trained on 3D conformations of molecules that are sensitive to the 3D geometry of molecules. The 3D Signaturizers are able to distinguish between stereoisomers. 

![](logo.png)

## Get started

### Install
Create a virtual environment with Python 3.9 or higher.
```shell
conda create -n sign3D-env python=3.10
```
Install signaturizer3d.
```shell
pip install signaturizer3d
```
Install pytorch. Pytorch needs to be installed separately. Find the correct install
command for your compute platform (CPU, GPU, ...) and install tool (pip, conda) [on this page](https://pytorch.org/get-started/locally/).
Fex, if you want to install with Conda for the CPU only you would run.
```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
### Infer signatures for molecules
Instantiate a signaturizer for one of the 25 bioactivity spaces in the chemical checker:
```python
from signaturizer3d import Signaturizer, CCSpace

signaturizer = Signaturizer(CCSpace.B4)
```
The first time you load a space it will download and cache the model weights
locally.

Infer signaturers from a list of SMILES.
```python
smiles_list = ['C', 'CCC', "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1" ]
signatures = signaturizer.infer_from_smiles(smiles_list)
print(signatures.shape) # -> (3, 128) one 128D vector per molecule
```

Infer signatures from an SDF file or a directory of SDF files by specifying a path.
```python
signatures = signaturizers.infer_from_sdf("/path/to/file.sdf")
```

See the notebook for further examples TODO link

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