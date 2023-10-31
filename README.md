# Signaturizer3D

Signaturizers trained on 3D conformations of molecules that are sensitive to the 3D geometry of molecules. The 3D Signaturizers are able to distinguish between stereoisomers. 

![](logo.png)


## Install
Dependencies are managed via poetry. Install them with poetry like this:
```shell
poetry install
```

If you don't like or know poetry you can use conda and install the dependencies directly from `requirements.txt`:
```shell
conda create --name unimol-tools
pip install -r requirements.txt
```
The dependencies in the file are exported directly from poetry.
<!-- TODO: Export requirements to from poetry with CI -->

**install pytorch**
Pytorch needs to be installed outside poetry for now, as it has special install instructions depending on your system and CUDA version,
check https://pytorch.org/get-started/locally/ to ensure you install the right one.
To install for CPU only:
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```

