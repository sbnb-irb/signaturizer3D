# Where are the model weights uploaded?
The fine tuned model weights are uploaded to an empty github repo as a set of binary files for a release:
https://github.com/aksell/test-pytorch-modelhub/releases/tag/full-CC

The weights are uploaded as a github release because releases have no size limit,
and the 25 models total ~14GB, which means they're way to large for the sbnb gitlab
release max size of 10MB. 
Putting the weighs in the gitlab repo would bloat the repo massively
and it suck majorly to clone it, especially if we release an updated set of weights.

A new release with updated weights could be created like this, here the `/release/` dir holds
the model weights.
```
gh repo clone aksell/test-pytorch-modelhub
cd test-pytorch-modelhub
gh release create full-CC /aloy/home/alenes/signaturizer3d/weights/release/*
```
