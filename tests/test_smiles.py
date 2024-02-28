import numpy as np
import pytest


@pytest.mark.skip(reason="Weights outdated")
def test_infer_from_smiles(signaturizer, signature_C):
    smiles, expected_sig = signature_C
    smiles_list = [smiles]

    result = signaturizer.infer_from_smiles(smiles_list)

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (1, 128)
    assert np.allclose(result, expected_sig)
