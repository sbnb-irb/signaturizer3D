import pickle
from pathlib import Path

import numpy as np


def test_sig_inference_from_coordinates(signaturizer, atoms_coords_sig_single):
    atoms_list, coordinates_list, expected_sig = atoms_coords_sig_single

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (1, 128)
    assert np.allclose(result, expected_sig, atol=1e-6)


def test_sig_inference_from_coordinates_full(signaturizer):
    data_dir = Path.cwd() / "tests" / "data"
    with open(data_dir / "atoms_list.pkl", "rb") as f:
        atoms_list = pickle.load(f)
    with open(data_dir / "coordinates_list.pkl", "rb") as f:
        coordinates_list = pickle.load(f)
    with open(data_dir / "expected_sig4_output.pkl", "rb") as f:
        expected_sigs = pickle.load(f)

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert result is not None
    assert result.shape == (32, 128)
    assert np.allclose(result, expected_sigs, atol=1e-8)


def test_is_inference_repetables(signaturizer, atoms_coords_sig_single):
    atoms_list, coordinates_list, _ = atoms_coords_sig_single

    result1 = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)
    result2 = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert np.allclose(result1, result2)
