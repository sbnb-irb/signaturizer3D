import pickle
from pathlib import Path

import numpy as np
import pytest

from signaturizer3d.signaturizer import Signaturizer
from signaturizer3d.space import CCSpace


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
    with open(data_dir / "expected_sig4_output_A1.pkl", "rb") as f:
        expected_sigs = pickle.load(f)

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert result is not None
    assert result.shape == (32, 128)
    assert np.allclose(result, expected_sigs, atol=1e-6)


def test_is_inference_repetables(signaturizer, atoms_coords_sig_single):
    atoms_list, coordinates_list, _ = atoms_coords_sig_single

    result1 = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)
    result2 = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert np.allclose(result1, result2)


def test_space_string_is_parsed_as_enum():
    space = "A1"

    signaturizer = Signaturizer(space)

    assert isinstance(signaturizer.space, CCSpace)


def test_accepts_space_enum(mocker):
    space = CCSpace.B4
    mocker.patch("signaturizer3d.signaturizer.FineTunedUniMol")

    signaturizer = Signaturizer(space)

    assert isinstance(signaturizer.space, CCSpace)
    assert signaturizer.space == space


def test_raises_on_invalid_space():
    with pytest.raises(
        ValueError,
        match="A6 is not a valid CCSpace. Valid spaces are A1 ... E5, see the CCSpace enum.",
    ):
        _ = Signaturizer("A6")


def test_sig_inference_from_coordinates_B4():
    signaturizer = Signaturizer(CCSpace.B4)
    data_dir = Path.cwd() / "tests" / "data"
    with open(data_dir / "atoms_list.pkl", "rb") as f:
        atoms_list = pickle.load(f)
    with open(data_dir / "coordinates_list.pkl", "rb") as f:
        coordinates_list = pickle.load(f)
    with open(data_dir / "expected_sig4_output_B4_CPU.pkl", "rb") as f:
        expected_sigs = pickle.load(f)

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert result is not None
    assert result.shape == (32, 128)
    assert np.allclose(result, expected_sigs, atol=1e-6)


def test_sig_inference_from_coordinates_cpu_equals_gpu():
    signaturizer = Signaturizer(CCSpace.B4)
    data_dir = Path.cwd() / "tests" / "data"
    with open(data_dir / "atoms_list.pkl", "rb") as f:
        atoms_list = pickle.load(f)
    with open(data_dir / "coordinates_list.pkl", "rb") as f:
        coordinates_list = pickle.load(f)
    with open(data_dir / "expected_sig4_output_B4_CPU.pkl", "rb") as f:
        expected_sigs = pickle.load(f)

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    assert result is not None
    assert result.shape == (32, 128)
    assert np.allclose(result, expected_sigs, atol=1e-6)


@pytest.fixture
def sdf_C(tmp_path):
    sdf_content = """
CT1001789336


  5  4  0  0  0               999 V2000
    0.0021   -0.0041    0.0020 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0127    1.0858    0.0080 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0099    1.4631    0.0003 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5399    1.4469   -0.8751 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5229    1.4373    0.9048 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$
"""
    sdf_file = tmp_path / "valid.sdf"
    with open(sdf_file, "w") as file:
        file.write(sdf_content.strip())
    return str(sdf_file)


def test_inference_from_sdf(signaturizer, sdf_C, signature_C):
    # The expected signature is that of a single carbon atom
    # since the sdf file contains one C and 4 H atoms that will
    # be removed before signature inference
    _, expected_signature = signature_C

    result = signaturizer.infer_from_sdf(sdf_C)

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (1, 128)
    assert np.allclose(result, expected_signature, atol=1e-6)
