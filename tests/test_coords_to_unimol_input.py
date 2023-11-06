import pathlib

import numpy as np
import pytest

from signaturizer3d.data.unimol import coordinates_to_unimol
from signaturizer3d.unicore.dictionary import Dictionary


@pytest.fixture
def dictionary():
    DICT_PATH = (
        pathlib.Path(__file__).resolve().parents[1]
        / "signaturizer3d/models/"
        / "dict.txt"
    )
    dictionary = Dictionary.load(DICT_PATH.as_posix())
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


def test_remove_hydrogens(dictionary):
    atoms = ["H", "C", "H", "O", "C"]
    coords = np.random.rand(5, 3).astype(np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary, remove_hs=True)

    non_h_atom_count = sum(atom != "H" for atom in atoms)
    assert (
        len(result["src_tokens"]) == non_h_atom_count + 2
    ), "Hydrogen atoms should be removed"


def test_max_atoms_limit(dictionary):
    atoms = ["C"] * 300  # More than max_atoms
    coords = np.random.rand(300, 3).astype(np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary, max_atoms=256)

    assert (
        len(result["src_tokens"]) <= 256 + 2
    ), "Number of tokens should not exceed max_atoms + 2 (for start and end tokens)"


def test_tokenization(dictionary):
    atoms = ["C", "O"]
    coords = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    expected_tokens = (
        [dictionary.bos()]
        + [dictionary.index(atom) for atom in atoms]
        + [dictionary.eos()]
    )
    np.testing.assert_array_equal(
        result["src_tokens"], expected_tokens, "Tokens do not match expected values"
    )


def test_padding(dictionary):
    atoms = ["C"]
    coords = np.array([[1, 0, 0]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    assert len(result["src_tokens"]) == 3
    assert len(result["src_coord"]) == 3
    assert result["src_coord"][0].sum() == 0 and result["src_coord"][-1].sum() == 0


def test_coordinate_normalization(dictionary):
    atoms = ["C", "O"]
    coords = np.array([[5, 5, 5], [10, 10, 10]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    expected_coords_mean = np.array(
        [[0.0, 0.0, 0.0], [-2.5, -2.5, -2.5], [2.5, 2.5, 2.5], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(result["src_coord"], expected_coords_mean, atol=1e-7)


def test_distance_matrix(dictionary):
    atoms = ["C", "O"]
    coords = np.array([[0.5, 0, 0], [-0.5, 0, 0]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    # Compute expected distance matrix manually or with a known-good function
    expected_distance_matrix_atoms = np.array([[0, 1], [1, 0]], dtype=np.float32)
    atom_distances = result["src_distance"][1:3, 1:3]
    np.testing.assert_allclose(
        atom_distances, expected_distance_matrix_atoms, atol=1e-7
    )
    expected_distance_matrix = np.array(
        [
            [0.0, 0.5, 0.5, 0.0],
            [0.5, 0.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 0.5],
            [0.0, 0.5, 0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(result["src_distance"], expected_distance_matrix)


def test_edge_type_array(dictionary):
    atoms = ["C", "O"]
    coords = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    assert result["src_edge_type"].shape == (4, 4)
    expected_edge_types = np.array(
        [
            [32, 35, 37, 33],
            [125, 128, 130, 126],
            [187, 190, 192, 188],
            [63, 66, 68, 64],
        ]  # ....
    )
    assert np.allclose(result["src_edge_type"], expected_edge_types)


def test_return_type(dictionary):
    atoms = ["C", "O"]
    coords = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    result = coordinates_to_unimol(atoms, coords, dictionary)

    assert isinstance(result, dict)
    assert all(isinstance(result[key], np.ndarray) for key in result)

    expected_tokens = np.array([1, 4, 6, 2])
    expected_distances = np.array(
        [
            [0.0, 0.70710677, 0.70710677, 0.0],
            [0.70710677, 0.0, 1.4142135, 0.70710677],
            [0.70710677, 1.4142135, 0.0, 0.70710677],
            [0.0, 0.70710677, 0.70710677, 0.0],
        ],
    )
    excpected_src_coord = np.array(
        [[0.0, 0.0, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
    )
    expected_edge_type = np.array(
        [[32, 35, 37, 33], [125, 128, 130, 126], [187, 190, 192, 188], [63, 66, 68, 64]]
    )
    assert np.array_equal(result["src_tokens"], expected_tokens)
    assert np.allclose(result["src_distance"], expected_distances)
    assert np.allclose(result["src_coord"], excpected_src_coord)
    assert np.array_equal(result["src_edge_type"], expected_edge_type)
