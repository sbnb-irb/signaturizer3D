import numpy as np
import pytest

from signaturizer3d.input.smiles import generate_conformations, validate_smiles


def test_validate_smiles_filters_invalid():
    smiles_list = ["CCO", "invalid_smiles", "CCC"]

    valid_smiles = validate_smiles(smiles_list)

    assert valid_smiles == ["CCO", "CCC"]


def test_validate_smiles_retains_valid():
    smiles_list = ["CCO", "CCC"]

    valid_smiles = validate_smiles(smiles_list)

    assert valid_smiles == ["CCO", "CCC"]


def test_single_carbon_conformation():
    smiles_list = ["C"]
    seed = 42  # You can use a fixed seed for reproducibility

    atoms_list, coordinates_list = generate_conformations(smiles_list, seed)

    assert atoms_list is not None
    assert coordinates_list is not None
    assert len(atoms_list) == len(coordinates_list)
    assert all(
        len(atoms) == coordinates.shape[0]
        for atoms, coordinates in zip(atoms_list, coordinates_list)
    )

    assert atoms_list == [
        [
            "C",
            "H",
            "H",
            "H",
            "H",
        ]
    ]

    mol_coords = coordinates_list[0]
    assert all(len(coord) == 3 for coord in mol_coords)
    num_zero_rows = np.sum(np.all(mol_coords == 0.0, axis=1))
    assert num_zero_rows < 2, "At most 1 atom can have all zero coords"


@pytest.mark.parametrize(
    "smiles, expected_atoms",
    [
        (["O"], [["O", "H", "H"]]),
        (["CC"], [["C", "C", "H", "H", "H", "H", "H", "H"]]),
    ],
)
def test_multiple_smiles_conformations(smiles, expected_atoms):
    atoms_list, coordinates_list = generate_conformations(smiles, seed=-1)

    assert all(
        len(atoms) == len(coordinates)
        for atoms, coordinates in zip(atoms_list, coordinates_list)
    )
    assert atoms_list == expected_atoms

    for mol_coords in coordinates_list:
        num_zero_rows = np.sum(np.all(mol_coords == 0.0, axis=1))
        assert num_zero_rows < 2, "At most 1 atom can have all zero coords"
