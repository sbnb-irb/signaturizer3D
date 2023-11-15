from typing import List, Union

import numpy as np
import numpy.typing as npt

from signaturizer3d.input.smiles import generate_conformations, validate_smiles
from signaturizer3d.unicore.dictionary import Dictionary


def pairwise_distance_numpy(coords: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # Calculate pairwise distance matrix using broadcasting and vector norm
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix.astype(np.float32)


def coordinates_to_unimol(
    atoms: List[str],
    coordinates: Union[npt.NDArray[np.float32], List[List[float]]],
    dictionary: Dictionary,
    max_atoms: int = 256,
    remove_hs: bool = True,
) -> dict:
    """
    Converts atom symbols and coordinates to input in unimol's expected representation.

    Parameters:
    atoms (List[str]): List of atom symbols.
    coordinates (numpy array): (n_atoms, 3) numpy array of x, y, z coordinates for each atom.
    dictionary (Dictionary): A Uni-Core dictionay
    max_atoms (int): Maximum number of atoms to consider in the molecule
    remove_hs (bool): Whether to remove hydrogen atoms from the representation.

    Returns:
    dict: A dictionary with keys `src_tokens`, `src_distance`, `src_coord`, `src_edge_type`.
    """
    coordinates = np.array(coordinates).astype(np.float32)
    atoms_array = np.array(atoms)

    if remove_hs:
        non_hydrogen_mask = atoms_array != "H"
        atoms_array = atoms_array[non_hydrogen_mask]
        coordinates_array = coordinates[non_hydrogen_mask]

    # Randomly select atoms if more than max_atoms
    if len(atoms_array) > max_atoms:
        selected_indices = np.random.choice(len(atoms_array), max_atoms, replace=False)
        atoms_array = atoms_array[selected_indices]
        coordinates_array = coordinates_array[selected_indices]

    # Create source tokens with start and end tokens
    src_tokens = np.array(
        [dictionary.bos()]
        + [dictionary.index(atom) for atom in atoms_array]
        + [dictionary.eos()],
    )

    # Normalize coordinates and add padding for start and end tokens
    src_coord = coordinates_array - coordinates_array.mean(axis=0)
    src_coord = np.pad(src_coord, ((1, 1), (0, 0)), "constant", constant_values=0)

    src_distance = pairwise_distance_numpy(src_coord)

    # Calculate edge type, what does this do?
    # src_edge_type = np.outer(src_tokens, src_tokens).astype(int)
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(
        1, -1
    )

    return {
        "src_tokens": src_tokens.astype(int),
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type.astype(int),
    }


def coordinates_list_to_unimol(
    atoms_list: List[List[str]],
    coordinates_list: List[List[List[float]]],
    dictionary: Dictionary,
) -> List[dict]:
    assert (
        len(dictionary) == 31
    ), "Dictionary length was 31 in unimol_tools and should be here too"

    unimol_input = []
    for atoms, coordinates in zip(atoms_list, coordinates_list):
        unimol_input.append(
            coordinates_to_unimol(
                atoms,
                coordinates,
                dictionary,
                max_atoms=256,
                remove_hs=True,
            )
        )
    return unimol_input


def smiles_to_unimol(smiles_list: List[str], dictionary: Dictionary) -> List[dict]:
    valid_smiles = validate_smiles(smiles_list)
    atoms_list, coordinates_list = generate_conformations(valid_smiles)
    unimol_input = coordinates_list_to_unimol(atoms_list, coordinates_list, dictionary)  # type: ignore
    return unimol_input
