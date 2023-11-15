import logging
import os
from typing import List

from rdkit import Chem

logger = logging.getLogger(__name__)


def parse_sdf(sdf_file: str) -> List[tuple[List[str], List[List[float]]]]:
    """
    Parses an SDF file to extract atoms and their coordinates.

    Parameters:
    sdf_file (str): Path to the SDF file.

    Returns:
    List[Tuple[List[str], List[List[float]]]]: A List of tuples, where each tuple contains a list of atom types
    and a list of their corresponding 3D coordinates for each molecule in the SDF file.
    """
    try:
        molecules = Chem.SDMolSupplier(sdf_file)
    except OSError:
        logger.error(f"Unable to read file {sdf_file}")
        return []

    parsed_data = []

    for mol in molecules:
        if mol is None:  # Handling cases where a molecule can't be read
            logger.warning(f"Unable to read molecule from {sdf_file}")
            continue

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # Using GetConformer() with the assumption that the molecule has 3D coordinates set.
        coordinates = [
            list(mol.GetConformer().GetAtomPosition(idx))
            for idx in range(mol.GetNumAtoms())
        ]

        parsed_data.append((atoms, coordinates))

    return parsed_data


def gather_sdf_data(path: str) -> tuple[List[List[str]], List[List[List[float]]]]:
    """
    Gather atom and coordinate data from an SDF file or a directory of SDF files.

    Returns:
    Tuple containing two lists:
        - A list of lists of atom types for each molecule.
        - A list of lists of coordinates for each molecule.
    """
    all_atoms = []
    all_coords = []

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".sdf")]
    else:
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    for file in files:
        molecules = parse_sdf(file)
        for atoms, coords in molecules:
            all_atoms.append(atoms)
            all_coords.append(coords)

    return all_atoms, all_coords
