import logging

from rdkit import Chem

logger = logging.getLogger(__name__)


def parse_sdf(sdf_file: str) -> list[tuple[list[str], list[list[float]]]]:
    """
    Parses an SDF file to extract atoms and their coordinates.

    Parameters:
    sdf_file (str): Path to the SDF file.

    Returns:
    List[Tuple[List[str], List[List[float]]]]: A list of tuples, where each tuple contains a list of atom types
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
