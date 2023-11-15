import logging
from multiprocessing import Pool
from typing import List

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_smiles(smiles_list: List[str]) -> List[str]:
    valid_smiles = []
    invalid_smiles = []

    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is not None:
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append(smiles)

    if invalid_smiles:
        logger.warning(f"Invalid SMILES excluded: {invalid_smiles}")

    return valid_smiles


def inner_smi2coords(smi: str, seed: int):
    """Calculate 3D coordinates for SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        if not atoms:
            raise ValueError(f"No atoms in molecule: {smi}")

        # Embed 3D coordinates
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass  # MMFF optimization failed
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        else:
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)

        return atoms, coordinates

    except Exception as e:
        logger.warning(f"Failed to generate conformers for {smi}, reason: {e}")
        atoms = []
        coordinates = np.zeros((0, 3))
        return atoms, coordinates


def smi2coords_wrapper(args):
    smi, seed = args
    return inner_smi2coords(smi, seed)


def generate_conformations(
    smiles_list: List[str], seed: int = -1
) -> tuple[List[List[str]], List[npt.NDArray[np.float32]]]:
    """
    Generate 3D coordinates for a list of SMILES strings

    Returns:
        atoms_list: list of atom types, [["C", "H", "H"], ["C"]]
        coordinates_list: list of numpy arrays containing the cordinates of a molecule with shape (n_atoms,3)
        [[x1, y1, z1], [x2, y2, z2], ...]
    """

    # Initialize multiprocessing pool and generate conformers
    with Pool() as pool:
        logger.info("Start generating conformers...")
        args_list = [(smi, seed) for smi in smiles_list]
        conformers = list(tqdm(pool.imap(smi2coords_wrapper, args_list)))

    # Log success and failure rates
    failed_conformers = [
        conformer for atoms, conformer in conformers if conformer.shape[0] == 0
    ]
    failed_rate = len(failed_conformers) / len(conformers) if len(conformers) else 0
    logger.info(f"Conformer generation success rate: {1 - failed_rate:.2%}")

    atoms_list = [atoms for atoms, _ in conformers]
    coordinates_list = [coords for _, coords in conformers]

    return atoms_list, coordinates_list
