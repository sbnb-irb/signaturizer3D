# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import warnings
from functools import partial

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import (
    distance_matrix,
)  # TODO consider replacing, sizeable dep for a simple func
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")
import pathlib
from multiprocessing import Pool

from tqdm import tqdm

from signaturizer3d.unicore.dictionary import Dictionary

from ..utils import logger

DICT_PATH = pathlib.Path(__file__).resolve().parents[2] / "weights" / "dict.txt"


class ConformerGen(object):
    def __init__(self, **params):
        self.seed = params.get("seed", 42)
        self.max_atoms = params.get("max_atoms", 256)
        self.remove_hs = params.get("remove_hs", False)

        self.dictionary = Dictionary.load(DICT_PATH.as_posix())
        self.dictionary.add_symbol("[MASK]", is_special=True)

    def transform_coords(self, atoms_list, coordinates_list):
        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(
                coords2unimol(  # You do this per molecule, maybe slow for many?
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def single_process(self, smiles):
        atoms, coordinates = inner_smi2coords(
            smiles, seed=self.seed, remove_hs=self.remove_hs
        )
        return coords2unimol(atoms, coordinates, self.dictionary, self.max_atoms)

    def transform_smiles(self, smiles_list):
        pool = Pool()
        logger.info("Start generating conformers...")

        inputs = [item for item in tqdm(pool.imap(self.single_process, smiles_list))]

        pool.close()
        failed_cnt = np.mean([(item["src_coord"] == 0.0).all() for item in inputs])
        if failed_cnt > 0:
            logger.warning(
                f"Failed to generate conformers for {failed_cnt:.2%} of molecules."
            )
        else:
            logger.info("All conformers are generated successfully.")

        failed_3d_cnt = np.mean(
            [(item["src_coord"][:, 2] == 0.0).all() for item in inputs]
        )
        if failed_3d_cnt > 0:
            logger.warning(
                f"Failed to generate 3d conformers for {failed_3d_cnt:.2%} of molecules."
            )
        else:
            logger.info("All 3d conformers generated successfully.")

        return inputs


def inner_smi2coords(smi, seed, remove_hs=True):
    """
    Removes hydrogens if remove_hs
    """
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms) > 0, "No atoms in molecule: {}".format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        logger.warning("Failed to generate conformers, replace with zeros.")
        coordinates = np.zeros((len(atoms), 3))

    assert len(atoms) == len(
        coordinates
    ), "coordinates shape is not align with {}".format(smi)

    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != "H"]
        atoms_no_h = [atom for atom in atoms if atom != "H"]
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), f"Coordinate and atom count differ for {smi}"
        atoms, coordinates = atoms_no_h, coordinates_no_h

    return atoms, coordinates


def coords2unimol(
    atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params
):
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)

    # Remove hydrogens
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != "H"]
        atoms_no_h = [atom for atom in atoms if atom != "H"]
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(
            coordinates_no_h
        ), "Failure trying to remove hydrogens"
        atoms, coordinates = atoms_no_h, coordinates_no_h
    else:
        pass

    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]

    # Pad tokens(atoms)
    src_tokens = np.array(
        [dictionary.bos()]
        + [dictionary.index(atom) for atom in atoms]
        + [dictionary.eos()]
    )
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))

    # Normalize coordinates and add padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1, 3)), src_coord, np.zeros((1, 3))], axis=0)

    # Calculate distance matrix
    src_distance = distance_matrix(src_coord, src_coord)

    # Claculate edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(
        1, -1
    )
    return {
        "src_tokens": src_tokens.astype(int),
        "src_distance": src_distance.astype(np.float32),
        "src_coord": src_coord.astype(np.float32),
        "src_edge_type": src_edge_type.astype(int),
    }
