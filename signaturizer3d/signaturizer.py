from typing import List

import numpy as np
import numpy.typing as npt

from signaturizer3d.input.sdf import gather_sdf_data
from signaturizer3d.model import FineTunedUniMol
from signaturizer3d.space import CCSpace


class Signaturizer:
    def __init__(self, space: CCSpace):
        if not isinstance(space, CCSpace):
            try:
                space = CCSpace[space]
            except KeyError:
                raise ValueError(
                    f"{space} is not a valid CCSpace. Valid spaces are A1 ... E5, see the CCSpace enum."
                )
        self.space = space

        # Hydrogens are always removed as the signaturizers models are fine
        # tuned from the pre-trained UniMol model trained without hydrogens
        self.model = FineTunedUniMol(space=self.space, remove_hs=True)

    def infer_from_coordinates(
        self, atoms: List[str], coordinates: List[List[float]]
    ) -> npt.NDArray[np.float32]:  # (n, 128) dim array for n signatures
        sig_output = self.model.get_sig4_coordinates(atoms, coordinates)
        return sig_output

    def infer_from_smiles(self, smiles_list: List[str]) -> npt.NDArray[np.float32]:
        sig_output = self.model.get_sig4_smiles(smiles_list)
        return sig_output

    def infer_from_sdf(self, path: str) -> npt.NDArray[np.float32]:
        """
        Perform inference on molecules in an SDF file or all SDF files within a directory.

        Parameters:
        path (str): Path to an SDF file or a directory containing SDF files.
        """
        atoms_list, coords_list = gather_sdf_data(path)
        return self.infer_from_coordinates(atoms_list, coords_list)
