from typing import List

import numpy as np
import numpy.typing as npt

from signaturizer3d.unimol import UniMolRepr


class Signaturizer:
    def __init__(self):
        self.model = UniMolRepr(data_type="finetuned", remove_hs=True)

    def infer_from_coordinates(
        self, atoms: List[str], coordinates: List[List[int]]
    ) -> npt.NDArray[np.float64]:  # (n, 128) dim array for n signatures
        sig_output = self.model.get_sig4_coordinates(atoms, coordinates)
        return sig_output
