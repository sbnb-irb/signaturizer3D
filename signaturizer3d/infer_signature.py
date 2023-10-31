import numpy as np
import numpy.typing as npt

from signaturizer3d.unimol import UniMolRepr


class Signaturizer:
    def __init__(self):
        self.model = UniMolRepr(data_type="finetuned", remove_hs=True)

    def infer_from_coordinates(
        self, atoms: list[str], coordinates: list[list[float]]
    ) -> npt.NDArray[np.float32]:  # (n, 128) dim array for n signatures
        sig_output = self.model.get_sig4_coordinates(atoms, coordinates)
        return sig_output

    def infer_from_smiles(self, smiles_list: list[str]) -> npt.NDArray[np.float32]:
        sig_output = self.model.get_sig4_smiles(smiles_list)
        return sig_output
