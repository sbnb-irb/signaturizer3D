from typing import List

from signaturizer3d.unimol import UniMolRepr


def infer_from_coordinates(atoms: List[str], coordinates: List[List[int]]):
    clf = UniMolRepr(data_type="finetuned", remove_hs=True)
    sig_output = clf.get_sig4_coordinates(atoms, coordinates)
    return sig_output
