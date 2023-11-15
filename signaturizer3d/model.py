from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from signaturizer3d.inference import run_inference
from signaturizer3d.input import coordinates_list_to_unimol, smiles_to_unimol
from signaturizer3d.space import CCSpace
from signaturizer3d.unimol import UniMolModel


class MolDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class FineTunedUniMol(object):
    def __init__(
        self,
        space: CCSpace,
        remove_hs: bool = True,
        use_gpu: bool = True,
        use_local_weights: bool = False,
    ):
        self.space = space
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        )

        model_file_name, model_file_url = None, None
        if use_local_weights:
            model_file_name = f"{self.space.name}_split0.pt"
        else:
            model_file_url = (
                "https://github.com/aksell/test-pytorch-modelhub/releases/download/full-CC/"
                + f"{self.space.name}_split0.pt"
            )

        self.model = UniMolModel(
            model_file_name=model_file_name,
            model_file_URL=model_file_url,
            classification_head_name=self.space.name,
            output_dim=128,
            remove_hs=remove_hs,
        ).to(self.device)

        self.model.eval()
        self.params = {"remove_hs": remove_hs}

    def get_sig4_coordinates(
        self, atoms: List[List[str]], coordinates: List[List[List[float]]]
    ):
        unimol_input = coordinates_list_to_unimol(
            atoms, coordinates, self.model.dictionary
        )
        dataset = MolDataset(unimol_input)
        sig4_output = run_inference(
            self.model, space=self.space, dataset=dataset, device=self.device
        )
        return sig4_output

    def get_sig4_smiles(self, smiles_list: Union[List[str], str]):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        unimol_input = smiles_to_unimol(smiles_list, self.model.dictionary)
        dataset = MolDataset(unimol_input)
        sig4_output = run_inference(
            self.model, space=self.space, dataset=dataset, device=self.device
        )
        return sig4_output
