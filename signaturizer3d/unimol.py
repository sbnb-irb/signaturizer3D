import numpy as np
import torch
from torch.utils.data import Dataset

from signaturizer3d.data import coordinates_list_to_unimol, smiles_to_unimol
from signaturizer3d.inference import run_inference
from signaturizer3d.models import UniMolModel


class MolDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class UniMolRepr(object):
    def __init__(
        self,
        data_type="molecule",
        remove_hs=False,
        use_gpu=True,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        self.model = UniMolModel(
            output_dim=128 if data_type == "finetuned" else 1,
            data_type=data_type,
            remove_hs=remove_hs,
        ).to(self.device)

        self.model.eval()
        self.params = {"data_type": data_type, "remove_hs": remove_hs}

    def get_sig4_coordinates(
        self, atoms: list[list[str]], coordinates: list[list[list[float]]]
    ):
        unimol_input = coordinates_list_to_unimol(atoms, coordinates)
        dataset = MolDataset(unimol_input)
        sig4_output = run_inference(self.model, dataset=dataset, device=self.device)
        return sig4_output

    def get_sig4_smiles(self, smiles_list: list[str] | str):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        unimol_input = smiles_to_unimol(smiles_list)
        dataset = MolDataset(unimol_input)
        sig4_output = run_inference(self.model, dataset=dataset, device=self.device)
        return sig4_output
