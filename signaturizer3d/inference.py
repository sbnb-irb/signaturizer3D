from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from signaturizer3d.normalize.normalize import normalize_for_space

if TYPE_CHECKING:
    from signaturizer3d.model import MolDataset, UniMolModel
    from signaturizer3d.space import CCSpace


def run_inference(
    model: "UniMolModel",
    space: "CCSpace",
    dataset: "MolDataset",
    device,
    batch_size: int = 32,
):
    model = model.to(device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=model.batch_collate_fn,
        shuffle=False,
        drop_last=False,
    )
    model = model.eval()
    raw_outputs_list = []
    for batch in dataloader:
        input_data, _ = batch
        input_data = {k: v.to(device) for k, v in input_data.items()}
        with torch.no_grad():
            raw_outputs = model(**input_data, features_only=True)
            normalized_output = normalize_for_space(raw_outputs, space=space)
            raw_outputs_list.append(normalized_output.cpu().numpy())

    raw_outputs_array = np.concatenate(raw_outputs_list, axis=0)
    return raw_outputs_array
