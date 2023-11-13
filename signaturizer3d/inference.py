import pathlib
import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from signaturizer3d.space import CCSpace
    from signaturizer3d.model import MolDataset, UniMolModel

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[0]


def normalize_output_for_space(
    raw_outputs: torch.Tensor, space: "CCSpace"
) -> torch.Tensor:
    means = pickle.load(
        open(
            PACKAGE_ROOT / "space_means.pkl",
            "rb",
        )
    )
    stds = pickle.load(
        open(
            PACKAGE_ROOT / "space_stds.pkl",
            "rb",
        )
    )

    space_mean = torch.tensor(means[space.value], device=raw_outputs.device)
    space_std = torch.tensor(stds[space.value], device=raw_outputs.device)
    normalized_output = raw_outputs * space_std + space_mean
    return normalized_output


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
            normalized_output = normalize_output_for_space(raw_outputs, space=space)
            raw_outputs_list.append(normalized_output.cpu().numpy())

    raw_outputs_array = np.concatenate(raw_outputs_list, axis=0)
    return raw_outputs_array
