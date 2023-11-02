import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader


def run_inference(model, dataset, device, batch_size=32):
    model = model.to(device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=model.batch_collate_fn,
        shuffle=False,
        drop_last=False,
    )
    model = model.eval()
    raw_outputs_list = []  # A list to collect raw model outputs
    for batch in dataloader:
        input_data, _ = batch  # No target for inference
        input_data = {
            k: v.to(device) for k, v in input_data.items()
        }  # Move tensors to device
        with torch.no_grad():
            raw_outputs = model(**input_data, features_only=True)
            # normalize for A1 std and mean
            means = pickle.load(
                open(
                    "/aloy/home/alenes/playgrnd/Uni-Mol/unimol/notebooks/signaturizers/means.pkl",
                    "rb",
                )
            )
            stds = pickle.load(
                open(
                    "/aloy/home/alenes/playgrnd/Uni-Mol/unimol/notebooks/signaturizers/stds.pkl",
                    "rb",
                )
            )

            mean = torch.tensor(means["A1"], device=raw_outputs.device)
            std = torch.tensor(stds["A1"], device=raw_outputs.device)
            normalized_output = raw_outputs * std + mean

            raw_outputs_list.append(normalized_output.cpu().numpy())

    raw_outputs_array = np.concatenate(raw_outputs_list, axis=0)
    return raw_outputs_array
