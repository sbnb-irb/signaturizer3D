import pathlib
import pickle
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from signaturizer3d.space import CCSpace

DIR = pathlib.Path(__file__).resolve().parents[0]


def normalize_for_space(raw_outputs: torch.Tensor, space: "CCSpace") -> torch.Tensor:
    means = pickle.load(
        open(
            DIR / "space_means.pkl",
            "rb",
        )
    )
    stds = pickle.load(
        open(
            DIR / "space_stds.pkl",
            "rb",
        )
    )

    space_mean = torch.tensor(means[space.name], device=raw_outputs.device)
    space_std = torch.tensor(stds[space.name], device=raw_outputs.device)
    normalized_output = raw_outputs * space_std + space_mean
    return normalized_output
