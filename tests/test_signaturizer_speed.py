import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from signaturizer3d import CCSpace
from signaturizer3d.data.unimol import coordinates_list_to_unimol
from signaturizer3d.unimol import FineTunedUniMol


@pytest.mark.performance
def test_data_processing_speed():
    data_dir = Path.cwd() / "tests" / "data"
    with open(data_dir / "atoms_list.pkl", "rb") as f:
        atoms_list = pickle.load(f)
    with open(data_dir / "coordinates_list.pkl", "rb") as f:
        coordinates_list = pickle.load(f)
    signaturizer = FineTunedUniMol(space=CCSpace.B4)

    start_time = time.time()
    unimol_input = coordinates_list_to_unimol(
        atoms_list, coordinates_list, signaturizer.model.dictionary
    )
    end_time = time.time()

    data_processing_time = end_time - start_time
    print(f"Data preprocessing time: {data_processing_time} seconds")
    assert unimol_input is not None
    assert data_processing_time < 0.01


@pytest.mark.performance
def test_inference_speed(signaturizer):
    data_dir = Path.cwd() / "tests" / "data"
    with open(data_dir / "atoms_list.pkl", "rb") as f:
        atoms_list = pickle.load(f)
    with open(data_dir / "coordinates_list.pkl", "rb") as f:
        coordinates_list = pickle.load(f)

    # Start timing
    start_time = time.time()

    result = signaturizer.infer_from_coordinates(atoms_list, coordinates_list)

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time} seconds")

    assert inference_time < 5
    assert result is not None
    assert result.shape == (32, 128)
