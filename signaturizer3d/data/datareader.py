# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..utils import logger


class MolDataReader(object):
    def read_data(self, data=None, is_train=True, **params):
        smiles_col = params.get("smiles_col", "SMILES")
        smi_strict = params.get("smi_strict", False)

        if isinstance(data, dict):
            # load from dict
            data = pd.DataFrame(data)
            if "target" in data:
                data = data.drop(columns=["target"])
        elif isinstance(data, list):
            # load from smiles list
            data = pd.DataFrame(data, columns=["SMILES"])
        else:
            raise ValueError("Unknown data type: {}".format(type(data)))

        dd = {
            "raw_data": data,
        }
        if smiles_col in data.columns:
            mask = data[smiles_col].apply(
                lambda smi: self.check_smiles(smi, is_train, smi_strict)
            )
            data = data[mask]
            dd["smiles"] = data[
                smiles_col
            ].tolist()  # unsure if we need this, my guess is it is for training and splittin purposes
            dd["scaffolds"] = data[smiles_col].map(self.smi2scaffold).tolist()
        else:
            dd["smiles"] = np.arange(data.shape[0]).tolist()
            dd["scaffolds"] = np.arange(data.shape[0]).tolist()

        if "atoms" in data.columns and "coordinates" in data.columns:
            dd["atoms"] = data["atoms"].tolist()
            dd["coordinates"] = data["coordinates"].tolist()

        return dd

    def check_smiles(self, smi, is_train, smi_strict):
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                logger.info(f"Illegal SMILES clean: {smi}")
                return False
            else:
                raise ValueError(f"SMILES rule is illegal: {smi}")
        return True

    def smi2scaffold(self, smi):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=True
            )
        except:
            return smi
