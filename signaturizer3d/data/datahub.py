# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .conformer import ConformerGen
from .datareader import MolDataReader


class DataHub(object):
    def __init__(self, data=None, **params):
        self.data = MolDataReader().read_data(data, **params)

        if "atoms" in self.data and "coordinates" in self.data:
            no_h_list = ConformerGen(**params).transform_coords(
                self.data["atoms"], self.data["coordinates"]
            )
        else:
            smiles_list = self.data["smiles"]
            no_h_list = ConformerGen(**params).transform_smiles(smiles_list)

        self.data["unimol_input"] = no_h_list
