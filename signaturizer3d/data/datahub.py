# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .conformer import ConformerGen
from .datareader import MolDataReader
from .datascaler import TargetScaler


class DataHub(object):
    def __init__(self, data=None, is_train=True, save_path=None, **params):
        self.data = data
        self.is_train = is_train
        self.save_path = save_path
        self.task = params.get("task", None)
        self.target_cols = params.get("target_cols", None)
        self.multiclass_cnt = params.get("multiclass_cnt", None)
        self.ss_method = params.get("target_normalize", "none")
        self._init_data(**params)

    def _init_data(self, **params):
        self.data = MolDataReader().read_data(self.data, self.is_train, **params)
        self.data["target_scaler"] = TargetScaler(
            self.ss_method, self.task, self.save_path
        )
        if self.task == "inference":
            self.data["target"] = None  # No targets for inference
        else:
            raise ValueError("Unknown task: {}".format(self.task))

        if "atoms" in self.data and "coordinates" in self.data:
            no_h_list = ConformerGen(**params).transform_coords(
                self.data["atoms"], self.data["coordinates"]
            )
        else:
            smiles_list = self.data["smiles"]
            no_h_list = ConformerGen(**params).transform_smiles(smiles_list)

        self.data["unimol_input"] = no_h_list
