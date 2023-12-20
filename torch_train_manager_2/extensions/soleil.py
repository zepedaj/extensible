from typing import List, Optional, Union
from torch_train_manager_2.defs import Unassigned
from .default import CheckpointSaver as _CheckpointSaver
from pathlib import Path
import torch
from soleil.package import package_from_serializable, package_as_serializable
from soleil.solconf import root, package_overrides as get_package_overrides


class SoleilCheckpointSaver(_CheckpointSaver):
    def __init__(
        self,
        *args,
        root_config: Union[Path, str] = Unassigned,
        package_overrides: List[str] = Unassigned,
        **kwargs
    ):
        """
        Checkpoint saver for soleil-configured train managers.
        This class is assumed to be instantiated inside a `*.solconf` file in order to infer values for  *package_root* and *package_overrides*.
        Otherwise, you need to supply values for these parameters.
        """
        self.root_config = root().parent if root_config is Unassigned else root_config
        self.package_overrides = (
            get_package_overrides()
            if package_overrides is Unassigned
            else package_overrides
        )

        super().__init__(*args, **kwargs)

    def post_train_step_epoch(self, train_manager, fixtures, epoch_num):
        # Save a checkpoint file
        torch.save(
            {
                "state_dicts": {
                    attrib_name: getattr(train_manager, attrib_name).state_dict()
                    for attrib_name in self.get_saved_attribs(train_manager)
                },
                "fixtures": {name: fixtures[name] for name in self.saved_fixtures},
                "soleil_package": package_as_serializable(self.package_root),
                "soleil_overrides": self.package_overrides,
            },
            self.filepath(epoch_num),
        )
