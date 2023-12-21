from contextlib import nullcontext
from tempfile import TemporaryDirectory, mkdtemp
from typing import List, Optional, Union
from soleil.loader.loader import load_solconf
from soleil.overrides.overrides import merge_overrides
from torch_train_manager_2.defs import Unassigned
from torch_train_manager_2.extensible import Extensible
from .checkpoints import CheckpointSaver as _CheckpointSaver
from pathlib import Path
import torch
from soleil.package import package_from_serializable, package_as_serializable
from soleil.solconf import root, package_overrides as get_package_overrides
from soleil.solconf import rcall


class SoleilCheckpointSaver(_CheckpointSaver):
    """
    Checkpoint saver for soleil-configured train managers.
    This class is assumed to be instantiated inside a `*.solconf` file in order to infer values for  *root_config* and *package_overrides*.
    Otherwise, you need to supply values for these parameters.

    Train managers can be loaded from saved checkpoints using :func:`init_from_checkpoint`.

    .. warning:: In order for :func:`init_from_checkpoint` to work, it is assumed that the solconf package where this class is instantiated
        resolves to a :class:`TrainManager` instance -- i.e., its ``as_type`` member instantiates a :class:`TrainManager` object.

    """

    def __init__(
        self,
        *args,
        root_config: Optional[Union[Path, str]] = None,
        package_overrides: Optional[List[str]] = None,
        **kwargs
    ):
        self.root_config = Path(root_config or root())
        self.package_overrides = package_overrides or get_package_overrides()
        self.package = package_as_serializable(self.root_config.parent)

        super().__init__(*args, **kwargs)

    def post_train_step_epoch(self, train_manager, fixtures, epoch_num):
        torch.save(
            {
                "state_dicts": {
                    attrib_name: getattr(train_manager, attrib_name).state_dict()
                    for attrib_name in self.get_attribs_to_save(train_manager)
                },
                "fixtures": {name: fixtures[name] for name in self.saved_fixtures},
                "soleil_package": self.package,
                "soleil_root_config": self.root_config.stem,
                "soleil_overrides": self.package_overrides,
            },
            self.checkpoint_file_path(epoch_num),
        )


def init_from_checkpoint(checkpoint_file, overrides=None, solconf_package_dir=None):
    """
    Instantiates the object (presumbaly a train manager) specified in the solconf description within a checkpoint file and modifies its checkpoint
    loader to load weights and fixtures from the specified *checkpoint_file*.

    .. note::
        Since any fixtures need to be loaded in the correct context (e.g., the train or eval stage), the weights and fixtures are not immediately loaded
        from the checkpoint. Rather the loaded object's :class:`~torch_train_manager_2.extensions.checkpoints.CheckpointLoader` is modified
        by setting its *ckpt_spec* attribute to *checkpoint_file*. This will allow any fixtures (along with the weights) to load at the correct stage.

        If needed, weights can be loaded explicitly on the returned train manager using
        :meth:`CheckpointLoader.load_weights_and_fixtures <torch_train_manager_2.extensions.checkpoints.CheckpointLoader.load_weights_and_fixtures>`, setting
        ``fixtures=None`` to skip loading fixtures.

    :param overrides: Any overrides to apply when building the train manager from its solconf description.
    :param solconf_package_dir: A directory where the checkpoint's solconf package can be stored. By default, a temporary directory that is discarded before the function exists.
    """

    with nullcontext(
        solconf_package_dir
    ) if solconf_package_dir else TemporaryDirectory() as target_dir:
        # Initialize the obj from the solconf package in the checkpoint
        target_dir = Path(target_dir)
        checkpoint_data = torch.load(str(checkpoint_file), mmap=True)

        new_overrides = merge_overrides(
            checkpoint_data["soleil_overrides"], overrides or []
        )

        package_from_serializable(checkpoint_data["soleil_package"], target_dir)
        obj = load_solconf(
            (target_dir / checkpoint_data["soleil_root_config"]).with_suffix(
                ".solconf"
            ),
            overrides=new_overrides,
        )

        # Set checkpoint spec to the specified checkpoint file
        if (
            isinstance(obj, Extensible)
            and "checkpoint_loader" in obj.extensions
            and hasattr(obj.extensions["checkpoint_loader"], "ckpt_spec")
        ):
            obj.extensions["checkpoint_loader"].ckpt_spec = checkpoint_file
        else:
            raise Exception(
                "Unable to set checkpoint loader's load target to the specified checkpoint file"
            )

        return obj
