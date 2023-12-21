from pathlib import Path
from typing import Iterable, Optional, Tuple

from torch_train_manager_2.extensible import FixturesDict
from ..defs import *
from .extension import Extension
import torch


class CheckpointBase:
    ckpt_ext = ".ckpt"
    path: Path = None

    def __init__(self, path):
        if path:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)

    def checkpoint_file_path(self, epoch_num):
        """The path to a checkpoint file given the epoch number"""
        return self.path / f"{epoch_num}{self.ckpt_ext}"

    def pre_run(self, train_manager):
        """Assigns the default checkpoints directory"""
        if self.path is None:
            self.path = train_manager.train_dir / "checkpoints"
            self.path.mkdir(exist_ok=True)


class CheckpointSaver(CheckpointBase, Extension):
    """
    Saves checkpoints containing weights and fixtures at in the ``post_train_step_epoch`` stage.
    """

    def __init__(
        self,
        path=None,
        saved_fixtures: Optional[Iterable[str]] = ("epoch_num",),
        saved_attribs: Optional[Iterable[str]] = None,
    ):
        """
        :param path: The path where all values will be saved.
        :param saved_fixtures: The fixtures to save. By default, only ``'epoch_num'`` is saved.
        :param saved_attribs: If not specified, all train manager attributes with a ``state_dict`` attribute will be saved. This usually includes the model, optimizer and learning rate scheduler.
        a checkpoint to load. When training, set to ``True`` to continue training if any checkpoints exist (training will fail with the default value ``False`` if checkpoints exist).
        """
        self.saved_fixtures = saved_fixtures
        self._saved_attribs = saved_attribs
        super().__init__(path)

    def get_attribs_to_save(self, train_manager):
        # Deduce saved attribs if none provided
        return self._saved_attribs or [
            attr_name
            for attr_name in dir(train_manager)
            if hasattr(getattr(train_manager, attr_name), "state_dict")
        ]

    def post_train_step_epoch(self, train_manager, fixtures, epoch_num):
        """
        Save a checkpoint
        """
        torch.save(
            {
                "state_dicts": {
                    attrib_name: getattr(train_manager, attrib_name).state_dict()
                    for attrib_name in self.get_attribs_to_save(train_manager)
                },
                "fixtures": {name: fixtures[name] for name in self.saved_fixtures},
            },
            self.checkpoint_file_path(epoch_num),
        )


class CheckpointLoader(CheckpointBase, Extension):
    """
    Loads weights and fixtures at the *pre_run* stage of a training or evaluation run.

    By default, the last checkpoint is loaded, if any.
    """

    def __init__(
        self,
        path=None,
        ckpt_spec: Union[int, str, Path] = -1,
        permissive=True,
    ):
        """
        :param path: The path where checkpoints were saved.
        :param ckpt_spec: A checkpoint file path or epoch number (or ``-1`` for the latest checkpoint) specifying a checkpoint to load weights and fixtures from.
        :param permissive: If ``ckpt_spec=-1`` and there are no checkpoints, do not raise an error.
        """
        self.ckpt_spec = ckpt_spec
        self.permissive = permissive
        super().__init__(path)

    def get_latest_checkpoint(self) -> Optional[Tuple[int, Path]]:
        if ckpt_files := list(self.path.glob(f"*{self.ckpt_ext}")):
            epoch_num, ckpt_file = max(
                [(int(_cf.stem), _cf) for _cf in ckpt_files], key=lambda x: x[0]
            )
            return epoch_num, ckpt_file
        return None

    @classmethod
    def load_weights_and_fixtures(
        cls, ckpt_file, train_manager, fixtures: Optional[FixturesDict]
    ):
        """
        Loads the weights and sets the fixtures from the specified checkpoint file.
        """
        #
        saved_values = torch.load(str(ckpt_file), mmap=True)

        # Load states
        [
            attrib.load_state_dict(attrib_value)
            for attrib_name, attrib_value in saved_values["state_dicts"].items()
            # Some stateful vars ('optimizer', 'lr_schedule') might be set to `None`
            # during stanadlone evaluations -- skip loading vars that don't have a state dict.
            if hasattr(attrib := getattr(train_manager, attrib_name), "load_state_dict")
            and not isinstance(attrib, type)
        ]

        # Set fixtures
        if fixtures is not None:
            [
                fixtures.modify(fixture_name, fixture_value)
                for fixture_name, fixture_value in saved_values["fixtures"].items()
            ]

    def pre_run(self, train_manager, fixtures):
        """
        Loads the weights and fixtures at a particular checkpoint in preparation for standalone evaluations or for finetuning.

        Stages:

            load_checkpoint:

                fixtures:

                    load_checkpoint: ``bool``

        """

        with train_manager.staged("load_checkpoint"):
            super().pre_run(train_manager)

            # Get ckpt file
            if self.ckpt_spec == -1:
                ckpt_file = (
                    last_ckpt[1]
                    if (last_ckpt := self.get_latest_checkpoint())
                    else None
                )
            elif isinstance(self.ckpt_spec, int):
                ckpt_file = self.checkpoint_file_path(self.ckpt_spec)
            else:
                ckpt_file = Path(self.ckpt_spec)

            if (ckpt_file is None and not self.permissive) or (
                ckpt_file is not None and not ckpt_file.is_file()
            ):
                raise ValueError(
                    f"No checkpoint file corresponding to `{self.ckpt_spec}`"
                )

            # Load saved attribs if requested and available
            if ckpt_file is not None:
                self.load_weights_and_fixtures(ckpt_file, train_manager, fixtures)
                fixtures["checkpoint_loaded"] = True
            else:
                fixtures["checkpoint_loaded"] = False
