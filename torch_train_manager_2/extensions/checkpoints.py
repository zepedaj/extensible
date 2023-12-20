from pathlib import Path
from typing import Iterable, Optional, Tuple
from ..defs import *
from .extension import Extension
import torch


class CheckpointBase:
    ckpt_ext = ".ckpt"
    path: Path

    def filepath(self, epoch_num):
        return self.path / f"{epoch_num}{self.ckpt_ext}"


class CheckpointSaver(CheckpointBase, Extension):
    def __init__(
        self,
        path,
        saved_fixtures: Optional[Iterable[str]] = ("epoch_num",),
        saved_attribs: Optional[Iterable[str]] = None,
    ):
        """
        :param path: The path where all values will be saved.
        :param saved_fixtures: The fixtures to save. By default, only ``'epoch_num'`` is saved.
        :param saved_attribs: If not specified, all train manager attributes with a ``state_dict`` attribute will be saved. This usually includes the model, optimizer and learning rate scheduler.
        a checkpoint to load. When training, set to ``True`` to continue training if any checkpoints exist (training will fail with the default value ``False`` if checkpoints exist).
        """
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.saved_fixtures = saved_fixtures
        self._saved_attribs = saved_attribs

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
            self.filepath(epoch_num),
        )


class CheckpointLoader(CheckpointBase, Extension):
    def __init__(
        self,
        path,
        ckpt_num: int = -1,
        permissive=True,
    ):
        """
        :param path: The path where checkpoints were saved.
        :param ckpt_num: An epoch number (or ``-1`` for the latest checkpoint) specifying a checkpoint to load.
        :param permissive: If ``ckpt_num=-1`` and there are no checkpoints, do not raise an error.
        """
        self.path = Path(path)
        self.ckpt_num = ckpt_num
        self.permissive = permissive

    def get_latest_checkpoint(self) -> Optional[Tuple[int, Path]]:
        if ckpt_files := list(self.path.glob(f"*{self.ckpt_ext}")):
            epoch_num, ckpt_file = max(
                [(int(_cf.stem), _cf) for _cf in ckpt_files], key=lambda x: x[0]
            )
            return epoch_num, ckpt_file
        return None

    def pre_run(self, train_manager, fixtures):
        """
        Loads the weights and fixtures at a particular checkpoint in preparation for standalone evaluations or for finetuning.
        """

        # Get ckpt file
        if self.ckpt_num == -1:
            ckpt_file = (
                last_ckpt[1] if (last_ckpt := self.get_latest_checkpoint()) else None
            )
        else:
            ckpt_file = self.filepath(ckpt_num)

        if (ckpt_file is None and not self.permissive) or (
            ckpt_file is not None and not ckpt_file.is_file()
        ):
            raise ValueError(
                f"No checkpoint file for checkpoint number `{self.ckpt_num}`"
            )

        # Load saved attribs if requested and available
        if ckpt_file is not None:
            #
            saved_values = torch.load(ckpt_file)

            # Load states
            [
                attrib.load_state_dict(attrib_value)
                for attrib_name, attrib_value in saved_values["state_dicts"].items()
                # Some stateful vars ('optimizer', 'lr_schedule') might be set to `None`
                # during stanadlone evaluations -- skip loading vars that don't have a state dict.
                if hasattr(
                    attrib := getattr(train_manager, attrib_name), "load_state_dict"
                )
            ]

            # Set fixtures
            [
                fixtures.modify(fixture_name, fixture_value)
                for fixture_name, fixture_value in saved_values["fixtures"].items()
            ]
