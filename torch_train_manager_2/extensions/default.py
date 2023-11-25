from pathlib import Path
from typing import Iterable, Optional, Tuple
from ..defs import *
import torch


class Extension:
    pass


class EvalState(Extension):
    """
    Exposes the following fixtures:

    * **num_batches**: The cumulative number of batches currently visited during an evaluation epoch.
    * **num_samples**: The cumulative number of samples currently visited during an evaluation epoch.
    * **cum_loss**: The sum of all losses for all batches currently visited during an evalution epohc.

    Adds the following visualization at the end of the epoch:

    * Cumulative loss divided by the total number of batches in the epoch.
    """

    visualize: bool

    def __init__(self, visualize: bool = True):
        self.visualize = visualize

    def pre_eval_step_epoch(self, fixtures):
        fixtures.update({"num_batches": 0, "num_samples": 0, "cum_loss": 0.0})

    def post_eval_step_batch(self, fixtures, true_batch_size, loss):
        fixtures.modify("num_batches", fixtures["num_batches"] + 1)
        fixtures.modify("num_samples", fixtures["num_samples"] + true_batch_size)
        fixtures.modify("cum_loss", fixtures["cum_loss"] + loss.detach().item())

    def post_eval_step_epoch(
        self, datasource_name, writer, epoch_num, cum_loss, num_batches
    ):
        if self.visualize:
            writer.add_scalar(
                f"losses/{datasource_name}/loss",
                cum_loss / num_batches,
                epoch_num,
                smoothing=False,
            )


class CheckpointSaver(Extension):
    ckpt_ext = ".ckpt"

    def __init__(
        self,
        path,
        saved_fixtures: Optional[Iterable[str]] = ("epoch_num",),
        saved_attribs: Optional[Iterable[str]] = None,
        load_ckpt: Union[int, bool] = False,
    ):
        """
        :param path: The path where all values will be saved.
        :param saved_fixtures: The fixtures to save. By default, only ``'epoch_num'`` is saved.
        :param saved_attribs: If not specified, all model attributes with a ``state_dict`` attribute will be saved.
        :param load_ckpt: When carrying out standalone evaluations, an epoch number (or ``True`` for the latest checkpoint) specifying
        a checkpoint to load. When training, set to ``True`` to continue training if any checkpoints exist (training will fail with the default value ``False`` if checkpoints exist).
        """
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.saved_fixtures = saved_fixtures
        self._saved_attribs = saved_attribs
        self.load_ckpt = load_ckpt

    def get_saved_attribs(self, train_manager):
        # Deduce saved attribs if none provided
        return self._saved_attribs or [
            attr_name
            for attr_name in dir(train_manager)
            if hasattr(getattr(train_manager, attr_name), "state_dict")
        ]

    def filepath(self, epoch_num):
        return self.path / f"{epoch_num}{self.ckpt_ext}"

    def get_latest_checkpoint(self) -> Optional[Tuple[int, Path]]:
        if ckpt_files := list(self.path.glob(f"*{self.ckpt_ext}")):
            epoch_num, ckpt_file = max(
                [(int(_cf.stem), _cf) for _cf in ckpt_files], key=lambda x: x[0]
            )
            return epoch_num, ckpt_file
        return None

    def load_checkpoint(self, train_manager, fixtures, ckpt_num: int):
        #
        saved_values = torch.load(self.filepath(ckpt_num))

        # Load states
        [
            attrib.load_state_dict(attrib_value)
            for attrib_name, attrib_value in saved_values["state_dicts"].items()
            # Some stateful vars ('optimizer', 'lr_schedule') might be set to `None`
            # during stanadlone evaluations -- skip loading vars that don't have a state dict.
            if hasattr(attrib := getattr(train_manager, attrib_name), "load_state_dict")
        ]

        # Set fixtures
        [
            fixtures.modify(fixture_name, fixture_value)
            for fixture_name, fixture_value in saved_values["fixtures"].items()
        ]

    def pre_train(self, train_manager, fixtures, standalone_eval):
        """
        Loads the state at a particular checkpoint in preparation for standalone evaluations.
        """
        # Param load_ckpt must be a boolean when training
        if not isinstance(self.load_ckpt, bool):
            raise Exception(
                "Parameter ``load_ckpt`` must be a boolean when training  -- can only load the latest checkpoint when training."
            )

        # If no request to load checkpoint, check no training has previously happened
        if not self.load_ckpt and self.get_latest_checkpoint() is not None:
            raise Exception(
                f"Checkpoints exist in the specified path `{self.path}` -- you need to specify `load_ckpt=True` if you wish to continue training"
            )

        # Load saved attribs if requested and available
        if self.load_ckpt and (ckpt := self.get_latest_checkpoint()) is not None:
            self.load_checkpoint(train_manager, fixtures, ckpt[0])

    def pre_eval(self, train_manager, fixtures, standalone_eval):
        """
        Loads the state at a particular checkpoint in preparation for standalone evaluations.

        Fails if no checkpoints are available.
        """
        # Currently training, no need to load ckpt here
        if not standalone_eval:
            return

        # Standalone evaluation - we need to load a ckpt
        if self.load_ckpt is False:
            raise Exception(
                f"Parameter `load_ckpt` must be an epoch number (or `True` for the latest checkpoint) when executing standalone evaluations"
            )
        elif self.load_ckpt is True:
            if (ckpt := self.get_latest_checkpoint()) is None:
                raise ValueError(
                    "Did not find any checkpoints -- cannot carry out a standalone evaluation"
                )
            ckpt_num = ckpt[0]
        else:
            ckpt_num = self.load_ckpt

        # Load the specified checkpoint
        self.load_checkpoint(train_manager, fixtures, ckpt_num)

    def post_train_step_epoch(self, train_manager, fixtures, epoch_num):
        # Save a checkpoint file
        torch.save(
            {
                "state_dicts": {
                    attrib_name: getattr(train_manager, attrib_name).state_dict()
                    for attrib_name in self.get_saved_attribs(train_manager)
                },
                "fixtures": {name: fixtures[name] for name in self.saved_fixtures},
            },
            self.filepath(epoch_num),
        )
