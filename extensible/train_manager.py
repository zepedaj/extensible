from collections import OrderedDict as OrderedDict_
from typing import OrderedDict as OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from jztools.contextlib import reentrant_context_manager
import ploteries
import torch.optim
from torch import nn
from extensible.extensions.checkpoints import CheckpointLoader
from tqdm import tqdm

from extensible.extensions import CheckpointSaver, EvalState, Setup
from .defs import *
from .extensible import Extensible, Extension


@dataclass
class TrainManager(Extensible):
    model: torch.nn.Module
    """
    The model applied to each batch. Overload any of :meth:`model_forward`, :meth:`eval_model_forward`, :meth:`train_model_forward` if needed to call the model correctly.
    If the model implements method ``initialize_params``, that method will be called by :meth:`initialize_params`.
    """
    loss: Callable[[Batch, Prediction], ScalarTensor]
    """
    The loss computed from the batch output. Overload any of :meth:`model_forward`, :meth:`eval_model_forward`, :meth:`train_model_forward` if needed to call the loss correctly.
    """
    epochs: int
    """
    The number of training epochs
    """
    train_data: DataSource
    """
    The training data
    """
    eval_data: Optional[Dict[str, DataSource]] = None
    """
    Various validation data sets.
    """
    extensions: OrderedDict[str, Extension] = field(default_factory=OrderedDict_)
    """
    User-provided extensions. By default, if the keys ``'eval_state'`` and ``'ckpt_saver'`` are not included, :class:`~extensions.EvalState` and :class:`~extensions.CheckpointSaver` extensions are added at those keys.
    """
    train_dir: Path = Path(f"./{str(datetime.now())}")
    """
    Path that will contain the writer output and default checkpoints directory when training; checkpoints will be loaded from here when evaluating
    """
    writer: ploteries.Writer = Unassigned
    """
    A visualization writer. Defaults to a :class:`ploteries.Writer` object with path ``f"{train_dir}/ploteries.plts"``
    """
    device: Union[torch.device, str] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu:0"
    )
    """
    The main device used. The model will be moved to this device by :meth:`setup`.
    """
    optimizer: Union[
        torch.optim.Optimizer,
        Callable[
            [
                ParamsOrGroups,  # The params or param groups
            ],
            torch.optim.Optimizer,
        ],
    ] = torch.optim.Adam
    """
    The optimizer.
    """
    lr_schedule: Optional[LRSchedule] = None
    """
    An optional learning rate scheduler
    """
    mode_name: str = field(init=False)
    """
    The current mode. Can be one of ``'train'`` or ``'eval'``.
    """

    def __post_init__(self):
        #
        super().__init__(self.extensions)
        self._train_manager_stage = self.staged("train_manager")
        self._train_manager_stage.__enter__()  # This stage is never exited.
        self.fixtures.update({"train_manager": self, "fixtures": self.fixtures})

        # Copy the eval data dictionary
        self.eval_data = OrderedDict_(self.eval_data or {})

        # Make the output dir
        self.train_dir = Path(self.train_dir)
        self.train_dir.mkdir(exist_ok=True)

        # Set the writer
        if self.writer is Unassigned:
            self.writer = ploteries.Writer(self.train_dir / "ploteries.pltr")
        self.fixtures["writer"] = self.writer

        # Add checkpoint extensions. By dfeault, will load the latest checkpoint if available
        self.add_extension(
            "checkpoint_saver", CheckpointSaver(), at_start=False, as_default=True
        )
        self.add_extension(
            "checkpoint_loader", CheckpointLoader(), at_start=False, as_default=True
        )
        self.add_extension("eval_state", EvalState(), at_start=True, as_default=True)
        self.add_extension("setup", Setup(), at_start=True, as_default=True)

    @reentrant_context_manager
    def run_stage(self, *args, **kwargs):
        """
        The run stage is entered either at the start of :meth:`train` or, for standalone evaluations, at the start of :meth:`eval`
        and supports adding hooks to the top-level method being executed.
        """
        with self.staged("run", *args, **kwargs):
            yield

    def get_true_batch_size(self, batch: Batch) -> int:
        """
        .. note::
            When keeping track of the total number of samples, this method needs to be implemented.
            It should return the actual batch size, which might be different from the nominal size, particularly for the last batch in the epoch.
            By default, it will return ``len(batch)``, if batch supports it, otherwise ``1``.

        """
        try:
            return len(batch)
        except TypeError:
            return 1

    def initialize_params(self):
        """
        By default, this method attempts to call method ``self.model.initialize_params()``.
        If :attr:`self.model` does not implement that method, it carries out xavier uniform initialization on all parameters with more than one dimension.

        .. note:: Consider adding an ``initialize_params()`` method to your model or overloading this method in your train manager. Note that overloads of this method can take fixture arguments.
        """
        if hasattr(self.model, "initialize_params"):
            self.fixtures(self.model.initialize_params)
        else:
            for name, p in self.model.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    #### MODEL FORWARD

    def model_forward(self, batch):
        """
        .. note:: This method likely needs to be overloaded to extract the correct model input from the batch.

        .. note:: Unless an explicit implementation is provided in a derived class, this method takes the place of both :meth:`train_model_forward` and :meth:`eval_model_forward`.

        """
        return self.model(batch)

    def train_model_forward(self, *args, **kwargs):
        return self.model_forward(*args, **kwargs)

    def eval_model_forward(self, *args, **kwargs):
        return self.model_forward(*args, **kwargs)

    #### LOSS FORWARD

    def loss_forward(self, batch, prediction) -> ScalarTensor:
        """
        .. note:: This method likely needs to be overloaded to extract the correct model input from the batch and the model's prediction.

        .. note:: Unless an explicit implementation is provided in a derived class, this method takes the place of both :meth:`train_loss_forward` and :meth:`eval_loss_forward`.
        """
        return self.loss(batch, prediction)

    def train_loss_forward(self, batch, prediction):
        return self.loss_forward(batch, prediction)

    def eval_loss_forward(self, batch, prediction):
        return self.loss_forward(batch, prediction)

    #### TRAIN AND EVAL

    @contextmanager
    def mode(self, mode_name):
        if mode_name not in ["train", "eval"]:
            raise ValueError(f"Invalid mode name `{mode_name}`")

        current_training_mode = self.model.training
        try:
            self.model.train({"train": True, "eval": False}[mode_name])
            with {"train": nullcontext, "eval": torch.no_grad}[mode_name]():
                self.current_mode = mode_name
                yield

        finally:
            self.model.train(current_training_mode)

    def eval(self, eval_data: Optional[Dict[str, DataSource]] = None):
        """
        Runs evaluation over all the evaluation datasets.
        """

        if (eval_data or self.eval_data) is None:
            return

        #
        with self.run_stage({"epoch_num": 0, "standalone_eval": True}), self.staged(
            "eval"
        ), self.mode("eval"):
            #
            for datasource_name, datasource in (eval_data or self.eval_data).items():
                with self.staged(
                    "eval_step_epoch", {"datasource_name": datasource_name}
                ):
                    for batch in tqdm(datasource, desc=f"Eval({datasource_name})"):
                        with self.staged(
                            "eval_step_batch",
                            {
                                "batch": batch,
                                "true_batch_size": self.get_true_batch_size(batch),
                            },
                        ):
                            prediction = self.eval_model_forward(batch)
                            self.fixtures["prediction"] = prediction

                            loss = self.eval_loss_forward(batch, prediction)
                            self.fixtures["loss"] = loss

    def train(self):
        """
        Trains the model over multiple epochs, evaluating against all evaluation datasets before the first epoch and after every epoch.
        """
        with self.run_stage({"epoch_num": 0, "standalone_eval": False}), self.staged(
            "train"
        ), self.mode("train"):
            # Eval before all training
            if self.fixtures["epoch_num"] == 0:
                # Extension CheckpointSaver could have set `epoch_num`
                # to a value other than zero
                self.eval()

            # Train
            for _ in range(self.fixtures["epoch_num"], self.epochs):
                with self.staged("train_step_epoch"):
                    for batch in tqdm(
                        self.train_data,
                        desc=f"Train epoch {self.fixtures['epoch_num']+1}",
                    ):
                        with self.staged(
                            "train_step_batch",
                            {
                                "batch": batch,
                                "true_batch_size": self.get_true_batch_size(batch),
                            },
                        ):
                            prediction = self.train_model_forward(batch)
                            self.fixtures["prediction"] = prediction

                            loss = self.train_loss_forward(batch, prediction)
                            self.fixtures["loss"] = loss

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    self.fixtures.modify("epoch_num", self.fixtures["epoch_num"] + 1)

                # Eval after every epoch
                self.eval()
