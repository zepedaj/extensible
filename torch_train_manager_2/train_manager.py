from collections import OrderedDict as OrderedDict_
from typing import OrderedDict as OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from pglib.contextlib import reentrant_context_manager
import ploteries
import torch.optim
from torch import nn
from tqdm import tqdm

from torch_train_manager_2.extensions import CheckpointSaver, EvalState
from .defs import *
from .extensible import Extensible, Extension


class Unassigned:
    pass


@dataclass
class TrainManager(Extensible):
    model: torch.nn.Module
    loss: Callable[[Batch, Prediction], ScalarTensor]
    epochs: int
    train_data: DataSource
    eval_data: Optional[Dict[str, DataSource]] = None
    extensions: OrderedDict[str, Extension] = field(default_factory=OrderedDict_)
    """
    User-provided extensions. If an extension with key ``'eval_state'`` is not included, it is added by default as an instance of :class:`EvalState`.
    """
    output_dir: Path = Path(f"./{str(datetime.now())}")
    writer: ploteries.Writer = Unassigned
    device: Union[torch.device, str] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu:0"
    )
    optimizer: Union[
        torch.optim.Optimizer,
        Callable[
            [
                ParamsOrGroups,  # The params or param groups
            ],
            torch.optim.Optimizer,
        ],
    ] = torch.optim.Adam
    lr_schedule: Optional[LRSchedule] = None
    mode_name: str = field(init=False)
    load_ckpt: InitVar[bool] = False

    def __post_init__(self, load_ckpt):
        super().__init__(self.extensions)

        # Copy the eval data dictionary
        self.eval_data = OrderedDict_(self.eval_data or {})

        # Make the output dir
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set the writer
        if self.writer is Unassigned:
            self.writer = ploteries.Writer(self.output_dir / "ploteries.pltr")

        # Add the default extensions
        self.add_extension("eval_state", EvalState(), at_start=True, as_default=True)
        self.add_extension(
            "ckpt_saver",
            CheckpointSaver(path=self.output_dir / "checkpoints", load_ckpt=load_ckpt),
            at_start=False,
            as_default=True,
        )

        # Setup
        self.setup()

    def setup(self):
        """
        This method 1) sets up a default writer if none was provided, 2) moves the model to the device and
        3) builds the optimizer if it was provided as a callable.
        """

        self.model = self.model.to(self.device)
        self.initialize_params()
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer = self.optimizer(self.model.parameters())

    def get_true_batch_size(self, batch: Batch) -> int:
        """
        .. note::
            When keeping track of the total number of samples, this method needs to be implemented.
            It should return the actual batch size, which might be different from the nominal size, particularly for the last batch in the epoch.
            By default, it will return ``1``, meaning that it counts batches instead of batch sizes.

        """
        return 1

    def initialize_params(self):
        """
        .. note:: Consider overloading this method.
        """
        # Initialize all parameters in the model
        for name, p in self.model.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #### MODEL FORWARD

    def model_forward(self, batch):
        """
        .. todo:: This method likely needs to be overloaded to extract the correct model input from the batch.

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

    def train_loss_forward(self, *args, **kwargs):
        return self.loss_forward(*args, **kwargs)

    def eval_loss_forward(self, *args, **kwargs):
        return self.loss_forward(*args, **kwargs)

    #### TRAIN AND EVAL

    @contextmanager
    def mode(self, mode_name):
        current_training_mode = self.model.training
        self.model.train({"train": True, "eval": False}[mode_name])
        try:
            with {"train": nullcontext, "eval": torch.no_grad}[mode_name]():
                self.current_mode = mode_name
                yield
        finally:
            self.model.train(current_training_mode)

    @reentrant_context_manager
    def train_manager_stage(self):
        """
        Wraps both :meth:`train` or stand-alond :meth:`eval calls.
        """
        with self.staged(
            "train_manager",
            {"train_manager": self, "writer": self.writer, "fixtures": self.fixtures},
        ):
            yield

    def eval(self, eval_data: Optional[Dict[str, DataSource]] = None):
        if (eval_data or self.eval_data) is None:
            return

        #
        with self.train_manager_stage(), self.mode("eval"), self.staged(
            "eval",
            defaults={"epoch_num": 0},  # In case this is a stand-alone eval.
        ):
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
        with self.train_manager_stage(), self.mode("train"), self.staged(
            "train", {"epoch_num": 0}
        ):
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
