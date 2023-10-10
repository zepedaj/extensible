from collections import OrderedDict as OrderedDict_
from typing import OrderedDict as OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union
import ploteries
import torch.optim
from torch import nn
from tqdm import tqdm

from torch_train_manager_2.state import EvalState, TrainState
from .defs import *
from .extensions import Extensible, Extension


@dataclass
class TrainManager(Extensible):
    model: torch.nn.Module
    loss: Callable[[Batch, Prediction], ScalarTensor]
    epochs: int
    train_data: DataSource
    eval_data: Optional[Dict[str, DataSource]] = None
    extensions: OrderedDict[str, Extension] = field(default_factory=OrderedDict_)
    """ User-provided extensions. If not included, extensions with keys ``'eval_state'`` and ``'train_state'`` are added
    at the front, with instances of :class:`EvalState` and :class:`TrainState`, respectively. The ``'eval_state'`` extension
    will reset when starting evaluation over each dataset. The ``'train_state'`` evaluation is not reset for the duration of training."""
    writer: ploteries.Writer = field(
        default_factory=lambda: ploteries.Writer(Path(f"./{str(datetime.now())}.pltr"))
    )
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

    def __post_init__(self):
        super().__init__(self.extensions)

        # Copy the eval data dictionary
        self.eval_data = OrderedDict_(self.eval_data or {})

        # Add the default extensions
        self.add_extension("eval_state", EvalState(), at_start=True, as_default=True)
        self.add_extension("train_state", TrainState(), at_start=True, as_default=True)

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

    def eval(self, eval_data: Optional[Dict[str, DataSource]] = None):
        if (eval_data or self.eval_data) is None:
            return

        #
        with self.mode("eval"), self.staged(
            "eval",
            fixtures={"eval_state": self["eval_state"]},
            defaults={  # In case this is a stand-alone eval.
                "train_manager": self,
                "writer": self.writer,
                "fixtures": self.fixtures,
            },
        ):
            #
            for datasource_name, datasource in (eval_data or self.eval_data).items():
                with self.staged(
                    "eval_step_epoch", {"eval_datasource_name": datasource_name}
                ):
                    for batch in tqdm(datasource, desc="Eval"):
                        with self.staged("eval_step_batch", {"batch": batch}):
                            prediction = self.eval_model_forward(batch)
                            self.fixtures["prediction"] = prediction

                            loss = self.eval_loss_forward(batch, prediction)
                            self.fixtures["loss"] = loss

    def train(self):
        with self.mode("train"), self.staged(
            "train",
            {
                "train_manager": self,
                "writer": self.writer,
                "fixtures": self.fixtures,
                "train_state": self["train_state"],
            },
        ):
            # Eval before all training
            self.eval()

            # Train
            for epoch_num in range(self.epochs):
                with self.staged("train_step_epoch"):
                    for batch in tqdm(self.train_data, desc=f"Train epoch {epoch_num}"):
                        with self.staged("train_step_batch", {"batch": batch}):
                            prediction = self.train_model_forward(batch)
                            self.fixtures["prediction"] = prediction

                            loss = self.train_loss_forward(batch, prediction)
                            self.fixtures["loss"] = loss

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                # Eval after every epoch
                self.eval()
