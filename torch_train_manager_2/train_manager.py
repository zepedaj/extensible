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
        self.eval_data = OrderedDict_(self.eval_data)

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
        .. warning::
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

    #### LOSS FORWARD

    def loss_forward(self, batch, prediction) -> ScalarTensor:
        """
        .. todo:: This method likely needs to be overloaded to extract the correct model input from the batch and the model's prediction.

        .. note:: Unless an explicit implementation is provided in a derived class, this method takes the place of both :meth:`train_loss_forward` and :meth:`eval_loss_forward`.
        """
        return self.loss(batch, prediction)

    #### STEP BATCH

    def eval_step_batch(self, batch):
        prediction = self.staged_call("eval_model_forward", batch)
        loss_value = self.staged_call("eval_loss_forward", batch, prediction)
        return prediction, loss_value

    def train_step_batch(self, batch):
        prediction = self.staged_call("train_model_forward", batch)
        loss_value = self.staged_call("train_loss_forward", batch, prediction)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()
        return prediction, loss_value

    #### STEP EPOCH

    def eval_step_epoch(self, datasource):
        for k_batch, batch in enumerate(tqdm(datasource)):
            self.staged_call("eval_step_batch", batch)

    def train_step_epoch(self, datasource):
        for k_batch, batch in enumerate(tqdm(datasource)):
            self.staged_call("train_step_batch", batch)

    #### STEP DATASOURCE

    def eval_step_datasource(self, datasource_name, datasource):
        return self.staged_call("eval_step_epoch", datasource)

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

    def eval(self):
        #
        if self.eval_data is None:
            return

        #
        with self.mode("eval"):
            for datasource_name, datasource in self.eval_data.items():
                self.staged_call("eval_step_datasource", datasource_name, datasource)

    def train(self):
        with self.mode("train"):
            self.eval()
            # Train
            for _ in tqdm(range(self.epochs)):
                self.staged_call("train_step_epoch", self.train_data)
                self.eval()
