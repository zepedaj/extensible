from dataclasses import dataclass, field
from .defs import *
from .extensions import Extension


class EpochState:
    """The current state within one epoch"""

    batch_num: int
    """ The batch number within this epoch"""
    sample_num: int
    """ The cummulative number of samples from all batches up the and including the current batch"""
    cumm_loss: ScalarTensor
    """The cummulative loss"""

    def reset(self):
        self.batch_num = 0
        self.sample_num = 0
        self.cumm_loss = 0

    def update(self, true_batch_size, loss_value):
        self.batch_num += 1
        self.sample_num += true_batch_size
        self.cumm_loss += loss_value.detach()


class EvalState(EpochState, Extension):
    datasource_name: str
    visualize: bool

    def __init__(self, visualize: bool = True):
        self.visualize = visualize

    def pre_eval_step_datasource(self, datasource_name, datasource):
        self.datasource_name = datasource_name
        self.reset()

    def post_eval_step_batch(self, batch, out):
        prediction, loss_value = out
        self.update(self.train_manager.get_true_batch_size(batch), loss_value)

    def post_eval_step_datasource(self, datasource_name, datasource, out):
        if datasource_name != self.datasource_name:
            raise ValueError(
                f"Data source name does not match (`{datasource_name}!={self.datasource_name}`)"
            )
        if self.visualize:
            self.train_manager.writer.add_scalar(
                f"losses/{self.datasource_name}/loss",
                self.cumm_loss.detach().item() / self.sample_num,
                self.train_manager["train_state"].epoch_num,
            )


@dataclass
class TrainState(Extension):
    epoch_num: int = 0
    """ The running number of epochs"""
    running_batch_num: int = 0
    """ The number of batches across all epochs"""
    running_sample_num: int = 0
    """ The number of samples across all epochs"""
    epoch_state: EpochState = field(default_factory=EpochState)
    """ The state of the current epoch """

    def pre_train_step_epoch(self, *_, **__):
        self.epoch_state.reset()

    def post_train_step_batch(self, batch, out):
        prediction, loss_value = out
        self.epoch_state.update(
            self.train_manager.get_true_batch_size(batch), loss_value
        )

    def post_train_step_epoch(self, *_, **__):
        self.epoch_num += 1
