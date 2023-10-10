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
    visualize: bool

    def __init__(self, visualize: bool = True):
        self.visualize = visualize

    def pre_eval_step_epoch(self):
        self.reset()

    def post_eval_step_batch(self, train_manager, batch, loss):
        self.update(train_manager.get_true_batch_size(batch), loss)

    def post_eval_step_epoch(self, eval_datasource_name, train_manager):
        if self.visualize:
            train_manager.writer.add_scalar(
                f"losses/{eval_datasource_name}/loss",
                self.cumm_loss.detach().item() / self.sample_num,
                train_manager["train_state"].epoch_num,
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

    def pre_train_step_epoch(self):
        self.epoch_state.reset()

    def post_train_step_batch(self, train_manager, batch, loss):
        num_samples = train_manager.get_true_batch_size(batch)
        self.epoch_state.update(num_samples, loss)
        self.running_batch_num += 1
        self.running_sample_num += num_samples

    def post_train_step_epoch(self):
        self.epoch_num += 1
