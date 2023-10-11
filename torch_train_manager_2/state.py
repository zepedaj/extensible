from dataclasses import dataclass, field
from .defs import *
from .extensions import Extension


class EvalState(Extension):
    visualize: bool

    def __init__(self, visualize: bool = True):
        self.visualize = visualize

    def pre_eval_step_epoch(self, fixtures):
        fixtures.update({"num_batches": 0, "num_samples": 0, "cumm_loss": 0.0})

    def post_eval_step_batch(self, fixtures, true_batch_size, loss):
        fixtures.modify("num_batches", fixtures["num_batches"] + 1)
        fixtures.modify("num_samples", fixtures["num_samples"] + true_batch_size)
        fixtures.modify("cumm_loss", fixtures["cumm_loss"] + loss.detach().item())

    def post_eval_step_epoch(
        self, eval_datasource_name, writer, epoch_num, cumm_loss, num_samples
    ):
        if self.visualize:
            writer.add_scalar(
                f"losses/{eval_datasource_name}/loss",
                cumm_loss / num_samples,
                epoch_num,
                smoothing=False,
            )
