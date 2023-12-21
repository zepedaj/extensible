from ..defs import *
from .extension import Extension


class Setup(Extension):

    """
    Basic train manager setup.
    """

    def pre_run(self, train_manager, standalone_eval):
        """
        Basic train manager setup. This hook

        #. moves the model to the device,
        #. builds the optimizer if it was provided as a callable, passing in all model parameters and

        """
        # TODO: Is there a way to avoid copying these uninitialized weights?
        train_manager.model = train_manager.model.to(train_manager.device)
        if not standalone_eval and not isinstance(
            train_manager.optimizer, torch.optim.Optimizer
        ):
            train_manager.optimizer = train_manager.optimizer(
                train_manager.model.parameters()
            )

    def post_load_checkpoint(self, checkpoint_loaded, fixtures, train_manager):
        """
        Initializes weights if no checkpoint was loaded.
        """
        if not checkpoint_loaded:
            fixtures(train_manager.initialize_params)


class EvalState(Extension):
    """
    Exposes the following fixtures:

    Fixtures
        num_batches
            The cumulative number of batches currently visited during an evaluation epoch.

        num_samples
            The cumulative number of samples currently visited during an evaluation epoch.

        cum_loss
            The sum of all losses for all batches currently visited during an evalution epoch.

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
