from .default import Extension


class Avg(Extension):
    """
    Computes the average of a scalar fixture across an entire evaluation epoch

    Optionally, plots a visualization of the average at the end of the evaluation epoch.

    This extension can be used by adding an instance of it to a train manager after an extension that generates the fixture
    to average. Another possibility is to derive it, e.g., :


    .. code-block::

        class ThresholdedLoss(Avg):
            def __init__(self, *args, **kwargs):
                super().__init__("thresh_loss", *args, **kwargs)

            def post_eval_step_batch(self, train_manager, fixtures, batch, prediction):

                # Add the fixture to average.
                hard_prediction = prediction>0.5
                fixtures["thresh_loss"] = train_manager.loss(targets, hard_prediction)

                # Call the super's hook to carry out the averaging
                fixtures(super().post_eval_step_batch)


    Exposes the following fixtures:

    * ``self.accum``, which by default is ``'accum_' + self.target``

    Depends on :class:`EvalState`

    """

    visualize: bool

    def __init__(
        self,
        target,
        accum=None,
        visualize: bool = True,
        fig: str = "avgs/{datasource_name}/{target}",
        counter="num_batches",
    ):
        """
        :param target: The name of the fixture to accumulate
        :param accum: The name of the fixture to create as an accumulator
        :param visualize: Whether to create a scalar graph
        :param fig: The name of the figure visualization will be placed -- can include formatting placeholders for any fixture and ``target`` and ``accum``.
        :param counter': The counter that will be used to divide the running sum
        """
        self.visualize = visualize
        self.target = target
        self.accum = accum or "accum_" + target
        self.counter = counter
        self.fig = fig

    def pre_eval_step_epoch(self, fixtures):
        if self.accum in fixtures:
            raise ValueError(f"Fixture `{self.accum}` already exists")
        fixtures.update({self.accum: 0.0})

    def post_eval_step_batch(self, fixtures):
        fixtures.modify(
            self.accum,
            fixtures[self.accum] + fixtures[self.target].detach().item(),
        )

    def post_eval_step_epoch(self, writer, epoch_num, fixtures):
        if self.visualize:
            writer.add_scalar(
                self.fig.format(
                    **{**fixtures, "target": self.target, "accum": self.accum}
                ),
                fixtures[self.accum] / fixtures[self.counter],
                epoch_num,
                smoothing=False,
            )
