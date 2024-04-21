from collections import defaultdict
import torch
from extensible.extensions.default import Extension
from extensible.train_manager import TrainManager

TAB_NAME = "grads"


class GradNormVis(Extension):
    def __init__(
        self, norm_fxn=lambda grad: torch.norm(grad), name=f"{TAB_NAME}/grad_norm"
    ):
        self.name = name
        self.norm_fxn = norm_fxn

    def pre_train_step_epoch(self):
        self.cum_norms = defaultdict()
        self.steps = defaultdict(int)

    def post_train_step_batch(self, train_manager: TrainManager):
        if self.cum_norms.default_factory is None:
            self.cum_norms.default_factory = lambda: torch.zeros(
                1, device=train_manager.device
            )

        for name, param in train_manager.model.named_parameters():
            if param.grad is not None:
                self.steps[name] += 1
                self.cum_norms[name] += self.norm_fxn(param.grad.detach())

    def post_train_step_epoch(self, writer, epoch_num):
        contents = [
            (name, value.cpu().item() / steps)
            for name, value in self.cum_norms.items()
            if (steps := self.steps[name]) > 0
        ]
        writer.add_scalars(
            self.name,
            [val for _, val in contents],
            epoch_num,
            [{"name": name} for name, _ in contents],
            smoothing=False,
            layout_kwargs={"yaxis_type": "log"},
        )


class GradCoeffRMSVis(GradNormVis):
    def __init__(self, name=f"{TAB_NAME}/grad_coeff_rms"):
        super().__init__(
            norm_fxn=lambda grad: torch.mean(grad**2) ** 0.5,
            name=name,
        )
