from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from extensible import TrainManager


def loss(batch, prediction):
    return prediction.sum()


@contextmanager
def get_train_manager(**kwargs):
    with TemporaryDirectory() as temp_dir:
        datasource = [torch.rand(10, 100) for _ in range(20)]
        tm = TrainManager(
            **{
                **dict(
                    model=torch.nn.Linear(100, 30),
                    loss=loss,
                    epochs=7,
                    train_data=datasource,
                    eval_data=[("Eval", datasource)],
                    device=torch.device("cpu:0"),
                    train_dir=temp_dir,
                ),
                **kwargs,
            }
        )
        yield tm
        if tm.writer:
            tm.writer.flush()
