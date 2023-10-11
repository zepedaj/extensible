from contextlib import contextmanager
from pathlib import Path
import ploteries
from torch_train_manager_2 import train_manager as mdl
import torch.nn
from tempfile import TemporaryDirectory


class TrainManager(mdl.TrainManager):
    def get_true_batch_size(self, batch) -> int:
        return len(batch)


def loss(batch, prediction):
    return prediction.sum()


class TestTrainManager:
    @contextmanager
    def get_train_manager(self):
        with TemporaryDirectory() as temp_dir:
            datasource = [torch.rand(10, 100) for _ in range(20)]
            tm = TrainManager(
                model=torch.nn.Linear(100, 30),
                loss=loss,
                epochs=40,
                train_data=datasource,
                eval_data=[("Eval", datasource)],
                device=torch.device("cpu:0"),
                writer=ploteries.Writer(Path(temp_dir) / "ploteries.pltr"),
            )
            yield tm

    def test_all(self):
        with self.get_train_manager() as tm:
            tm.train()

    def test_get_extension_methods(self):
        [
            "eval_model_forward",
            "eval_loss_forward",
            "train_model_forward",
            "train_loss_forward",
            "eval_step_batch",
        ]

        class Ext(mdl.Extension):
            def pre_model_forward(self):
                pass

            def pre_eval_model_forward(self):
                pass

            def post_model_forward(self):
                pass

            def post_eval_model_forward(self):
                pass

        tm = mdl.TrainManager(
            model=torch.nn.Linear(10, 10),
            loss=None,
            epochs=40,
            train_data=None,
            extensions={"ext": (ext := Ext())},
            writer=None,
        )

        #
        ext_methods = tm.get_extension_methods("pre", "model_forward")
        assert set(ext_methods) == {ext.pre_model_forward}

        #
        ext_methods = tm.get_extension_methods("pre", "eval_model_forward")
        assert set(ext_methods) == {ext.pre_eval_model_forward}
