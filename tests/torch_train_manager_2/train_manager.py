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


class EpochsGatherer(mdl.Extension):
    def __init__(self):
        self.trained_epochs = []

    def post_train_step_epoch(self, epoch_num):
        self.trained_epochs.append(epoch_num)


class TestTrainManager:
    @contextmanager
    def get_train_manager(self, **kwargs):
        with TemporaryDirectory() as temp_dir:
            datasource = [torch.rand(10, 100) for _ in range(20)]
            kwargs.setdefault("extensions", {})["epochs_gatherer"] = EpochsGatherer()
            tm = TrainManager(
                **{
                    **dict(
                        model=torch.nn.Linear(100, 30),
                        loss=loss,
                        epochs=40,
                        train_data=datasource,
                        eval_data=[("Eval", datasource)],
                        device=torch.device("cpu:0"),
                        output_dir=temp_dir,
                    ),
                    **kwargs,
                }
            )
            yield tm

    def test_all(self):
        with self.get_train_manager() as tm:
            tm.train()

            # Assert all checkpoints were saved
            orig_epochs = tm.epochs
            assert {x.stem for x in tm["ckpt_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 1)
            }
            assert tm["epochs_gatherer"].trained_epochs == list(range(1, tm.epochs + 1))

            # Continue training
            tm["ckpt_saver"].load_ckpt = True
            tm.epochs = orig_epochs + 1
            tm.train()
            assert tm["epochs_gatherer"].trained_epochs == list(
                range(1, orig_epochs + 2)
            )

            # Assert all checkpoints were saved
            assert {x.stem for x in tm["ckpt_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 2)
            }

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

        with self.get_train_manager(
            model=torch.nn.Linear(10, 10),
            loss=None,
            epochs=(epochs := 40),
            train_data=None,
            extensions={"ext": (ext := Ext())},
            writer=None,
        ) as tm:
            #
            ext_methods = tm.get_extension_methods("pre", "model_forward")
            assert set(ext_methods) == {ext.pre_model_forward}

            #
            ext_methods = tm.get_extension_methods("pre", "eval_model_forward")
            assert set(ext_methods) == {ext.pre_eval_model_forward}
