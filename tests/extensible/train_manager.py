from extensible import train_manager as mdl
import torch.nn
from . import helpers

#


class TestTrainManager:
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

        with helpers.get_train_manager(
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
