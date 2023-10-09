from torch_train_manager_2 import train_manager as mdl
import torch.nn


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

        tm = mdl.TrainManager(
            model=torch.nn.Linear(10, 10),
            loss=None,
            epochs=40,
            train_data=None,
            extensions={"ext": (ext := Ext())},
            writer=None,
        )

        #
        call_type, base_name, ext_methods = tm.get_extension_methods(
            "model_forward", "pre"
        )
        assert call_type is None
        assert base_name == "model_forward"
        assert set(ext_methods) == {ext.pre_model_forward}

        #
        call_type, base_name, ext_methods = tm.get_extension_methods(
            "eval_model_forward", "pre"
        )
        assert call_type == "eval"
        assert base_name == "model_forward"
        assert set(ext_methods) == {ext.pre_model_forward, ext.pre_eval_model_forward}
