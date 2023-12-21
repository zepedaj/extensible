from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import numpy.testing as npt
from soleil import load_solconf
from soleil.package import package_from_serializable
from torch_train_manager_2.extensions import soleil as mdl
from torch_train_manager_2.extensions.extension import Extension
from .. import helpers
from .checkpoints import EpochsGatherer, StateSaver

solconf_package = {
    "main.solconf": """
from soleil.solconf import *
from torch_train_manager_2.extensions.soleil import SoleilCheckpointSaver
import torch.nn
from tests.torch_train_manager_2.extensions.checkpoints import EpochsGatherer, StateSaver
from tests.torch_train_manager_2.extensions.soleil import LossGatherer

torch.manual_seed(0)
datasource:hidden = [torch.rand(10, 100) for _ in range(20)]
type: as_type = "torch_train_manager_2:TrainManager"
model = torch.nn.Linear(100, 30)
loss = lambda batch, prediction: prediction.sum()
epochs = 7
train_data = datasource
eval_data = [("Eval", datasource)]
device = torch.device("cpu:0")
train_dir = req()
extensions = {
    'checkpoint_saver':SoleilCheckpointSaver(),
    'epochs_gatherer':EpochsGatherer(),
    'state_saver':StateSaver(),
    'loss_gatherer':LossGatherer()
    }
"""
}


class LossGatherer(Extension):
    def post_eval_step_epoch(self, cum_loss, num_batches):
        self.cum_loss = cum_loss
        self.num_batches = num_batches


class TestCheckpointSaver:
    def test_all(self):
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            package_from_serializable(solconf_package, temp_dir)
            tm = load_solconf(
                temp_dir / "main.solconf",
                overrides=[f'train_dir="{temp_dir}/train_dir"'],
            )
            #
            tm.train()

            # Assert all checkpoints were saved
            orig_epochs = tm.epochs
            assert {x.stem for x in tm["checkpoint_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 1)
            }
            assert tm["epochs_gatherer"].trained_epochs == list(range(1, tm.epochs + 1))

            previous_train_state = tm["state_saver"].end_state_dict

            # Continue training
            # tm["checkpoint_saver"].load_ckpt = True
            tm.epochs = orig_epochs + 1
            tm.train()
            assert tm["epochs_gatherer"].trained_epochs == list(
                range(1, orig_epochs + 2)
            )

            # Assert all checkpoints were saved
            assert {x.stem for x in tm["checkpoint_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 2)
            }
            npt.assert_equal(
                previous_train_state,
                tm["state_saver"].start_state_dict,
            )

            tm.writer.flush()

    def test_reload_eval(self):
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            package_from_serializable(solconf_package, temp_dir)
            tm = load_solconf(
                temp_dir / "main.solconf",
                overrides=[f'train_dir="{temp_dir}/train_dir"'],
            )
            #
            tm.train()

            # Assert all checkpoints were saved
            orig_epochs = tm.epochs
            assert {x.stem for x in tm["checkpoint_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 1)
            }
            assert tm["epochs_gatherer"].trained_epochs == list(range(1, tm.epochs + 1))

            previous_train_state = tm["state_saver"].end_state_dict

            # Load checkpoint, check weights equal last weights
            ldd_tm = mdl.init_from_checkpoint(
                tm.extensions["checkpoint_saver"].checkpoint_file_path(tm.epochs)
            )

            # Load weights by calling eval
            ldd_tm.eval()

            assert ldd_tm["loss_gatherer"].cum_loss == tm["loss_gatherer"].cum_loss
            assert (
                ldd_tm["loss_gatherer"].num_batches == tm["loss_gatherer"].num_batches
            )

            # Load params, check all match
            tm_params = {name: param for name, param in tm.model.named_parameters()}
            ldd_tm_params = {
                name: param for name, param in ldd_tm.model.named_parameters()
            }

            for key in set(tm_params.keys()).union(ldd_tm_params.keys()):
                npt.assert_array_equal(
                    tm_params[key].detach().cpu().numpy(),
                    ldd_tm_params[key].detach().cpu().numpy(),
                )

            # Continue training
            # tm["checkpoint_saver"].load_ckpt = True
            tm.epochs = orig_epochs + 1
            tm.train()
            assert tm["epochs_gatherer"].trained_epochs == list(
                range(1, orig_epochs + 2)
            )

            # Assert all checkpoints were saved
            assert {x.stem for x in tm["checkpoint_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 2)
            }
            npt.assert_equal(
                previous_train_state,
                tm["state_saver"].start_state_dict,
            )

            ldd_tm.writer.flush()
            tm.writer.flush()

    def test_reload_continue_training(self):
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            package_from_serializable(solconf_package, temp_dir)
            tm = load_solconf(
                temp_dir / "main.solconf",
                overrides=[f'train_dir="{temp_dir}/train_dir"'],
            )
            #
            tm.train()

            # Assert all checkpoints were saved
            orig_epochs = tm.epochs
            assert {x.stem for x in tm["checkpoint_saver"].path.glob("*")} == {
                str(k) for k in range(1, orig_epochs + 1)
            }
            assert tm["epochs_gatherer"].trained_epochs == list(range(1, tm.epochs + 1))

            previous_train_state = tm["state_saver"].end_state_dict

            # Load checkpoint, check weights equal last weights
            ldd_tm = mdl.init_from_checkpoint(
                tm.extensions["checkpoint_saver"].checkpoint_file_path(tm.epochs)
            )

            # Continue training
            # tm["checkpoint_saver"].load_ckpt = True
            tm.epochs = orig_epochs + 1
            tm.train()

            ldd_tm.epochs = orig_epochs + 1
            ldd_tm.train()

            # Check evals match
            assert ldd_tm["loss_gatherer"].cum_loss == tm["loss_gatherer"].cum_loss
            assert (
                ldd_tm["loss_gatherer"].num_batches == tm["loss_gatherer"].num_batches
            )

            # Load params, check all match
            tm_params = {name: param for name, param in tm.model.named_parameters()}
            ldd_tm_params = {
                name: param for name, param in ldd_tm.model.named_parameters()
            }

            for key in set(tm_params.keys()).union(ldd_tm_params.keys()):
                npt.assert_array_equal(
                    tm_params[key].detach().cpu().numpy(),
                    ldd_tm_params[key].detach().cpu().numpy(),
                )

            ldd_tm.writer.flush()
            tm.writer.flush()
