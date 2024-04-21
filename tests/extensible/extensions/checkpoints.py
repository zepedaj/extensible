from copy import deepcopy
import numpy as np
import numpy.testing as npt
from extensible import extensions as mdl
from .. import helpers


class EpochsGatherer(mdl.Extension):
    def __init__(self):
        self.trained_epochs = []

    def post_train_step_epoch(self, epoch_num):
        self.trained_epochs.append(epoch_num)


class StateSaver(mdl.Extension):
    def _as_numpy_state_dict(self, sd):
        return {key: np.array(value.numpy()) for key, value in sd.items()}

    def pre_train(self, train_manager):
        self.start_state_dict = self._as_numpy_state_dict(
            train_manager.model.state_dict()
        )

    def post_train(self, train_manager):
        self.end_state_dict = self._as_numpy_state_dict(
            train_manager.model.state_dict()
        )


class TestCheckpointSaver:
    def test_all(self):
        with helpers.get_train_manager(
            extensions={
                "epochs_gatherer": EpochsGatherer(),
                "state_saver": StateSaver(),
            }
        ) as tm:
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
