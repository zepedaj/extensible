from extensible import extensible as mdl
import torch.nn
from . import helpers


class TestFixturesDict:
    def test_all(self):
        d = mdl.FixturesDict()
        assert d.stage_fixtures == []
        d.start_stage()
        d.update({"z": -1})
        d["a"] = 1
        assert d.stage_fixtures == [{"a", "z"}]

        d.update({"b": 2, "c": 3})
        assert d.stage_fixtures == [{"a", "b", "c", "z"}]

        assert d == {"a": 1, "b": 2, "c": 3, "z": -1}

        d.end_stage()
        assert d.stage_fixtures == []
