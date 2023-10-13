from collections import OrderedDict as OrderedDict_, UserDict
from contextlib import contextmanager
from inspect import getfullargspec, signature
from typing import Any, Dict, Iterable, List, Optional, OrderedDict as OrderedDict, Set
from dataclasses import dataclass, field
import warnings


class Extension:
    pass


class FixturesDict(UserDict):
    stage_fixtures: List[Set[str]]
    """ Contains the names of all the fixtures for nested stages """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.stage_fixtures = []

    def __setitem__(self, key, value, force=False):
        if not self.stage_fixtures:
            raise Exception("No stage has been started.")
        if key not in self.data:
            # This avoids adding the fixture to the current stage if the stage updates an existing variable
            # e.g., tm.fixtures['count']+=1
            self.stage_fixtures[-1].add(key)
        elif not force:
            raise Exception(
                f"Attempting to create fixture {key} that already exists. Use method `modify` to modify its value."
            )
        super().__setitem__(key, value)

    def modify(self, key, value):
        if key not in self.data:
            raise Exception(f"Cannot modify non-existing fixture `{key}`.")
        self.__setitem__(key, value, force=True)

    def start_stage(self):
        self.stage_fixtures.append(set())

    def end_stage(self):
        for fixture_name in self.stage_fixtures.pop():
            self.pop(fixture_name, None)

    def __call__(self, method, **kwargs):
        arg_names = signature(method).parameters
        try:
            params = {key: kwargs.get(key, self[key]) for key in arg_names}
        except KeyError as err:
            raise TypeError(
                f'Failed to supply fixture `{", ".join(err.args)}` when attempting to call `{method}`'
            )

        return method(**params)


class Extensible:
    extensions: OrderedDict
    fixtures: FixturesDict
    """ Contains all fixtures that will be injected as parameters to extension methods """

    def __init__(self, extensions=None):
        self.extensions = OrderedDict_(extensions or {})
        self.fixtures = FixturesDict()

    def add_extension(
        self, name: str, ext: Extension, at_start=False, as_default=False, warn=True
    ):
        """
        Add the specified extension, moving it to the beginning or end of the ordered dictionary containing all the extensions.
        """
        if as_default and name in self.extensions:
            return
        else:
            self.extensions[name] = ext
            self.extensions.move_to_end(name, last=not at_start)

    def get_extension_methods(self, prefix: str, stage_name: str):
        """
        Returns all the extension methods for the specified stage.

        .. rubric:: Examples:

        For  ``stage_name='my_stage'``, ``prefix='pre'``, will return all ``'pre_my_stage'`` extension methods that are not ``None``.

        :param stage_name: The name of the stage to get extension methods for.
        :prefix: Either ``'pre'`` or ``'post'``.
        """

        ext_method_name = f"{prefix}_{stage_name}"
        return [
            _meth
            for _ext in self.extensions.values()
            if (_meth := getattr(_ext, ext_method_name, None)) is not None
        ]

    @contextmanager
    def staged(
        self,
        stage_name,
        fixtures: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager containing the code for the center stage. All pre and post methods from all extensions will be called before and after the context.

        :param stage_name: The name of the stage. Extensions can implement methods named `f"pre_{stage_name}"` or `f"post_{stage_name}"` that will be called in the pre and post substages.
        :param fixtures: Constains fixtures that will be cleaned up at the end of the stage and that are available to the pre substage. Note that any fixtures provided here that
        were already handled by a parent stage will be set to the specified value but will not be cleaned up at the end of this stage.
        :param pre_defaults: Similar to ``fixtures`` but assigned only if the corresponding fixture does not yet exist. Those that get assigned will be cleared at the end of the stage.
        """

        # Start recording added fixtures
        self.fixtures.start_stage()
        self.fixtures.update(fixtures or {})
        [
            self.fixtures.setdefault(key, value)
            for key, value in (defaults or {}).items()
        ]

        # Call pre methods
        for _meth in self.get_extension_methods("pre", stage_name):
            self.fixtures(_meth)

        # Yield to center stage
        try:
            yield
        except Exception:
            raise
        else:
            # Call post methods
            for _meth in self.get_extension_methods("post", stage_name):
                self.fixtures(_meth)

            # Remove all stage fixtures
            self.fixtures.end_stage()

    def __getitem__(self, name):
        return self.extensions[name]
