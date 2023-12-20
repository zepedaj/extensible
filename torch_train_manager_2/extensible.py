from collections import OrderedDict as OrderedDict_, UserDict
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, OrderedDict as OrderedDict, Set
from uuid import uuid1
from .extensions import Extension


def randname():
    return str(uuid1())


class FixtureError(KeyError):
    pass


class PreHookContextManager:
    """
    All pre- hooks are executed as the :meth:`__enter__` method of this context manager within an :class:`ExitStack`.
    """

    def __init__(self, extensible: "Extensible", method: Callable):
        self.extensible = extensible
        self.method = method

    def __enter__(self):
        self.extensible.fixtures(self.method)

    def __exit__(self, *_):
        pass


def always(method):
    """
    By default, post methods do not run if an error occurred in the center stage unless they are
    decorated with this decorator. When decorated, they run whether an error occurred or not.

    Post methods decorated with :func:`always` should have default values for all their fixture parameters,
    as exceptions might result in missing fixtures.
    """
    # TODO: Make it possible to ignore missing fixtures when the hook method provides default parameter values.
    method.__extensible_run_always__ = True
    return method


class PostHookContextManager(AbstractContextManager):
    """
    All post- hooks are executed as the :meth:`__exit__` method of this context manager within an :class:`ExitStack`.
    The post- hook calls are wrapped in a stage that injects ``'exc_type'``, ``'exc_value'``, and ``'traceback'`` hooks
    that follow the same protocol as standard python ``__exit__`` methods. In particular, they will all be ``None`` if no
    exception occurred during the center stage and non-``None`` otherwise.
    """

    def __init__(self, extensible: "Extensible", method: Callable):
        self.extensible = extensible
        self.method = method

    def __exit__(self, exc_type, exc_value, traceback):
        if traceback is None or getattr(
            self.method, "__extensible_run_always__", False
        ):  # Run on clean center stage or when marked as on error
            #
            with self.extensible.staged(
                "__exit__" + randname()
            ):  # Not sure if random needed here, but does not hurt
                self.extensible.fixtures.update(
                    {
                        "exc_type": exc_type,
                        "exc_value": exc_value,
                        "traceback": traceback,
                    }
                )
                # Temporarily remove the exception stage to have the correct context
                # when calling the fixture
                with self.extensible.fixtures.temp_pop_stage():
                    return self.extensible.fixtures(self.method)


class FixturesDict(UserDict):
    stage_fixtures: List[Set[str]]
    """ Contains the names of all the fixtures for nested stages """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.stage_fixtures = []

    def __setitem__(self, key: str, value, force=False):
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

    def __getitem__(self, key: str):
        try:
            return super().__getitem__(key)
        except KeyError as err:
            raise FixtureError(*err.args)

    @contextmanager
    def temp_pop_stage(self):
        """
        Pops the current stage during the context to use the parent stage for fixture registration,
        and adds it back at the end of the context.
        """
        stage_fixtures = self.stage_fixtures.pop()
        try:
            yield
        finally:
            self.stage_fixtures.append(stage_fixtures)

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
        except FixtureError as err:
            raise TypeError(
                f'Failed to supply fixture `{", ".join(err.args)}` when attempting to call `{method}`'
            )

        return method(**params)


class Extensible:
    extensions: OrderedDict
    """ Contains all objects that expose hooks """
    fixtures: FixturesDict
    """ Contains all fixtures that will be injected as parameters to extension methods """

    def __init__(self, extensions=None):
        self.extensions = OrderedDict_(extensions or {})
        self.fixtures = FixturesDict()

    def add_extension(
        self, name: str, ext: Extension, at_start=False, as_default=False
    ):
        """
        Add the specified extension, moving it to the beginning or end of the ordered dictionary containing all the extensions.
        :param name: The key to use in the extensions dictionary.
        :param ext: The extension object.
        :param at_start: Whether to move to the start of the ordered dict of extensions.
        :param as_default: Whether to include only if an extension with the same name is not currently in the extensions.
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

        # Start recording added fixtures and add pre fixtures and defaults
        self.fixtures.start_stage()
        self.fixtures.update(fixtures or {})
        [
            self.fixtures.setdefault(key, value)
            for key, value in (defaults or {}).items()
        ]

        #
        try:
            with ExitStack() as stack:
                # Add post first to ensure they all execute on failures
                # since PostHookContextManager have no __enter__.
                [
                    stack.enter_context(PostHookContextManager(self, _meth))
                    # First added must be *inner-most* in nesting level
                    # so as to execute first
                    for _meth in list(self.get_extension_methods("post", stage_name))[
                        ::-1
                    ]
                ]
                # Execute all pre contexts
                [
                    stack.enter_context(PreHookContextManager(self, _meth))
                    # First added must be *outer-most* in nesting level
                    # so as to execute first
                    for _meth in self.get_extension_methods("pre", stage_name)
                ]
                yield
        finally:
            self.fixtures.end_stage()

    def __getitem__(self, name):
        return self.extensions[name]
