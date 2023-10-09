from collections import OrderedDict as OrderedDict_
from contextlib import contextmanager
from typing import OrderedDict as OrderedDict
from dataclasses import dataclass, field


class Extension:
    train_manager: "Extensible"

    def register(self, train_manager: "Extensible"):
        self.train_manager = train_manager


class Extensible:
    extensions: OrderedDict[str, Extension]
    """ User-provided extensions. If not included, extensions with keys ``'eval_state'`` and ``'train_state'`` are added
    at the front, with instances of :class:`EvalState` and :class:`TrainState`, respectively. """

    def __init__(self, extensions=None):
        self.extensions = OrderedDict_(extensions or {})
        [_ext.register(self) for _ext in self.extensions.values()]

    def add_extension(
        self, name: str, ext: Extension, at_start=False, as_default=False
    ):
        """
        Add the specified extension, moving it to the beginning or end of the ordered dictionary containing the all extensions.
        """
        if as_default and name in self.extensions:
            return
        else:
            self.extensions[name] = ext
            self.extensions.move_to_end(name, last=not at_start)
            ext.register(self)

    def get_extension_methods(self, method_name: str, prefix: str):
        """
        Returns all the extensions methods for the specified method.

        .. rubric:: Examples:

        For  ``method_name='basemethod'``, ``prefix='pre'``, will return all ``'pre_basemethod'`` extension methods that are not ``None``.

        For  ``method_name='train_basemethod'``, ``prefix='pre'``, will return all ``'pre_train_basemethod'`` and ``'pre_basemethod'`` extension methods that are not ``None``.


        :param method_name: The name of the method to get extension methods for.
        :prefix: Either ``'pre'`` or ``'post'``.
        """
        call_type = (
            "train"
            if method_name.startswith("train_")
            else ("eval" if method_name.startswith("eval_") else None)
        )
        base_name = (
            method_name if call_type is None else method_name[len(call_type) + 1 :]
        )

        ext_method_names = [f"{prefix}_{base_name}"] + (
            [f"{prefix}_{call_type}_{base_name}"] if call_type is not None else []
        )
        ext_methods = [
            _meth
            for ext in self.extensions.values()
            for _ext_meth_name in ext_method_names
            if (_meth := getattr(ext, _ext_meth_name, None)) is not None
        ]
        return call_type, base_name, ext_methods

    def staged_call(self, method_name, *args, **kwargs):
        """
        Calls the pre calls for all extensions for the specified method, then, the method itself and then all the post calls.
        See :meth:`get_extension_methods` to see what extension methods get called for a given method name.
        """

        # Get center stage method
        _, base_name, _ = self.get_extension_methods(method_name, "pre")
        if (
            method := getattr(self, method_name, getattr(self, base_name, None))
        ) is None:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute `{method_name}` or `{base_name}`."
            )

        # Do staged call
        with self.staged(method_name, *args, **kwargs) as extra:
            out = method(*args, **kwargs)
            extra["out"] = out

        return out

    @contextmanager
    def staged(self, stage_name, *args, **kwargs):
        """
        Context manager containing the code for the center stage. All pre and post methods from all extensions will be called before and after the context.
        """

        # Get pre and post methods
        _, _, pre_methods = self.get_extension_methods(stage_name, "pre")
        _, _, post_methods = self.get_extension_methods(stage_name, "post")

        # Call pre methods
        for _meth in pre_methods:
            _meth(*args, **kwargs)

        # Yield to center stage
        extra_kwargs = {}
        yield extra_kwargs

        # Call post methods
        for _meth in post_methods:
            _meth(*args, **kwargs, **extra_kwargs)

    def __getitem__(self, name):
        return self.extensions[name]
