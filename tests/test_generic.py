"""Test Basic functions"""

import inspect
import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass
    
from test_conf import block_shape_list

from unittest import TestCase
from parameterized import parameterized, parameterized_class

from dasf_attributes import attributes
from dasf_attributes.attributes import util


class TestNewModules(TestCase):
    """
    Test new implemented modules without properly imports
    """
    def test_modules_match(self):
        defined_cls = [name for name, obj in inspect.getmembers(attributes) if inspect.isclass(obj)]

        all_cls = attributes.__all__

        # Sort them to match the same sequence
        defined_cls.sort()
        all_cls.sort()

        self.assertListEqual(defined_cls, all_cls)


def parameterize_all_methods_attributes():
    classes = ["attributes." + name for name, obj in inspect.getmembers(attributes) if inspect.isclass(obj)]
    
    functions = []
    for cls in classes:
        obj = eval(cls)()
        funcs = [func for func in dir(obj) if callable(getattr(obj, func)) and \
                                              not func.startswith("_") and \
                                              func != "create_array" and \
                                              func != "set_cuda"]
        for func in funcs:
            if func not in block_shape_list:
                functions.append({"obj": obj, "func": func})

    return functions


@parameterized_class(parameterize_all_methods_attributes())
class TestShapeAttributes(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check(self):
        in_shape = (20, 20, 20)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        func = getattr(self.obj, self.func)

        try:
            out_data = func(in_data).compute()
        except NotImplementedError:
            return

        self.assertEqual(in_shape, out_data.shape)


def parameterize_class_attributes():
    return [{"obj": eval("attributes." + name)()} for name, obj in inspect.getmembers(attributes) if inspect.isclass(obj)]


@parameterized_class(parameterize_class_attributes())
class TestInheritanceBaseClass(TestCase):
    def test_inheritance(self):
        self.assertEqual(len(type(self.obj).__bases__), 1)
        
        base = type(self.obj).__bases__[0]
        
        self.assertEqual(base.__name__, 'BaseAttributes')
