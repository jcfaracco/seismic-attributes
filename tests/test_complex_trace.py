"""Test ComplexAttributes functions"""

import logging

import numpy as np

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase

from . import TestSetupBase 

from dasf_attributes.attributes import util
from dasf_attributes.attributes import ComplexAttributes


class TestComplexTrace(TestSetupBase):
    @staticmethod
    def load_shape_test_cases():
        obj = ComplexAttributes()

        funcs = [func for func in dir(obj) if callable(getattr(obj, func)) and \
                                             not func.startswith("__") and \
                                             func != "create_array"]
        log = logging.getLogger("SomeTest.testSomething")
        log.error(funcs)
        return funcs

