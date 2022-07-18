"""Test Basic functions"""

import logging

import numpy as np

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase
from parameterized import parameterized

from dasf_attributes.attributes import util


class TestSetupBase(TestCase):
    """
    Base class for testing attributes.
    """
    attribute = None

    @staticmethod
    def load_shape_test_cases():
        log = logging.getLogger( "SomeTest.testSomething" )
        log.error("Here")
        return []

    @staticmethod                                                   
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    @parameterized.expand(load_shape_test_cases(), skip_on_empty=True)
    def test_shape_check(self, obj, func_name):
        in_shape = (20, 20, 20)
        
        if util.is_cuda_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
            
        func = getattr(obj, func_name)
        
        out_data = func(in_data)
        
        self.assertEqual(in_shape, out_data.shape)
