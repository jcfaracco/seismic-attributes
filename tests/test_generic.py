"""Test Basic functions"""

import pytest
import inspect
import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass
    
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

    block_shape_list = [
        "gradient_dips",              # Implemented by test_dip_azm
        "gradient_structure_tensor",  # Implemented by test_dip_azm
        "gst_2D_dips",                # Implemented by test_dip_azm
        "volume_curvature",           # Implemented by test_edge_detection
        "trace_agc",
        "bandpass_filter",
        "cwt_ormsby",
        "cwt_ricker",
        "highpass_filter",
        "lowpass_filter",
        "phase_rotation",
        "rescale_amplitude_range"
        ]
    
    functions = []
    for cls in classes:
        obj_default = eval(cls)()
        obj_np = eval(cls)(use_cuda=False)
        obj_cp = eval(cls)(use_cuda=True)

        funcs = [func for func in dir(obj_default) if callable(getattr(obj_default, func)) and \
                                                      not func.startswith("_") and \
                                                      func != "create_array" and \
                                                      func != "set_cuda"]
        for func in funcs:
            if func not in block_shape_list:
                functions.append({"obj_default": obj_default, "obj_np": obj_np, "obj_cp": obj_cp, "func": func})

    return functions


@parameterized_class(parameterize_all_methods_attributes())
class TestShapeAttributes(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        in_shape = (20, 20, 20)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
            self.obj_default.set_cuda(True)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
            self.obj_default.set_cuda(False)

        func = getattr(self.obj_default, self.func)

        try:
            out_data = func(in_data).compute()
        except NotImplementedError:
            return

        self.assertEqual(in_shape, out_data.shape)

    def test_shape_check_from_dask_array(self):
        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
            self.obj_default.set_cuda(True)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
            self.obj_default.set_cuda(False)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        func = getattr(self.obj_default, self.func)

        try:
            out_data = func(in_data).compute()
        except NotImplementedError:
            return

        self.assertEqual(in_shape, out_data.shape)

    def test_dtype_as_np_from_dask_array(self):
        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        func = getattr(self.obj_np, self.func)

        try:
            out_data = func(in_data)
            out_data_comp = out_data.compute()
        except NotImplementedError:
            return

        self.assertEqual(out_data.dtype, out_data_comp.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        func = getattr(self.obj_cp, self.func)

        try:
            out_data = func(in_data)
            out_data_comp = out_data.compute()
        except NotImplementedError:
            return

        self.assertEqual(out_data.dtype, out_data_comp.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        func_cp = getattr(self.obj_cp, self.func)
        func_np = getattr(self.obj_np, self.func)

        try:
            out_data_cp = func_cp(in_data_cp)
            out_data_np = func_np(in_data_np)
        except NotImplementedError:
            return

        try:
            arr1 = out_data_cp.compute().get()
            arr2 = out_data_np.compute()
            np.testing.assert_array_almost_equal(arr1, arr2)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(arr1 != arr2)
            total = len(arr1.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


def parameterize_class_attributes():
    return [{"obj": eval("attributes." + name)()} for name, obj in inspect.getmembers(attributes) if inspect.isclass(obj)]


@parameterized_class(parameterize_class_attributes())
class TestInheritanceBaseClass(TestCase):
    def test_inheritance(self):
        self.assertEqual(len(type(self.obj).__bases__), 1)
        
        base = type(self.obj).__bases__[0]
        
        self.assertEqual(base.__name__, 'BaseAttributes')
