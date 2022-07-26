"""Test SignalProcess functions"""

import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase

from dasf_attributes.attributes import util, SignalProcess


class TestPhaseRotation(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        rotation = 90
        sp = SignalProcess(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = sp.phase_rotation(in_data, rotation)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        rotation = 90
        sp = SignalProcess(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.phase_rotation(in_data, rotation)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        rotation = 90
        sp = SignalProcess(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.phase_rotation(in_data, rotation)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

        self.assertEqual(np.ndarray, type(out))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        rotation = 90
        sp = SignalProcess(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.phase_rotation(in_data, rotation)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

        self.assertEqual(cp.ndarray, type(out))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        rotation = 90
        sp_cp = SignalProcess(use_cuda=True)
        sp_np = SignalProcess(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = sp_cp.phase_rotation(in_data_cp, rotation)
        out_data_np = sp_np.phase_rotation(in_data_np, rotation)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp.get(), out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestRescaleAmplitudeRange(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        min_val = 0.1
        max_val = 0.7
        sp = SignalProcess(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = sp.rescale_amplitude_range(in_data, min_val, max_val)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        min_val = 0.1
        max_val = 0.7
        sp = SignalProcess(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.rescale_amplitude_range(in_data, min_val, max_val)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        min_val = 0.1
        max_val = 0.7
        sp = SignalProcess(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.rescale_amplitude_range(in_data, min_val, max_val)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

        self.assertEqual(np.ndarray, type(out))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        min_val = 0.1
        max_val = 0.7
        sp = SignalProcess(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = sp.rescale_amplitude_range(in_data, min_val, max_val)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

        self.assertEqual(cp.ndarray, type(out))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        min_val = 0.1
        max_val = 0.7
        sp_cp = SignalProcess(use_cuda=True)
        sp_np = SignalProcess(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = sp_cp.rescale_amplitude_range(in_data_cp,
                                                    min_val, max_val)
        out_data_np = sp_np.rescale_amplitude_range(in_data_np,
                                                    min_val, max_val)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp.get(), out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))
