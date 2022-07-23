"""Test Frequency functions"""

import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase

from dasf_attributes.attributes import util, Frequency


class TestBandpassFilter(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = freq.bandpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

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

        out_data = freq.bandpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.bandpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.bandpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq_cp = Frequency(use_cuda=True)
        freq_np = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = freq_cp.bandpass_filter(in_data_cp, freq_lp, freq_hp)
        out_data_np = freq_np.bandpass_filter(in_data_np, freq_lp, freq_hp)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp, out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestHighpassFilter(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = freq.highpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

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

        out_data = freq.highpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.highpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.highpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq_cp = Frequency(use_cuda=True)
        freq_np = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = freq_cp.highpass_filter(in_data_cp, freq_lp, freq_hp)
        out_data_np = freq_np.highpass_filter(in_data_np, freq_lp, freq_hp)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp, out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestLowpassFilter(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = freq.lowpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=util.is_cupy_enabled())

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

        out_data = freq.lowpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.lowpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq = Frequency(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.lowpass_filter(in_data, freq_lp, freq_hp)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        freq_lp = 0.3
        freq_hp = 0.7
        freq_cp = Frequency(use_cuda=True)
        freq_np = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = freq_cp.lowpass_filter(in_data_cp, freq_lp, freq_hp)
        out_data_np = freq_np.lowpass_filter(in_data_np, freq_lp, freq_hp)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp, out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestCWTOrmsby(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        freqs = (20, 40, 60, 80)
        freq = Frequency(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = freq.cwt_ormsby(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        freqs = (20, 40, 60, 80)
        freq = Frequency(use_cuda=util.is_cupy_enabled())

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

        out_data = freq.cwt_ormsby(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        freqs = (20, 40, 60, 80)
        freq = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.cwt_ormsby(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        freqs = (20, 40, 60, 80)
        freq = Frequency(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.cwt_ormsby(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        freqs = (20, 40, 60, 80)
        freq_cp = Frequency(use_cuda=True)
        freq_np = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = freq_cp.cwt_ormsby(in_data_cp, freqs)
        out_data_np = freq_np.cwt_ormsby(in_data_np, freqs)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp, out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestCWTRicker(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        freqs = 70
        freq = Frequency(use_cuda=util.is_cupy_enabled())

        in_shape = (100, 100, 100)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = freq.cwt_ricker(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        freqs = 70
        freq = Frequency(use_cuda=util.is_cupy_enabled())

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

        out_data = freq.cwt_ricker(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        freqs = 70
        freq = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.cwt_ricker(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(out_data.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        freqs = 70
        freq = Frequency(use_cuda=True)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = freq.cwt_ricker(in_data, freqs)

        out = out_data.compute()

        self.assertEqual(out.dtype, out_data.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        freqs = 70
        freq_cp = Frequency(use_cuda=True)
        freq_np = Frequency(use_cuda=False)

        in_shape = (100, 100, 100)
        in_shape_chunks = (50, 50, 50)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = freq_cp.cwt_ricker(in_data_cp, freqs)
        out_data_np = freq_np.cwt_ricker(in_data_np, freqs)

        try:
            out_cp = out_data_cp.compute()
            out_np = out_data_np.compute()
            np.testing.assert_array_almost_equal(out_cp, out_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos = np.where(out_cp != out_np)
            total = len(out_cp.flatten())
            diff = len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))
