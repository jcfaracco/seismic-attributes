"""Test DipAzm functions"""

import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase

from dasf_attributes.attributes import util, DipAzm


class TestGradientDips(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = dip_azm.gradient_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(in_shape, out0.shape)
        self.assertEqual(in_shape, out1.shape)

    def test_shape_check_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(in_shape, out0.shape)
        self.assertEqual(in_shape, out1.shape)

    def test_dtype_as_np_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(out_data[0].dtype, out0.dtype)
        self.assertEqual(out_data[1].dtype, out1.dtype)

        self.assertEqual(np.ndarray, type(out0))
        self.assertEqual(np.ndarray, type(out1))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=True)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(out_data[0].dtype, out0.dtype)
        self.assertEqual(out_data[1].dtype, out1.dtype)

        self.assertEqual(cp.ndarray, type(out0))
        self.assertEqual(cp.ndarray, type(out1))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        dip_azm_cp = DipAzm(use_cuda=True)
        dip_azm_np = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = dip_azm_cp.gradient_dips(in_data_cp)
        out_data_np = dip_azm_np.gradient_dips(in_data_np)

        try:
            out0_cp = out_data_cp[0].compute()
            out1_cp = out_data_cp[1].compute()
            out0_np = out_data_np[0].compute()
            out1_np = out_data_np[1].compute()
            np.testing.assert_array_almost_equal(out0_cp.get(), out0_np)
            np.testing.assert_array_almost_equal(out1_cp.get(), out1_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos0 = np.where(out0_cp != out0_np)
            unequal_pos1 = np.where(out1_cp != out1_np)
            total = len(out0_cp.flatten()) + len(out1_cp.flatten())
            diff = len(unequal_pos0[0]) + len(unequal_pos1[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestGST2DDips(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = dip_azm.gst_2D_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(in_shape, out0.shape)
        self.assertEqual(in_shape, out1.shape)

    def test_shape_check_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gst_2D_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(in_shape, out0.shape)
        self.assertEqual(in_shape, out1.shape)

    def test_dtype_as_np_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gst_2D_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(out_data[0].dtype, out0.dtype)
        self.assertEqual(out_data[1].dtype, out1.dtype)

        self.assertEqual(np.ndarray, type(out0))
        self.assertEqual(np.ndarray, type(out1))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=True)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gst_2D_dips(in_data)

        self.assertEqual(len(out_data), 2)

        out0 = out_data[0].compute()
        out1 = out_data[1].compute()

        self.assertEqual(out_data[0].dtype, out0.dtype)
        self.assertEqual(out_data[1].dtype, out1.dtype)

        self.assertEqual(cp.ndarray, type(out0))
        self.assertEqual(cp.ndarray, type(out1))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        dip_azm_cp = DipAzm(use_cuda=True)
        dip_azm_np = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = dip_azm_cp.gst_2D_dips(in_data_cp)
        out_data_np = dip_azm_np.gst_2D_dips(in_data_np)

        try:
            out0_cp = out_data_cp[0].compute()
            out1_cp = out_data_cp[1].compute()
            out0_np = out_data_np[0].compute()
            out1_np = out_data_np[1].compute()
            np.testing.assert_array_almost_equal(out0_cp.get(), out0_np)
            np.testing.assert_array_almost_equal(out1_cp.get(), out1_np)
        except AssertionError as ae:
            # Check if the percentage of mismatch is higher than 5.0 %
            unequal_pos0 = np.where(out0_cp != out0_np)
            unequal_pos1 = np.where(out1_cp != out1_np)
            total = len(out0_cp.flatten()) + len(out1_cp.flatten())
            diff = len(unequal_pos0[0]) + len(unequal_pos1[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))


class TestGradientStructureTensor(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)
        tensor_kernel = (2, 2, 2)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_data = dip_azm.gradient_structure_tensor(in_data,
                                                     kernel=tensor_kernel)

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_structure_tensor(in_data,
                                                     kernel=tensor_kernel)

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_structure_tensor(in_data,
                                                     kernel=tensor_kernel)

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(item.dtype, out.dtype)

            self.assertEqual(np.ndarray, type(out))


    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=True)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_data = dip_azm.gradient_structure_tensor(in_data,
                                                     kernel=tensor_kernel)

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(item.dtype, out.dtype)

            self.assertEqual(cp.ndarray, type(out))

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        dip_azm_cp = DipAzm(use_cuda=True)
        dip_azm_np = DipAzm(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        kernel = (2, 2, 2)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_data_cp = dip_azm_cp.gradient_structure_tensor(in_data_cp,
                                                           kernel=kernel)
        out_data_np = dip_azm_np.gradient_structure_tensor(in_data_np,
                                                           kernel=kernel)

        try:
            for i in range(len(out_data_np)):
                out_cp = out_data_cp[i].compute()
                out_np = out_data_np[i].compute()
                np.testing.assert_array_almost_equal(out_cp.get(), out_np)
        except AssertionError as ae:
            total = diff = 0
            # Check if the percentage of mismatch is higher than 5.0 %
            for i in range(len(out_data_np)):
                out_cp = out_data_cp[i].compute()
                out_np = out_data_np[i].compute()
                unequal_pos = np.where(out_cp != out_np)
                total += len(out_cp.flatten())
                diff += len(unequal_pos[0])

            self.assertTrue(float((diff * 100)/total) > 5.0, msg=str(ae))
