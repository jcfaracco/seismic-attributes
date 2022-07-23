"""Test EdgeDetection functions"""

import unittest
import numpy as np

import dask.array as da

try:
    import cupy as cp
except Exception:
    pass

from unittest import TestCase

from dasf_attributes.attributes import util, DipAzm, EdgeDetection


class TestVolumeCurvature(TestCase):
    @staticmethod
    def __remove_function_prefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def test_shape_check_from_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())
        edge = EdgeDetection(use_cuda=util.is_cupy_enabled())

        in_shape = (20, 20, 20)
        tensor_kernel = (2, 2, 2)

        if util.is_cupy_enabled():
            rng = cp.random.default_rng(seed=42)
            in_data = rng.random(in_shape)
        else:
            rng = np.random.default_rng(seed=42)
            in_data = rng.random(in_shape)

        out_dip = dip_azm.gradient_dips(in_data, kernel=tensor_kernel)
        out_data = edge.volume_curvature(darray_il=out_dip[0],
                                         darray_xl=out_dip[1])

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(in_shape, out.shape)

    def test_shape_check_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=util.is_cupy_enabled())
        edge = EdgeDetection(use_cuda=util.is_cupy_enabled())

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

        out_dip = dip_azm.gradient_dips(in_data, kernel=tensor_kernel)
        out_data = edge.volume_curvature(darray_il=out_dip[0],
                                         darray_xl=out_dip[1])

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(in_shape, out.shape)

    def test_dtype_as_np_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=False)
        edge = EdgeDetection(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        rng = np.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_dip = dip_azm.gradient_dips(in_data, kernel=tensor_kernel)
        out_data = edge.volume_curvature(darray_il=out_dip[0],
                                         darray_xl=out_dip[1])

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(item.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_dtype_as_cp_from_dask_array(self):
        dip_azm = DipAzm(use_cuda=True)
        edge = EdgeDetection(use_cuda=True)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        rng = cp.random.default_rng(seed=42)
        in_data = rng.random(in_shape)

        # The input data is small, so we can import from array
        in_data = da.from_array(in_data, chunks=in_shape_chunks)

        out_dip = dip_azm.gradient_dips(in_data, kernel=tensor_kernel)
        out_data = edge.volume_curvature(darray_il=out_dip[0],
                                         darray_xl=out_dip[1])

        self.assertEqual(len(out_data), 6)

        for item in out_data:
            out = item.compute()

            self.assertEqual(item.dtype, out.dtype)

    @unittest.skipIf(not util.is_cupy_enabled(),
                     "not supported CUDA in this platform")
    def test_compare_attributes_cross_platforms(self):
        dip_azm_cp = DipAzm(use_cuda=True)
        dip_azm_np = DipAzm(use_cuda=False)
        edge_cp = EdgeDetection(use_cuda=True)
        edge_np = EdgeDetection(use_cuda=False)

        in_shape = (20, 20, 20)
        in_shape_chunks = (5, 5, 5)
        tensor_kernel = (2, 2, 2)

        rng = cp.random.default_rng(seed=42)
        in_data_cp = rng.random(in_shape)
        in_data_np = in_data_cp.copy().get()

        # The input data is small, so we can import from array
        in_data_cp = da.from_array(in_data_cp, chunks=in_shape_chunks)
        in_data_np = da.from_array(in_data_np, chunks=in_shape_chunks)

        out_dip_cp = dip_azm_cp.gradient_dips(in_data_cp, kernel=tensor_kernel)
        out_dip_np = dip_azm_np.gradient_dips(in_data_np, kernel=tensor_kernel)
        out_data_cp = edge_cp.volume_curvature(darray_il=out_dip_cp[0],
                                               darray_xl=out_dip_cp[1])
        out_data_np = edge_np.volume_curvature(darray_il=out_dip_np[0],
                                               darray_xl=out_dip_np[1])

        try:
            for i in range(len(out_data_np)):
                out_cp = out_data_cp[i].compute()
                out_np = out_data_np[i].compute()
                np.testing.assert_array_almost_equal(out_cp, out_np)
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
