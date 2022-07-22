"""Test Base Attribute functions"""

import pytest
import inspect
import unittest
import numpy as np

from unittest import TestCase

from dasf_attributes.attributes import util
from dasf_attributes.attributes.base import BaseAttributes


class TestCreateArray(TestCase):
    def test_create_array_no_arguments(self):
        in_shape = (20, 20, 20)
        
        rnd = np.random.random(in_shape)
        
        base = BaseAttributes()
        
        darray, chunks = base.create_array(rnd)
        
        self.assertEqual(darray.chunks, chunks)
        self.assertEqual(darray.shape, in_shape)
        
    def test_create_array_only_kernel(self):
        kernel = (4, 4, 4)
        in_shape = (20, 20, 20)
        not_trim_shape = (28, 28, 24)
        
        rnd = np.random.random(in_shape)
        
        base = BaseAttributes()
        
        darray, chunks = base.create_array(rnd, kernel=kernel)
        
        self.assertEqual(darray.chunks, chunks)
        self.assertEqual(darray.shape, not_trim_shape)
        
    def test_create_array_kernel_hw(self):
        kernel = (4, 4, 4)
        hw = (2, 2, 2)
        in_shape = (20, 20, 20)
        not_trim_shape = (28, 28, 24)
        
        rnd = np.random.random(in_shape)
        
        base = BaseAttributes()
        
        darray, chunks = base.create_array(rnd, kernel=kernel)
        
        self.assertEqual(darray.chunks, chunks)
        self.assertEqual(darray.shape, not_trim_shape)
        
    def test_create_array_map_not_trim(self):
        def sum_random(block):
            new = np.random.random(block.shape)
            
            return block + new

        kernel = (4, 4, 4)
        hw = (2, 2, 2)
        in_shape = (20, 20, 20)
        not_trim_shape = (28, 28, 24)
        
        rnd = np.random.random(in_shape)
        
        base = BaseAttributes()
        
        darray, chunks = base.create_array(rnd, kernel=kernel)
        
        out_data = darray.map_blocks(sum_random)
        
        out_data_np = out_data.compute()
        
        self.assertEqual(out_data_np.shape, not_trim_shape)
        
    def test_create_array_map_trim(self):
        def sum_random(block):
            new = np.random.random(block.shape)
            
            return block + new

        kernel = (4, 4, 4)
        hw = (2, 2, 2)
        in_shape = (20, 20, 20)
        
        rnd = np.random.random(in_shape)
        
        base = BaseAttributes()
        
        darray, chunks = base.create_array(rnd, kernel=kernel)
        
        out_data = darray.map_blocks(sum_random)
        
        out_data = util.trim_dask_array(out_data, kernel=kernel, hw=hw)
        
        out_data_np = out_data.compute()
        
        self.assertEqual(out_data_np.shape, in_shape)
