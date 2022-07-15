"""Test util functions"""
import numpy as np

from unittest import TestCase
from scipy.signal import hilbert

from dasf_attributes.attributes import util


class TestUtil(TestCase):
    def test_hilbert_fixed(self):
        in_data = np.arange(20)

        lib_hilbert = util.hilbert(in_data)
        
        scipy_hilbert = hilbert(in_data)

        self.assertTrue(np.allclose(lib_hilbert, scipy_hilbert))
