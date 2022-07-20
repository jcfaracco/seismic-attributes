# -*- coding: utf-8 -*-
"""
Noise Reduction attributes for Seismic Data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
from scipy import ndimage as ndi

try:
    from cupyx.scipy import ndimage as cundi
except ImportError:
    pass

from . import util
from .base import BaseAttributes


class NoiseReduction(BaseAttributes):
    """
    Description
    -----------
    Class object containing methods for reducing noise in 3D seismic

    Methods
    -------
    gaussian
    median
    convolution
    """
    def __init__(self, use_cuda=False):
        """
        Description
        -----------
        Constructor of noise reduction attribute class.

        Keywork Arguments
        -----------------
        use_cuda : Boolean, variable to set CUDA usage
        """
        super().__init__(use_cuda=use_cuda)

    def gaussian(self, darray, sigmas=(1, 1, 1), preview=None):
        """
        Description
        -----------
        Perform gaussian smoothing of input seismic

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
            Arrays

        Keywork Arguments
        -----------------
        sigmas : tuple (len 3), smoothing parameters in I, J, K
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output

        Returns
        -------
        result : Dask Array
        """

        # Generate Dask Array as necessary and perform algorithm
        kernel = tuple((np.array(sigmas) * 2.5).astype(int))
        darray, chunks_init = self.create_array(darray, kernel,
                                                preview=preview)
        if util.is_cupy_enabled(self._use_cuda):
            result = darray.map_blocks(cundi.gaussian_filter, sigma=sigmas,
                                       dtype=darray.dtype)
        else:
            result = darray.map_blocks(ndi.gaussian_filter, sigma=sigmas,
                                       dtype=darray.dtype)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0

        return result

    def median(self, darray, kernel=(3, 3, 3), preview=None):
        """
        Description
        -----------
        Perform median smoothing of input seismic data

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
            Arrays

        Keywork Arguments
        -----------------
        kernel : tuple (len 3), operator size in I, J, K
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output

        Returns
        -------
        result : Dask Array
        """

        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel,
                                                preview=preview)
        if util.is_cupy_enabled(self._use_cuda):
            result = darray.map_blocks(cundi.median_filter, size=kernel,
                                       dtype=darray.dtype)
        else:
            result = darray.map_blocks(ndi.median_filter, size=kernel,
                                       dtype=darray.dtype)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0

        return result

    def convolution(self, darray, kernel=(3, 3, 3), preview=None):
        """
        Description
        -----------
        Perform convolution smoothing of input seismic data

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
            Arrays

        Keywork Arguments
        -----------------
        kernel : tuple (len 3), operator size in I, J, K
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output

        Returns
        -------
        result : Dask Array
        """

        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel,
                                                preview=preview)
        if util.is_cupy_enabled(self._use_cuda):
            result = darray.map_blocks(cundi.uniform_filter, size=kernel,
                                       dtype=darray.dtype)
        else:
            result = darray.map_blocks(ndi.uniform_filter, size=kernel,
                                       dtype=darray.dtype)
        result = util.trim_dask_array(result, kernel)
        result[da.isnan(result)] = 0

        return result
