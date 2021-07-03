# -*- coding: utf-8 -*-
"""
Base Attributes for Seismic Data

@author: Julio Faracco
@email: jcfaracco@gmail.com

"""

# Import Libraries
import sys
import dask.array as da
import numpy as np

try:
    import cupy as cp

    USE_CUPY = True
except Exception:
    USE_CUPY = False

from . import util


class BaseAttributes(object):
    """
    Description
    -----------
    Class object containing generic methods to all attributes

    Methods
    -------
    create_array
    """
    def __init__(self, use_cuda=False):
        """
        Description
        -----------
        Constructor of base attributes class.

        Keywork Arguments
        ----------
        use_cuda : Boolean, variable to set CUDA usage
        """
        self.__use_cuda = use_cuda

        if not USE_CUPY and self.__use_cuda:
            print("Warning: skipping CUDA usage. Check if this system "
                  "supports CUDA or if it installed", file=sys.stderr)

    def create_array(self, darray, kernel=None, hw=None, boundary='reflect',
                     preview=None):
        """
        Description
        -----------
        Convert input to Dask Array with ideal chunk size as necessary. Perform
        necessary ghosting as needed for opertations utilizing windowed
        functions.

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
            Arrays

        Keywork Arguments
        -----------------
        kernel : tuple (len 3), operator size
        hw : tuple (len 3), height and width sizes
        boundary : str, indicates data reflection between data chunks
            For further reference see Dask Overlaping options.
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output

        Returns
        -------
        darray : Dask Array
        chunk_init : tuple (len 3), chunk size before ghosting.  Used in select
            cases
        """

        # Compute chunk size and convert if not a Dask Array
        if not isinstance(darray, da.core.Array):
            chunk_size = util.compute_chunk_size(darray.shape,
                                                 darray.dtype.itemsize,
                                                 kernel=kernel,
                                                 preview=preview)
            darray = da.from_array(darray, chunks=chunk_size)
            chunks_init = darray.chunks
        else:
            chunks_init = darray.chunks

        # Ghost Dask Array if operation specifies a kernel
        if kernel is not None:
            if hw is None:
                hw = tuple(np.array(kernel) // 2)
            darray = da.overlap.overlap(darray, depth=hw, boundary=boundary)

        return(darray, chunks_init)
