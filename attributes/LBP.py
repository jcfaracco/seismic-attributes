# -*- coding: utf-8 -*-
"""
Complex Trace Attributes for Seismic Data

@author: Julio Faracco
@email: jcfaracco@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
import util

from Base import BaseAttributes

from skimage.feature import local_binary_pattern


class LBPAttributes(BaseAttributes):
    """
    Description
    -----------
    Class object containing methods for computing local binary pattern attributes 
    from 3D seismic data. This class does belongs to Base object it has his own
    create_array() method.
    
    Methods
    -------
    local_binary_pattern_2d
    """
    def local_binary_pattern_2d(self, darray, preview=None):
        """
        Description
        -----------
        Compute Local Binary Filters for 2D input data

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays

        Keywork Arguments
        -----------------
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output

        Returns
        -------
        result : Dask Array
        """

        kernel = (1,1,25)
        radius = 3
        neighboors = radius * 8
        method = 'default'

        if not isinstance(darray, da.core.Array):
            darray = da.from_array(darray, chunks=(int(darray.shape[0]/4), int(darray.shape[1]), 1))

        darray, chunks_init = self.create_array(darray, kernel, boundary='periodic', preview=preview)

        def __local_binary_pattern(block, block_info=None):
            sub_cube = list()
            for i in range(0, block.shape[0]):
                sub_cube.append(local_binary_pattern(block[i, :, :], neighboors, radius, method))
            return da.from_array(np.array(sub_cube))
        lbp = darray.map_blocks(__local_binary_pattern, dtype=darray.dtype)
        result = util.trim_dask_array(lbp, kernel)

        return(result)

