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
from scipy.interpolate import RegularGridInterpolator as RGI


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

        kernel = (1, min(int(darray.shape[1]/4), 1000), int(darray.shape[2]))
        hw = (0, 2, 0)
        radius = 3
        neighboors = radius * 8
        method = 'default'

        if not isinstance(darray, da.core.Array):
            darray = da.from_array(darray, chunks=kernel)

        darray, chunks_init = self.create_array(darray, kernel, hw=hw, boundary='periodic', preview=preview)

        def __local_binary_pattern(block, block_info=None):
            sub_cube = list()
            for i in range(0, block.shape[0]):
                sub_cube.append(local_binary_pattern(block[i, :, :], neighboors, radius, method))
            return da.from_array(np.array(sub_cube))
        lbp = darray.map_blocks(__local_binary_pattern, dtype=darray.dtype)
        result = util.trim_dask_array(lbp, kernel, hw)

        return(result)


    def local_binary_pattern_diag_3d(self, darray, preview=None):

        hw = (2, 0, 0)
        kernel = (min(int((darray.shape[0] + 4)/4), 1000), darray.shape[1], darray.shape[2])

        if not isinstance(darray, da.core.Array):
            darray = da.from_array(darray, chunks=kernel)

        darray, chunks_init = self.create_array(darray, kernel, hw=hw, boundary='periodic', preview=preview)

        def __local_binary_pattern_diag_3d(block):
            img_lbp = np.zeros_like(block)
            neighboor = 3
            s0 = int(neighboor/2)
            for ih in range(0, block.shape[0] - neighboor + s0):
                for iw in range(0, block.shape[1] - neighboor + s0):
                    for iz in range(0, block.shape[2] - neighboor + s0):
                        img = block[ih:ih+neighboor,iw:iw+neighboor,iz:iz+neighboor]
                        center = img[1,1]
                        img_aux = (img >= center)*1.0
                        img_aux_vector = img_aux.flatten()

                        # Delete centroids
                        del_vec = [0, 2, 6, 8, 9, 11, 15, 17, 18, 20, 24, 26]
                        img_aux_vector = np.delete(img_aux_vector, del_vec)

                        where_img_aux_vector = np.where(img_aux_vector)[0]
                        if len(where_img_aux_vector) >= 1:
                            num = np.sum(2 ** where_img_aux_vector)
                        else:
                            num = 0
                        img_lbp[ih+1,iw+1,iz+1] = num
            return(img_lbp)

        lbp_diag_3d = darray.map_blocks(__local_binary_pattern_diag_3d, dtype=darray.dtype)
        result = util.trim_dask_array(lbp_diag_3d, kernel, hw)

        return(result)
