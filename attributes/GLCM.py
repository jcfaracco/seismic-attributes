# -*- coding: utf-8 -*-
"""
Grey-Level Co-occurrence Matrix for Seismic Data

@author: Julio Faracco
@email: jcfaracco@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np

from . import util
from .Base import BaseAttributes

from skimage.feature import greycomatrix, greycoprops


class GLCMAttributes(BaseAttributes):
    """
    Description
    -----------
    Class object containing methods for computing GLCM attributes from 3D
    seismic data. This class does belongs to Base object it has his own
    create_array() method.

    Methods
    -------
    glcm_contrast
    glcm_dissimilarity
    glcm_homogeneity
    glcm_energy
    glcm_correlation
    glcm_asm
    """
    def _glcm_generic(self, darray, glcm_type, levels=256, direction=0, distance=1, preview=None):
        """
        Description
        -----------
        Compute GLCM attribute for each 2D input data

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

        kernel = (1, darray.shape[1], darray.shape[2])
        darray, chunks_init = self.create_array(darray, kernel, preview=preview)

        mi = da.min(darray)
        ma = da.max(darray)

        def __glcm_block(block, block_info=None):
            d, h, w,  = block.shape
            kh = kw = distance

            new_atts = list()
            for k in range(d):
                new_att = np.zeros((h, w), dtype=np.float32)

                bins = np.linspace(mi, ma + 1, levels)
                gl = np.digitize(block[k, :, :], bins) - 1

                for i in range(h):
                    for j in range(w):
                        #windows needs to fit completely in image
                        if i < kh or j < kw:
                            continue
                        if i > (h - kh - 1) or j > (w - kw - 1):
                            continue

                        #Calculate GLCM on a 7x7 window
                        glcm_window = gl[i - kh:i + kh + 1, j - kw:j + kw + 1].astype(int)
                        glcm = greycomatrix(glcm_window, [distance],
                                            [direction], levels=levels,
                                            symmetric=True, normed=True)

                        #Calculate contrast and replace center pixel
                        new_att[i, j] = greycoprops(glcm, glcm_type)
                new_atts.append(new_att.astype(darray.type))

            return da.from_delayed(news_atts,dtype=darray.dtype,
                                   shape=darray.shape)

        glcm = darray.map_blocks(__glcm_block, dtype=darray.dtype)
        result = util.trim_dask_array(glcm, kernel)

        return(result)


    def glcm_contrast(self, darray, preview=None):
        return self._glcm_generic(darray, "contrast")

    def glcm_dissimilarity(self, darray, preview=None):
        return self._glcm_generic(darray, "dissimilarity")


