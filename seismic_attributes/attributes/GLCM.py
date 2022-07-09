# -*- coding: utf-8 -*-
"""
Grey-Level Co-occurrence Matrix for Seismic Data

@author: Julio Faracco
@email: jcfaracco@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np

from .Base import BaseAttributes

try:
    import cupy as cp

    from glcm_cupy import glcm as glcm_gpu
    from glcm_cupy import conf as glcm_conf

    USE_CUPY = True
except Exception:
    USE_CUPY = False

try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:
    # XXX: Deprecated after release 0.19 of scikit-image
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops


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
    glcm_mean
    glcm_var
    """
    def _glcm_generic(self, darray, glcm_type, levels=256, direction=0,
                      distance=1, preview=None):
        """
        Description
        -----------
        Compute GLCM attribute for each 2D input data

        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask
            Arrays

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

        if not isinstance(darray, da.core.Array):
            darray = da.from_array(darray, chunks=kernel)

        mi = da.min(darray)
        ma = da.max(darray)

        def __glcm_block(block, glcm_type_block, levels_block, direction_block,
                         distance_block, glb_mi, glb_ma, block_info=None):
            d, h, w, = block.shape
            kh = kw = distance_block

            new_atts = list()
            for k in range(d):
                new_att = np.zeros((h, w), dtype=np.float32)

                bins = np.linspace(glb_mi, glb_ma + 1, levels_block)
                gl = np.digitize(block[k, :, :], bins) - 1

                for i in range(h):
                    for j in range(w):
                        # Windows needs to fit completely in image
                        if i < kh or j < kw:
                            continue
                        if i > (h - kh - 1) or j > (w - kw - 1):
                            continue

                        # Calculate GLCM on a 7x7 window
                        glcm_window = gl[i - kh:i + kh + 1,
                                         j - kw:j + kw + 1].astype(int)
                        glcm = graycomatrix(glcm_window, [distance_block],
                                            [direction_block],
                                            levels=levels_block,
                                            symmetric=True, normed=True)

                        # Calculate contrast and replace center pixel
                        new_att[i, j] = graycoprops(glcm, glcm_type_block)
                new_atts.append(new_att.astype(block.dtype))

            return np.asarray(new_atts, dtype=block.dtype)

        def __glcm_block_cu(block, glcm_type_block, levels_block,
                            direction_block, distance_block, glb_mi, glb_ma,
                            block_info=None):
            def pad_with(vector, pad_width, iaxis, kwargs):
                pad_value = kwargs.get('padder', 10)
                vector[:pad_width[0]] = pad_value
                vector[-pad_width[1]:] = pad_value

            d, h, w, = block.shape

            bins = cp.linspace(glb_mi, glb_ma + 1, 255)
            gl = cp.digitize(block, bins) - 1

            new_atts = list()
            for k in range(d):
                image = gl[k, :, :, cp.newaxis]
                g = glcm_gpu(image, bin_from=256, bin_to=levels_block)
                new_atts.append(cp.pad(cp.asarray(g[...,
                                       glcm_type_block].squeeze(axis=2)), 3,
                                       pad_with, padder=0))

            return cp.asarray(new_atts, dtype=block.dtype)

        if USE_CUPY and self._use_cuda:
            if glcm_type == "contrast":
                glcm_type = glcm_conf.CONTRAST
            elif glcm_type == "homogeneity":
                glcm_type = glcm_conf.HOMOGENEITY
            elif glcm_type == "asm":
                glcm_type = glcm_conf.ASM
            elif glcm_type == "mean":
                glcm_type = glcm_conf.MEAN
            elif glcm_type == "correlation":
                glcm_type = glcm_conf.CORRELATION
            elif glcm_type == "var":
                glcm_type = glcm_conf.VAR
            else:
                raise Exception("GLCM type '%s' is not supported." % glcm_type)

            result = darray.map_blocks(__glcm_block_cu, glcm_type, levels,
                                       direction, distance, mi, ma,
                                       dtype=darray.dtype)
        else:
            result = darray.map_blocks(__glcm_block, glcm_type, levels,
                                       direction, distance, mi, ma,
                                       dtype=darray.dtype)

        return(result)

    def glcm_contrast(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "contrast", levels)

    def glcm_dissimilarity(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "dissimilarity", levels)

    def glcm_asm(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "asm", levels)

    def glcm_mean(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "mean", levels)

    def glcm_correlation(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "correlation", levels)

    def glcm_homogeneity(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "homogeneity", levels)

    def glcm_var(self, darray, levels=256, preview=None):
        return self._glcm_generic(darray, "var", levels)
