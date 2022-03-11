# -*- coding: utf-8 -*-
"""
Local Binary Pattern Attributes for Seismic Data

@author: Julio Faracco
@email: jcfaracco@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np

try:
    import cupy as cp

    USE_CUPY = True
except:
    USE_CUPY = False

from . import util
from .Base import BaseAttributes

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
    def __init__(self, use_cuda=False):
        """
        Description
        -----------
        Constructor of LBP attribute class.

        Keywork Arguments
        ----------
        use_cuda : Boolean, variable to set CUDA usage
        """
        super().__init__(use_cuda=use_cuda)

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

        if USE_CUPY and self._use_cuda:
            kernel = (darray.shape[0], darray.shape[1], darray.shape[2])
        else:
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

        def __local_binary_pattern_diag_3d_cu(block):
            __lbp_gpu = cp.RawKernel(r'''
                extern "C" __global__
                void local_binary_pattern_gpu(const float *a, float *out,
                                              float max_local, float min_local,
                                              unsigned int nx, unsigned int ny,
                                              unsigned int nz) {
                    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
                    unsigned int idz = threadIdx.z + blockIdx.z * blockDim.z;
                    int i, j, k;
                    float max, min;
                    unsigned int center, index, kernel_idx;
                    unsigned int max_idx, min_idx;
                    float exp, sum, mult, n;

                    unsigned char kernel[27] = {
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,
                    };

                    max = min_local - 1;
                    min = max_local + 1;

                    if ((idx > 0 && idy > 0 && idz > 0) &&
                        (idx < nx) && (idy < ny) && (idz < nz)) {
                        center = ((ny * nx) * idz) + (idy * nx + idx);

                        for(i = -1; i <= 1; i = i + 2) {
                            for(j = -1; j <= 1; j = j + 2) {
                                for(k = -1; k <= 1; k = k + 2) {
                                    /* Avoid illegal memory access */
                                    if ((idx + i >= nx || idy + j >= ny || idz + k >= nz) ||
                                        (idx + i < 0 || idy + j < 0 || idz + k < 0)) {
                                        continue;
                                    }

                                    index = ((ny * nx) * (idz + k)) + ((idy + j) * nx + (idx + i));
                                    kernel_idx = (9 * (k + 1)) + ((j + 1) * 3 + (i + 1));

                                    if (max < a[index]) {
                                        if (a[center] < a[index]) {
                                            max = a[index];
                                            max_idx = kernel_idx;
                                        }
                                    }

                                    if (min > a[index]) {
                                        if (a[center] > a[index]) {
                                            min = a[index];
                                            min_idx = kernel_idx;
                                        }
                                    }
                                }
                            }
                        }

                        if (max < max_local + 1 && max > min_local - 1) {
                            kernel[max_idx] = 1;
                        }

                        if (min > min_local - 1 && min < max_local + 1) {
                            kernel[min_idx] = 1;
                        }

                        mult = exp = sum = 0;

                        for(k = 0; k <= 2; k = k + 2) {
                            for(j = 0; j <= 2; j = j + 2) {
                                for(i = 0; i <= 2; i = i + 2) {
                                    if (kernel[(9 * k) + (j * 3 + i)] == 1) {
                                        /* Implementing our own pow() function */
                                        n = 0;
                                        mult = 1;
                                        while(n < exp) {
                                            mult *= 2;
                                            n++;
                                        }
                                        sum += mult;
                                    }
                                    exp++;
                                }
                            }
                        }

                        out[center] = sum;
                    }
                }
            ''', 'local_binary_pattern_gpu')

            dimz = block.shape[0]
            dimy = block.shape[1]
            dimx = block.shape[2]

            out = cp.zeros((dimz * dimy * dimx), dtype=cp.float32)
            inp = cp.asarray(block, dtype=cp.float32)

            # Numpy is faster than Cupy for min and max
            min_local = np.min(block.flatten())
            max_local = np.max(block.flatten())

            block_size = 10

            grid = (int(np.ceil(dimz/block_size)),
                    int(np.ceil(dimy/block_size)),
                    int(np.ceil(dimx/block_size)),)
            block = (block_size, block_size, block_size,)

            __lbp_gpu(grid, block, (inp, out, cp.float32(max_local.get()),
                                    cp.float32(min_local.get()), cp.int32(dimx),
                                    cp.int32(dimy), cp.int32(dimz)))

            unique_array = cp.unique(out)
            for i, e in enumerate(unique_array):
                out[out == e] = i

            return(cp.asnumpy(out).reshape(dimz, dimy, dimx))

        if USE_CUPY:
            lbp_diag_3d = darray.map_blocks(__local_binary_pattern_diag_3d_cu, dtype=cp.float32)
        else:
            lbp_diag_3d = darray.map_blocks(__local_binary_pattern_diag_3d, dtype=darray.dtype)
        result = util.trim_dask_array(lbp_diag_3d, kernel, hw)

        return(result)
