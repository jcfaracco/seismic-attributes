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
except ImportError:
    pass

from . import util
from .base import BaseAttributes

from skimage.feature import local_binary_pattern


class LBPAttributes(BaseAttributes):
    """
    Description
    -----------
    Class object containing methods for computing local binary pattern
    attributes from 3D seismic data. This class does belongs to Base
    object it has his own create_array() method.

    Methods
    -------
    local_binary_pattern_2d
    local_binary_pattern_diag_3d
    """
    def local_binary_pattern_2d(self, darray, preview=None):
        """
        Description
        -----------
        Compute Local Binary Filters for 2D input data

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
        hw = (0, 2, 0)
        radius = 3
        neighboors = radius * 8
        method = 'default'

        darray, chunks_init = self.create_array(darray, kernel, hw=hw,
                                                boundary='periodic',
                                                preview=preview)

        def __local_binary_pattern(block, block_info=None):
            sub_cube = list()
            for i in range(0, block.shape[0]):
                sub_cube.append(local_binary_pattern(block[i, :, :],
                                                     neighboors,
                                                     radius, method))
            return np.asarray(sub_cube)

        if util.is_cupy_enabled(self._use_cuda):
            raise NotImplementedError("CuCIM does not have any LBP method "
                                      "implemented yet")
        else:
            lbp = darray.map_blocks(__local_binary_pattern, dtype=darray.dtype)

        result = util.trim_dask_array(lbp, kernel, hw)

        return result

    def local_binary_pattern_diag_3d(self, darray, preview=None):

        kernel = (4, 0, 0)

        darray, chunks_init = self.create_array(darray, kernel=kernel,
                                                boundary='periodic',
                                                preview=preview)

        def __local_binary_pattern_unique(block, unique_array):
            for i in range(len(unique_array)):
                block[block == unique_array[i]] = i

            return block

        def __local_binary_pattern_diag_3d(block):
            img_lbp = np.zeros_like(block)
            neighboor = 3
            s0 = int(neighboor/2)
            for ih in range(0, block.shape[0] - neighboor + s0):
                for iw in range(0, block.shape[1] - neighboor + s0):
                    for iz in range(0, block.shape[2] - neighboor + s0):
                        img = block[ih:ih + neighboor,
                                    iw:iw + neighboor,
                                    iz:iz + neighboor]
                        center = img[1, 1, 1]
                        img_aux_vector = img.flatten()

                        # Delete centroids
                        del_vec = [1, 3, 4, 5, 7, 9, 10, 11, 12, 13,
                                   14, 15, 16, 17, 19, 21, 22, 23, 25]

                        img_aux_vector = np.delete(img_aux_vector, del_vec)

                        weights = 2 ** np.arange(len(img_aux_vector),
                                                 dtype=np.uint64)

                        mask_vec = np.zeros(len(img_aux_vector), dtype=np.int8)

                        idx_max = img_aux_vector.argmax()
                        idx_min = img_aux_vector.argmin()

                        if img_aux_vector[idx_max] > center:
                            mask_vec[idx_max] = 1

                        if img_aux_vector[idx_min] < center:
                            mask_vec[idx_min] = 1

                        num = np.sum(weights * mask_vec)

                        img_lbp[ih + 1, iw + 1, iz + 1] = num
            return img_lbp

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
                        (idx < nx - 1) && (idy < ny - 1) && (idz < nz - 1)) {
                        center = ((ny * nz) * idx) + (idy * nz + idz);

                        for(i = -1; i <= 1; i = i + 2) {
                            for(j = -1; j <= 1; j = j + 2) {
                                for(k = -1; k <= 1; k = k + 2) {
                                    /* Avoid illegal memory access */
                                    if ((idx + i >= nx ||
                                         idy + j >= ny ||
                                         idz + k >= nz) ||
                                        (idx + i < 0 ||
                                         idy + j < 0 ||
                                         idz + k < 0)) {
                                        continue;
                                    }

                                    index = (((ny * nz) * (idx + i)) +
                                             ((idy + j) * nz + (idz + k)));
                                    kernel_idx = ((9 * (i + 1)) + ((j + 1) *
                                                   3 + (k + 1)));

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

                        for(i = 0; i <= 2; i = i + 2) {
                            for(j = 0; j <= 2; j = j + 2) {
                                for(k = 0; k <= 2; k = k + 2) {
                                    if (kernel[(9 * i) + (j * 3 + k)] == 1) {
                                        // Implementing our own pow() function
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

            dimx = block.shape[0]
            dimy = block.shape[1]
            dimz = block.shape[2]

            out = cp.zeros((dimz * dimy * dimx), dtype=cp.float32)
            inp = cp.asarray(block, dtype=cp.float32)

            # Numpy is faster than Cupy for min and max
            min_local = np.min(block.flatten()).get()
            max_local = np.max(block.flatten()).get()

            block_size = 10

            grid = (int(np.ceil(dimx/block_size)),
                    int(np.ceil(dimy/block_size)),
                    int(np.ceil(dimz/block_size)),)
            block = (block_size, block_size, block_size,)

            __lbp_gpu(grid, block, (inp, out, cp.float32(max_local),
                                    cp.float32(min_local), cp.int32(dimx),
                                    cp.int32(dimy), cp.int32(dimz)))

            # XXX: we need to handle Numpy here due to Dask issue #7482
            return (cp.asnumpy(out).reshape(dimx, dimy, dimz))

        if util.is_cupy_enabled(self._use_cuda):
            lbp_diag_3d = darray.map_blocks(__local_binary_pattern_diag_3d_cu,
                                            dtype=cp.float32)
        else:
            lbp_diag_3d = darray.map_blocks(__local_binary_pattern_diag_3d,
                                            dtype=darray.dtype)
        result = util.trim_dask_array(lbp_diag_3d, kernel, hw)

        unique = da.unique(result)

        result = result.map_blocks(__local_binary_pattern_unique, unique,
                                   dtype=result.dtype)

        return result
