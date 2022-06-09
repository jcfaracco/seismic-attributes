# -*- coding: utf-8 -*-
"""
Frequency attributes for Seismic data

@author: Braden Fitz-Gerald
@email: braden.fitzgerald@gmail.com

"""

# Import Libraries
import dask.array as da
import numpy as np
from scipy import signal

try:
    import cupy as cp
    import cusignal

    USE_CUPY = True
except:
    USE_CUPY = False

from . import util
from .Base import BaseAttributes


class Frequency(BaseAttributes):
    """
    Description
    -----------
    Class object containing methods for performing frequency filtering
    
    Methods
    -------
    lowpass_filter
    highpass_filter
    bandpass_filter
    cwt_ricker
    cwt_ormsby
    """
    def __init__(self, use_cuda=False):
        """
        Description
        -----------
        Constructor of Frequency attribute class.

        Keywork Arguments
        ----------
        use_cuda : Boolean, variable to set CUDA usage
        """
        super().__init__(use_cuda=use_cuda)

    def lowpass_filter(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform low pass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            if USE_CUPY and self._use_cuda:
                out = cusignal.filtfilt(B, A, x=chunk)
            else:
                out = signal.filtfilt(B, A, x=chunk)
            
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, freq/nyq, btype='lowpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
    def highpass_filter(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform high pass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            if USE_CUPY and self._use_cuda:
                out = cusignal.filtfilt(B, A, x=chunk)
            else:
                out = signal.filtfilt(B, A, x=chunk)
            
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, freq/nyq, btype='highpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
        
    def bandpass_filter(self, darray, freq_lp, freq_hp, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform bandpass filtering of 3D seismic data
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq_lp : Number (Hz), frequency cutoff used in low pass filter
        freq_hp : Number (Hz), frequency cutoff used in high pass filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Filtering Function
        def filt(chunk, B, A):
            if USE_CUPY and self._use_cuda:
                out = cusignal.filtfilt(B, A, x=chunk)
            else:
                out = signal.filtfilt(B, A, x=chunk)

            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)        
        fs = 1000 / sample_rate
        nyq = fs * 0.5        
        B, A = signal.butter(6, (freq_lp/nyq, freq_hp/nyq), btype='bandpass', analog=False)        
        result = darray.map_blocks(filt, B, A, dtype=darray.dtype)
        
        return(result)
        
        
    def cwt_ricker(self, darray, freq, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform Continuous Wavelet Transform using Ricker Wavelet
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : Number (Hz), frequency defining Ricker Wavelet
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        
        # Generate wavelet of specified frequency
        def wavelet(freq, sample_rate):
            
            sr = sample_rate / 1000            
            t = np.arange(-0.512 / 2, 0.512 / 2, sr)
            out = (1 - (2 * (np.pi * freq * t) ** 2)) * np.exp(-(np.pi * freq * t) ** 2)
            
            return(out)
            
        # Convolve wavelet with trace
        def convolve(chunk, w):
            
            out = np.zeros(chunk.shape)
            
            for i,j in np.ndindex(chunk.shape[:-1]):                
                out[i, j] = signal.fftconvolve(chunk[i, j], w, mode='same')
                
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)
        w = wavelet(freq, sample_rate)
        result = darray.map_blocks(convolve, w=w, dtype=darray.dtype)
        
        return(result)
        
        
    def cwt_ormsby(self, darray, freqs, sample_rate=4, preview=None):
        """
        Description
        -----------
        Perform Continuous Wavelet Transform using Ormsby Wavelet
        
        Parameters
        ----------
        darray : Array-like, acceptable inputs include Numpy, HDF5, or Dask Arrays
        freq : tuple (len 4), frequency cutoff used in filter
        
        Keywork Arguments
        -----------------  
        sample_rate : Number, sample rate in milliseconds (ms)
        preview : str, enables or disables preview mode and specifies direction
            Acceptable inputs are (None, 'inline', 'xline', 'z')
            Optimizes chunk size in different orientations to facilitate rapid
            screening of algorithm output
        
        Returns
        -------
        result : Dask Array
        """
        # Generate wavelet of specified frequencyies
        def wavelet(freqs, sample_rate):
            
            f1, f2, f3, f4 = freqs
            sr = sample_rate / 1000            

            if USE_CUPY and self._use_cuda:
                t = cp.arange(-0.512 / 2, 0.512 / 2, sr)
            else:
                t = np.arange(-0.512 / 2, 0.512 / 2, sr)
            
            term1 = (((np.pi * f4) ** 2) / ((np.pi * f4) - (np.pi * f3))) * np.sinc(np.pi * f4 * t) ** 2
            term2 = (((np.pi * f3) ** 2) / ((np.pi * f4) - (np.pi * f3))) * np.sinc(np.pi * f3 * t) ** 2
            term3 = (((np.pi * f2) ** 2) / ((np.pi * f2) - (np.pi * f1))) * np.sinc(np.pi * f2 * t) ** 2
            term4 = (((np.pi * f1) ** 2) / ((np.pi * f2) - (np.pi * f1))) * np.sinc(np.pi * f1 * t) ** 2
            
            out = (term1 - term2) - (term3 - term4)
            
            return(out)
            
        # Convolve wavelet with trace
        def convolve(chunk, w):
            
            if USE_CUPY and self._use_cuda:
                out = cp.zeros(chunk.shape)
            else:
                out = np.zeros(chunk.shape)
            
            for i,j in np.ndindex(chunk.shape[:-1]):                
                if USE_CUPY and self._use_cuda:
                    out[i, j] = cusignal.fftconvolve(chunk[i, j], w, mode='same')
                else:
                    out[i, j] = signal.fftconvolve(chunk[i, j], w, mode='same')
                
            return(out)
        
        # Generate Dask Array as necessary and perform algorithm
        darray, chunks_init = self.create_array(darray, kernel=None, 
                                                preview=preview)
        w = wavelet(freqs, sample_rate)
        result = darray.map_blocks(convolve, w=w, dtype=darray.dtype)
        
        return(result)
