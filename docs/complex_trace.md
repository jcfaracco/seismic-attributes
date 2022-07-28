### Complex Trace Attributes

From: https://doi.org/10.1190/1.1440994

The conventional seismic trace can be viewed as the real component of a complex trace which can be uniquely calculated under usual conditions. The complex trace permits the unique separation of envelope amplitude and phase information and the calculation of instantaneous frequency. These and other quantities can be displayed in a color‐encoded manner which helps an interpreter see their interrelationship and spatial changes. The significance of color patterns and their geological interpretation is illustrated by examples of seismic data from three areas.

|       **Atribute**         | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:--------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|         Hilbert            |    Ready   |    X    |    X    |       X       |       X       |
|         Envelope           |    Ready   |    X    |    X    |       X       |       X       |
|   Instantaneous Phase      |  Unstable  |    X    |         |       X       |               |
| Cosine Instantaneous Phase |  Unstable  |    X    |         |       X       |               |
| Relative Amplitude Change  |    Ready   |    X    |    X    |       X       |       X       |
|  Instantaneous Frequency   |  Unstable  |    X    |         |       X       |               |
|  Instantaneous Bandwidth   |    Ready   |    X    |    X    |       X       |       X       |
|     Dominant Frequency     |  Unstable  |    X    |         |       X       |               |
|      Frequency Change      |  Unstable  |    X    |         |       X       |               |
|         Sweetness          |  Unstable  |    X    |         |       X       |               |
|       Quality Factor       |  Unstable  |    X    |         |       X       |               |
|       Response Phase       |  Unstable  |    X    |         |       X       |               |
|     Response Frequency     |    Ready   |    X    |    X    |       X       |       X       |
|     Response Amplitude     |  Unstable  |    X    |         |       X       |               |
|     Apparent Polarity      |  Unstable  |    X    |         |       X       |               |

#### Observations:

* The attributes *Cosine Instantaneous Phase*, *Dominant Frequency*, *Frequency Change*, *Instantaneous Frequency*, *Instantaneous Phase* and *Quality Factor* requires `angle()` but CuPy angle introduced `deg=False`. For further information, see issues:
  * Dask: [#9296](https://github.com/dask/dask/issues/9296)
  * CuPy: [#6900](https://github.com/cupy/cupy/issues/6900)
* The attributes *Apparent Polarity*, *Response Amplitude* and *Response Frequency* requires `cumsum()` which does not support CuPy:
  * Dask: [#9315](https://github.com/dask/dask/issues/9315)
  
