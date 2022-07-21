### Signal Process Attributes

|       **Atribute**         | **Description** | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:--------------------------:|:---------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|     First Derivative       |                 |    Ready   |    X    |    X    |       X       |       X       |
|    Second Derivative       |                 |    Ready   |    X    |    X    |       X       |       X       |
|   Histogram Equalization   |                 |    Ready   |    X    |    X    |       X       |       X       |
|        Time Gain           |                 |    Ready   |    X    |    X    |       X       |       X       |
|   Rescale Amplitude Range  |                 |    Ready   |    X    |    X    |       X       |       X       |
|           RMS              |                 |    Ready   |    X    |    X    |       X       |       X       |
|        Trace AGC           |                 |    Ready   |    X    |    X    |       X       |       X       |
|    Gradient Magnitude      |                 |    Ready   |    X    |    X    |       X       |       X       |
|   Reflection Intensity     |                 |  Unstable  |    X    |         |       X       |               |
|     Phase Rotation         |                 |    Ready   |    X    |    X    |       X       |       X       |


##### Observations:

*Reflection Intensity* attribute requires `trapz()` function. It is only available in CuPy release 11.0.0.
