### Edge Detection Attributes

|       **Atribute**        | **Status** | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:-------------------------:|:----------:|:-------:|:-------:|:-------------:|:-------------:|
|        Semblance          |    Ready   |    X    |    X    |       X       |       X       |
|       EIG Complex         |    Ready   |    X    |         |       X       |               |
| Gradient Structure Tensor |    Ready   |    X    |    X    |       X       |       X       |
|          Chaos            |    Ready   |    X    |    X    |       X       |       X       |
|     Volume Curvature      |    Ready   |    X    |    X    |       X       |       X       |

#### Observations:

* The attribute *EIG Complex* requires function `eigvals()` which is not implemented in CuPy yet.
