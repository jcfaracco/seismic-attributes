### Gray Level Co-occurence Matrix Attribute

|       **Atribute**        | **Description** |  **Status**  | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:-------------------------:|:---------------:|:------------:|:-------:|:-------:|:-------------:|:-------------:|
|      GLCM Contrast        |                 |     Ready    |    X    |    X    |       X       |       X       |
|    GLCM Dissimilarity     |                 |   Unstable   |    X    |         |       X       |               |
|     GLCM Homogeneity      |                 |     Ready    |    X    |    X    |       X       |       X       |
|       GLCM Energy         |                 |     Ready    |    X    |    X    |       X       |       X       |
|     GLCM Correlation      |                 |     Ready    |    X    |    X    |       X       |       X       |
|        GLCM ASM           |                 |     Ready    |    X    |    X    |       X       |       X       |
|        GLCM Mean          |                 |   Unstable   |         |    X    |               |       X       |
|      GLCM Variance        |                 |   Unstable   |         |    X    |               |       X       |

#### Observations:

* The attribute *GLCM Dissimilarity* requires latest changes of [glcm-cupy](https://github.com/Eve-ning/glcm-cupy) to support CUDA.
* The attributes *GLCM Mean* and *GLCM Variance* are not implemented into scikit image.
