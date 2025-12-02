# frft-CUDA for Python

Implementation of python use of the CUDA Fractional Fourier Transform (FrFT) version, by generating a DLL file that is used with cTypes in python for a class ```FrFT2D``` that expose the functions forward/inverse for both directions of the FrFT.

### Use case in python

```python

import numpy as np
import time

import matplotlib.pyplot as plt
from frft2d_cuda import FrFT2D  

N = 1024
alpha = 0.5
frft = FrFT2D(N, alpha)

# complex matrix: real + i * imag
real_part = np.random.randn(N, N).astype(np.float32)
imag_part = np.random.randn(N, N).astype(np.float32)
inp = real_part + 1j * imag_part
inp = inp.astype(np.complex64)  # ensure complex64

print(inp.shape, inp.dtype)

t0 = time.perf_counter_ns()
out = frft.forward(inp)
t1 = time.perf_counter_ns()
rec = frft.inverse(out)
print((time.perf_counter_ns() - t1)*1e-6, 'ms')

plt.figure()
plt.imshow(inp.imag, origin='lower')

plt.figure()
plt.imshow(rec.imag, origin='lower')

plt.show()
# reconstruction error
diff = rec - inp
rms_err = np.sqrt(np.mean(np.abs(diff)**2))
max_err = np.max(np.abs(diff))

print(f"RMS error: {rms_err:.6e}")
print(f"Max error: {max_err:.6e}")

```
