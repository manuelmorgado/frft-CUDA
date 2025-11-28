# Fractional Fourier Transform (FrFT) with CUDA

Implementation of a CUDA 2D plan for Fractional Fourier Transform with the sequence:

output -> scale * modulator * ifft2( filtor_fft * fft2(modulator * in) ) / N^2
---
## Kernels

- kernel_build_chirps_2d -> 2D modulator and filtor (time-domain) on a square grid NxN
- kernel_pointwise_mul_mod -> Pointwise: out = data * mod
- kernel_pointwise_mul_inplace -> Pointwise: data *= filtor_fft
- kernel_conjugate -> Pointwise: out = conjugate(in)
- kernel_pointwise_inv -> Pointwise: out = 1 / in   (complex reciprocal)
- kernel_final_scale_mod -> out = scale * (1 / (N^2)) * modulator * data
- kernel_final_scale_mod_inverse -> out = scale_inv * mod_inv * data

After preparing and building the CUDA event, the execution occurs when it is execute as:

```
frft2d_chirp_execute(ctx, h_in, h_out);
```
---
## Compilation

``` "%CUDA_PATH%\bin\nvcc" frft2d_benchmark.cu -I"%CUDA_PATH%\include" -L"%CUDA_PATH%\lib\x64" -lcufft -o frft2d_benchmark.exe ```

## Author

Manuel Morgado
