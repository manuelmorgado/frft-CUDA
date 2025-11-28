// 2D fractional-like transform using a chirp/FFT/ifft scheme.
// We implement a forward operator T and its exact discrete inverse T^{-1}.
// This way, inverse( forward(matrix) ) ~= matrix (up to FP error).
// Author: Manuel Morgado (manuelmorgadov@gmail.com)
// reference: https://docs.nvidia.com/cuda/cuda-runtime-api/

// Compilation as: "%CUDA_PATH%\bin\nvcc" frft2d_benchmark.cu -I"%CUDA_PATH%\include" -L"%CUDA_PATH%\lib\x64" -lcufft -o frft2d_benchmark.exe

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Error checking

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void CUFFT_CHECK(cufftResult res, const char* file, int line)
{
    if (res != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT Error %d at %s:%d\n", (int)res, file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUFFT_CHECK_CALL(call) CUFFT_CHECK((call), __FILE__, __LINE__)

// Host-side complex sqrt

// sqrt(x + i y)
static inline cuFloatComplex complex_sqrt(float x, float y)
{
    float r = std::sqrt(x*x + y*y);
    float t = std::sqrt((r + x) * 0.5f);
    float u = (r - x) <= 0.0f ? 0.0f : std::sqrt((r - x) * 0.5f);
    if (y < 0.0f) u = -u;
    return make_cuFloatComplex(t, u);
}

// Device helpers
__device__ inline cuFloatComplex cexp_j(float phase)
{
    float c = cosf(phase);
    float s = sinf(phase);
    return make_cuFloatComplex(c, s);
}

// Kernels

// 2D modulator and filtor (time-domain) on a square grid NxN
// We use r^2 = (x - N/2)^2 + (y - N/2)^2, and then:
//   modulator = exp( j * (cotphi - cscphi) * pi * r^2 / N )
//   filtor    = exp( j * (cscphi)          * pi * r^2 / N )
__global__ void kernel_build_chirps_2d(cuFloatComplex* __restrict__ modulator,
                                       cuFloatComplex* __restrict__ filtor,
                                       int N, float cotphi, float cscphi)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    int idx = y * N + x;
    float dx = (float)x - (float)(N / 2);
    float dy = (float)y - (float)(N / 2);
    float r2_over_N = (dx*dx + dy*dy) / (float)N;

    float phase_mod = (cotphi - cscphi) * (float)M_PI * r2_over_N;
    float phase_flt = (cscphi)          * (float)M_PI * r2_over_N;

    modulator[idx] = cexp_j(phase_mod);
    filtor[idx]    = cexp_j(phase_flt);
}

// Pointwise: out = data * mod
__global__ void kernel_pointwise_mul_mod(const cuFloatComplex* __restrict__ data,
                                         const cuFloatComplex* __restrict__ mod,
                                         cuFloatComplex* __restrict__ out,
                                         int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;
    out[i] = cuCmulf(data[i], mod[i]);
}

// Pointwise: data *= filtor_fft
__global__ void kernel_pointwise_mul_inplace(cuFloatComplex* __restrict__ data,
                                             const cuFloatComplex* __restrict__ filtor_fft,
                                             int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;
    data[i] = cuCmulf(data[i], filtor_fft[i]);
}

// Pointwise: out = conjugate(in)
__global__ void kernel_conjugate(const cuFloatComplex* __restrict__ in,
                                 cuFloatComplex* __restrict__ out,
                                 int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;
    out[i] = make_cuFloatComplex(in[i].x, -in[i].y);
}

// Pointwise: out = 1 / in   (complex reciprocal)
__global__ void kernel_pointwise_inv(const cuFloatComplex* __restrict__ in,
                                     cuFloatComplex* __restrict__ out,
                                     int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;

    float a = in[i].x;
    float b = in[i].y;
    float denom = a*a + b*b;
    if (denom == 0.0f) {
        out[i] = make_cuFloatComplex(0.0f, 0.0f);
    } else {
        // 1 / (a + i b) = (a - i b) / (a^2 + b^2)
        out[i] = make_cuFloatComplex(a / denom, -b / denom);
    }
}

// Final scaling and modulation for forward transform:
// out = scale * (1 / (N^2)) * modulator * data
__global__ void kernel_final_scale_mod(const cuFloatComplex* __restrict__ data,
                                       const cuFloatComplex* __restrict__ mod,
                                       cuFloatComplex* __restrict__ out,
                                       cuFloatComplex scale,
                                       float inv_N2,
                                       int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;

    cuFloatComplex z = data[i];
    // apply inverse FFT scaling:
    z.x *= inv_N2;
    z.y *= inv_N2;

    // multiply by modulator
    cuFloatComplex zm = cuCmulf(z, mod[i]);

    // multiply by scale (complex)
    out[i] = cuCmulf(zm, scale);
}

// out = scale_inv * mod_inv * data
__global__ void kernel_final_scale_mod_inverse(const cuFloatComplex* __restrict__ data,
                                               const cuFloatComplex* __restrict__ mod_inv,
                                               cuFloatComplex* __restrict__ out,
                                               cuFloatComplex scale_inv,
                                               int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;

    cuFloatComplex z = cuCmulf(data[i], mod_inv[i]);
    out[i] = cuCmulf(z, scale_inv);
}

// 2D transform context

struct FrFT2DChirpContext {
    int N;
    float alpha;
    float cotphi;
    float cscphi;
    cuFloatComplex scale;      // forward scale
    cuFloatComplex scale_inv;  // inverse scale (1 / (scale * N^2))

    cufftHandle plan2d;   // cuFFT 2D plan

    cuFloatComplex* d_modulator;      // NxN
    cuFloatComplex* d_filtor_fft;     // NxN (FFT of filtor)
    cuFloatComplex* d_modulator_inv;  // conj(modulator)
    cuFloatComplex* d_filtor_fft_inv; // 1 / filtor_fft
    cuFloatComplex* d_data;           // NxN (input/output)
    cuFloatComplex* d_tmp;            // NxN (workspace)
};

void frft2d_chirp_context_init(FrFT2DChirpContext& ctx, int N, float alpha)
{
    ctx.N = N;
    ctx.alpha = alpha;

    // Parameters similar to the original continuous FrFT-style formula
    float phi    = alpha * (float)M_PI / 2.0f;
    float cotphi = 1.0f / tanf(phi);
    float cscphi = sqrtf(1.0f + cotphi * cotphi);
    ctx.cotphi   = cotphi;
    ctx.cscphi   = cscphi;

    // scale = sqrt(1 - i*cotphi) / sqrt(N^2)
    cuFloatComplex z = make_cuFloatComplex(1.0f, -cotphi);
    cuFloatComplex z_sqrt = complex_sqrt(z.x, z.y);
    float inv_sqrt_N2 = 1.0f / sqrtf((float)N * (float)N);
    ctx.scale = make_cuFloatComplex(z_sqrt.x * inv_sqrt_N2,
                                    z_sqrt.y * inv_sqrt_N2);

    // scale_inv = 1 / (scale * N^2)
    float N2f = (float)N * (float)N;
    float a = ctx.scale.x;
    float b = ctx.scale.y;
    float denom = a*a + b*b;
    if (denom == 0.0f) denom = 1.0f;

    // 1/scale = (a - i b) / (a^2 + b^2)
    cuFloatComplex scale_rec = make_cuFloatComplex(a / denom, -b / denom);

    // scale_inv = (1/scale) * (1/N^2)
    ctx.scale_inv = make_cuFloatComplex(scale_rec.x / N2f,
                                        scale_rec.y / N2f);

    size_t bytes = sizeof(cuFloatComplex) * N * N;

    CUDA_CHECK(cudaMalloc(&ctx.d_modulator,      bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_filtor_fft,     bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_modulator_inv,  bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_filtor_fft_inv, bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_data,           bytes));
    CUDA_CHECK(cudaMalloc(&ctx.d_tmp,            bytes));

    // Build modulator and filtor in time domain
    dim3 block2d(16, 16);
    dim3 grid2d((N + block2d.x - 1) / block2d.x,
                (N + block2d.y - 1) / block2d.y);

    kernel_build_chirps_2d<<<grid2d, block2d>>>(
        ctx.d_modulator,
        ctx.d_filtor_fft,  // temporarily filtor (time-domain)
        N, cotphi, cscphi
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create 2D FFT plan
    CUFFT_CHECK_CALL(cufftPlan2d(&ctx.plan2d, N, N, CUFFT_C2C));

    // FFT2 of filtor: filtor_fft = FFT2(filtor)
    CUFFT_CHECK_CALL(cufftExecC2C(ctx.plan2d,
                                  ctx.d_filtor_fft,
                                  ctx.d_filtor_fft,
                                  CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Precompute inverse arrays: conj(modulator), 1/filtor_fft
    int N2 = N * N;
    dim3 block1d(256);
    dim3 grid1d((N2 + block1d.x - 1) / block1d.x);

    kernel_conjugate<<<grid1d, block1d>>>(ctx.d_modulator,
                                          ctx.d_modulator_inv,
                                          N2);
    CUDA_CHECK(cudaGetLastError());

    kernel_pointwise_inv<<<grid1d, block1d>>>(ctx.d_filtor_fft,
                                              ctx.d_filtor_fft_inv,
                                              N2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void frft2d_chirp_context_destroy(FrFT2DChirpContext& ctx)
{
    cufftDestroy(ctx.plan2d);
    cudaFree(ctx.d_modulator);
    cudaFree(ctx.d_filtor_fft);
    cudaFree(ctx.d_modulator_inv);
    cudaFree(ctx.d_filtor_fft_inv);
    cudaFree(ctx.d_data);
    cudaFree(ctx.d_tmp);
}

// Execute forward 2D transform T:
// out = scale * modulator * ifft2( filtor_fft * fft2(modulator * in) ) / N^2
//
// h_in, h_out: host pointers to NxN cuFloatComplex
void frft2d_chirp_execute(const FrFT2DChirpContext& ctx,
                          const cuFloatComplex* h_in,
                          cuFloatComplex* h_out)
{
    int N = ctx.N;
    int N2 = N * N;
    size_t bytes = sizeof(cuFloatComplex) * N2;

    CUDA_CHECK(cudaMemcpy(ctx.d_data, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block1d(256);
    dim3 grid1d((N2 + block1d.x - 1) / block1d.x);

    // tmp = modulator * data
    kernel_pointwise_mul_mod<<<grid1d, block1d>>>(
        ctx.d_data,
        ctx.d_modulator,
        ctx.d_tmp,
        N2
    );
    CUDA_CHECK(cudaGetLastError());

    // FFT2(tmp)
    CUFFT_CHECK_CALL(cufftExecC2C(ctx.plan2d,
                                  ctx.d_tmp,
                                  ctx.d_tmp,
                                  CUFFT_FORWARD));

    // tmp *= filtor_fft
    kernel_pointwise_mul_inplace<<<grid1d, block1d>>>(
        ctx.d_tmp,
        ctx.d_filtor_fft,
        N2
    );
    CUDA_CHECK(cudaGetLastError());

    // IFFT2(tmp)
    CUFFT_CHECK_CALL(cufftExecC2C(ctx.plan2d,
                                  ctx.d_tmp,
                                  ctx.d_tmp,
                                  CUFFT_INVERSE));

    // out = scale * (1/N^2) * modulator * tmp
    float inv_N2 = 1.0f / ((float)N * (float)N);
    kernel_final_scale_mod<<<grid1d, block1d>>>(
        ctx.d_tmp,
        ctx.d_modulator,
        ctx.d_data,   // reuse d_data as output buffer
        ctx.scale,
        inv_N2,
        N2
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, ctx.d_data, bytes, cudaMemcpyDeviceToHost));
}

// Execute inverse transform T^{-1} (exact algebraic inverse of above)
void frft2d_chirp_inverse_execute(const FrFT2DChirpContext& ctx,
                                  const cuFloatComplex* h_in,
                                  cuFloatComplex* h_out)
{
    int N = ctx.N;
    int N2 = N * N;
    size_t bytes = sizeof(cuFloatComplex) * N2;

    CUDA_CHECK(cudaMemcpy(ctx.d_data, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block1d(256);
    dim3 grid1d((N2 + block1d.x - 1) / block1d.x);

    // y1 = M^{-1} * y
    kernel_pointwise_mul_mod<<<grid1d, block1d>>>(
        ctx.d_data,
        ctx.d_modulator_inv,
        ctx.d_tmp,
        N2
    );
    CUDA_CHECK(cudaGetLastError());

    // FFT2(y1)
    CUFFT_CHECK_CALL(cufftExecC2C(ctx.plan2d,
                                  ctx.d_tmp,
                                  ctx.d_tmp,
                                  CUFFT_FORWARD));

    // y2 *= D^{-1}
    kernel_pointwise_mul_inplace<<<grid1d, block1d>>>(
        ctx.d_tmp,
        ctx.d_filtor_fft_inv,
        N2
    );
    CUDA_CHECK(cudaGetLastError());

    // IFFT2(y2)
    CUFFT_CHECK_CALL(cufftExecC2C(ctx.plan2d,
                                  ctx.d_tmp,
                                  ctx.d_tmp,
                                  CUFFT_INVERSE));

    // out = scale_inv * M^{-1} * result
    kernel_final_scale_mod_inverse<<<grid1d, block1d>>>(
        ctx.d_tmp,
        ctx.d_modulator_inv,
        ctx.d_data,
        ctx.scale_inv,
        N2
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, ctx.d_data, bytes, cudaMemcpyDeviceToHost));
}

// Helper functions for benchmarking and plotting

// Compute RMS and max absolute error between two N x N complex matrices
void compute_error(const cuFloatComplex* a,
                   const cuFloatComplex* b,
                   int N,
                   double& rms_err,
                   double& max_err)
{
    int N2 = N * N;
    double sum_sq = 0.0;
    double max_e  = 0.0;

    for (int i = 0; i < N2; ++i) {
        float dr = a[i].x - b[i].x;
        float di = a[i].y - b[i].y;
        double e = std::sqrt((double)dr * dr + (double)di * di);
        sum_sq += e * e;
        if (e > max_e) max_e = e;
    }

    rms_err = std::sqrt(sum_sq / (double)N2);
    max_err = max_e;
}

// Save real part of an N x N complex matrix to CSV (for plotting)
void save_matrix_real_csv(const char* filename,
                          const cuFloatComplex* h,
                          int N)
{
    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::perror("fopen");
        return;
    }
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = y * N + x;
            float val = h[idx].x; // real part
            std::fprintf(f, "%g", val);
            if (x < N - 1) std::fprintf(f, ",");
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

// Save imaginary part of an N x N complex matrix to CSV (for plotting)
void save_matrix_imag_csv(const char* filename,
                          const cuFloatComplex* h,
                          int N)
{
    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::perror("fopen");
        return;
    }
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = y * N + x;
            float val = h[idx].y; // imaginary part
            std::fprintf(f, "%g", val);
            if (x < N - 1) std::fprintf(f, ",");
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

// main()

int main()
{
    const float alpha = 0.5f;
    const int repetitions = 10000;
    int sizes[] = {1024}; //64, 128, 256, 512, 1024, 5120
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < num_sizes; ++si) {
        int N = sizes[si];
        printf("\n=== N = %d, alpha = %f ===\n", N, alpha);

        size_t bytes = sizeof(cuFloatComplex) * N * N;
        cuFloatComplex* h_in  = (cuFloatComplex*)malloc(bytes);
        cuFloatComplex* h_out = (cuFloatComplex*)malloc(bytes);
        cuFloatComplex* h_rec = (cuFloatComplex*)malloc(bytes);

        // Fill input with something simple (real ramp)
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                float val = (float)(y * N + x);
                h_in[y * N + x] = make_cuFloatComplex(val, 0.0f);
            }
        }

        // Init context (plan, chirps, device buffers)
        FrFT2DChirpContext ctx;
        frft2d_chirp_context_init(ctx, N, alpha);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        frft2d_chirp_execute(ctx, h_in, h_out);

        float total_ms = 0.0f;
        for (int i = 0; i < repetitions; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            frft2d_chirp_execute(ctx, h_in, h_out);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }

        float mean_ms = total_ms / (float)repetitions;
        printf("Mean 2D FrFT (chirp/3xFFT2, compute-only) over %d runs: %.3f ms\n",
               repetitions, mean_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // For the LAST size only
        if (si == num_sizes - 1) {
            // forward + inverse once
            frft2d_chirp_execute(ctx, h_in, h_out);
            frft2d_chirp_inverse_execute(ctx, h_out, h_rec);

            double rms_err = 0.0, max_err = 0.0;
            compute_error(h_in, h_rec, N, rms_err, max_err);
            printf("Reconstruction error (last size):\n");
            printf("  RMS error = %.6e\n", rms_err);
            printf("  Max error = %.6e\n", max_err);

            // Save CSVs for real and imaginary parts
            save_matrix_real_csv("frft_input_real.csv",  h_in,  N);
            save_matrix_real_csv("frft_recon_real.csv",  h_rec, N);
            save_matrix_imag_csv("frft_input_imag.csv",  h_in,  N);
            save_matrix_imag_csv("frft_recon_imag.csv",  h_rec, N);

            printf("Saved CSVs:\n");
            printf("  frft_input_real.csv,  frft_recon_real.csv\n");
            printf("  frft_input_imag.csv,  frft_recon_imag.csv\n");
        }

        frft2d_chirp_context_destroy(ctx);
        free(h_in);
        free(h_out);
        free(h_rec);
    }

    return 0;
}
