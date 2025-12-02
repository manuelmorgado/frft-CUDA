import os
import ctypes
import numpy as np

# Load library
if os.name == "nt":
    libname = "C:\\Users\\manue\\Documents\\20251125_frft_cuda\\frft2d.dll"
else:
    libname = "./libfrft2d.so"

lib = ctypes.cdll.LoadLibrary(libname)

FrFT2DContextPtr = ctypes.c_void_p
ComplexPtr = ctypes.c_void_p  # np.complex64 buffers as void

# C signatures/types
lib.frft2d_create_context.argtypes = [ctypes.c_int, ctypes.c_float]
lib.frft2d_create_context.restype  = FrFT2DContextPtr

lib.frft2d_destroy_context.argtypes = [FrFT2DContextPtr]
lib.frft2d_destroy_context.restype  = None

lib.frft2d_forward.argtypes = [FrFT2DContextPtr, ComplexPtr, ComplexPtr]
lib.frft2d_forward.restype  = None

lib.frft2d_inverse.argtypes = [FrFT2DContextPtr, ComplexPtr, ComplexPtr]
lib.frft2d_inverse.restype  = None


class FrFT2D:
    def __init__(self, N: int, alpha: float):

        self.N = int(N)
        self.alpha = float(alpha)

        ctx = lib.frft2d_create_context(self.N, ctypes.c_float(self.alpha))
        
        if not ctx:
            raise RuntimeError("frft2d_create_context returned NULL")
        self._ctx = ctx

    def __del__(self):

        try:
            if getattr(self, "_ctx", None):
                lib.frft2d_destroy_context(self._ctx)
                self._ctx = None

        except Exception:
            pass

    def forward(self, arr: np.ndarray) -> np.ndarray:

        if arr.shape != (self.N, self.N):
            raise ValueError(f"expected shape {(self.N, self.N)}, got {arr.shape}")

        if arr.dtype != np.complex64:
            arr = arr.astype(np.complex64, copy=False)

        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        out = np.empty_like(arr)

        in_ptr  = arr.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)
        lib.frft2d_forward(self._ctx, in_ptr, out_ptr)

        return out

    def inverse(self, arr: np.ndarray) -> np.ndarray:

        if arr.shape != (self.N, self.N):
            raise ValueError(f"expected shape {(self.N, self.N)}, got {arr.shape}")

        if arr.dtype != np.complex64:
            arr = arr.astype(np.complex64, copy=False)

        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        out = np.empty_like(arr)
        in_ptr  = arr.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)
        lib.frft2d_inverse(self._ctx, in_ptr, out_ptr)

        return out
