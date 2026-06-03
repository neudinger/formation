#include "src/matmul_cublas.h"

#include "src/cuda_runtime_helpers.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <optional>
#include <vector>

namespace tp::la {
namespace {

typedef struct cublasContext *cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0 };
using CreateFn = cublasStatus_t (*)(cublasHandle_t *);
using DestroyFn = cublasStatus_t (*)(cublasHandle_t);
using SgemmFn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                   int, int, int, const float *, const float *, int,
                                   const float *, int, const float *, float *, int);

class DynamicLibrary {
public:
  DynamicLibrary() {
#if defined(_WIN32)
    handle_ = LoadLibraryA("cublas64_12.dll");
#else
    handle_ = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      handle_ = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
    }
#endif
  }
  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }
  [[nodiscard]] bool available() const { return handle_ != nullptr; }
  template <typename Fn> Fn symbol(const char *name) const {
#if defined(_WIN32)
    return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    return reinterpret_cast<Fn>(dlsym(handle_, name));
#endif
  }
private:
#if defined(_WIN32)
  HMODULE handle_ = nullptr;
#else
  void *handle_ = nullptr;
#endif
};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      TP_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }
  ~DeviceBuffer() { (void)cudaFree(ptr_); }
  [[nodiscard]] T *get() const { return ptr_; }
  [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }
private:
  T *ptr_ = nullptr;
  std::size_t count_ = 0;
};

} // namespace

std::string_view cublas_strategy_name() { return "dynamic-cublas"; }

std::optional<std::vector<float>> try_matmul_cublas(const std::vector<float> &a,
                                                    const std::vector<float> &b,
                                                    const std::vector<float> &c,
                                                    int m, int n, int k,
                                                    float alpha, float beta) {
  DynamicLibrary lib;
  if (!lib.available()) {
    return std::nullopt;
  }
  const auto create = lib.symbol<CreateFn>("cublasCreate_v2");
  const auto destroy = lib.symbol<DestroyFn>("cublasDestroy_v2");
  const auto sgemm = lib.symbol<SgemmFn>("cublasSgemm_v2");
  if (create == nullptr || destroy == nullptr || sgemm == nullptr) {
    return std::nullopt;
  }
  std::vector<float> out(static_cast<std::size_t>(m * n));
  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_out(out.size());
  TP_CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), d_a.bytes(), cudaMemcpyHostToDevice));
  TP_CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), d_b.bytes(), cudaMemcpyHostToDevice));
  TP_CUDA_CHECK(cudaMemcpy(d_out.get(), c.data(), d_out.bytes(), cudaMemcpyHostToDevice));
  cublasHandle_t handle = nullptr;
  if (create(&handle) != CUBLAS_STATUS_SUCCESS) {
    return std::nullopt;
  }
  const cublasStatus_t status = sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                      &alpha, d_b.get(), n, d_a.get(), k,
                                      &beta, d_out.get(), n);
  destroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return std::nullopt;
  }
  TP_CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
  return out;
}

} // namespace tp::la
