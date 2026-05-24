#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iterator>
#include <print>
#include <string>
#include <string_view>

namespace {

[[noreturn]] void fail(cudaError_t error, std::string_view call) {
  std::println(stderr, "{} failed: {} ({})", call, cudaGetErrorString(error),
               static_cast<int>(error));
  std::exit(EXIT_FAILURE);
}

void check(cudaError_t error, std::string_view call) {
  if (error != cudaSuccess) {
    fail(error, call);
  }
}

template <typename Value>
void print_value(std::string_view name, const Value &value) {
  std::println("  {:<42} : {}", name, value);
}

template <typename Value, std::size_t Size>
void print_array(std::string_view name, const Value (&values)[Size]) {
  std::print("  {:<42} : [", name);
  for (std::size_t i = 0; i < Size; ++i) {
    if (i != 0) {
      std::print(", ");
    }
    std::print("{}", values[i]);
  }
  std::println("]");
}

std::string hex_bytes(const char *bytes, std::size_t size) {
  std::string out;
  for (std::size_t i = 0; i < size; ++i) {
    if (i != 0) {
      out.push_back(':');
    }
    std::format_to(
        std::back_inserter(out), "{:02x}",
        static_cast<unsigned int>(static_cast<unsigned char>(bytes[i])));
  }
  return out;
}

std::string cuda_version_string(int encoded_version) {
  const int major = encoded_version / 1000;
  const int minor = (encoded_version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor) + " (" +
         std::to_string(encoded_version) + ")";
}

void print_device_properties(int device, const cudaDeviceProp &prop) {
  std::println("\nDevice {}: {}", device, std::string_view(prop.name));
  std::println("{}", std::string(80, '-'));

  print_value("name", std::string_view(prop.name));
  print_value("uuid", hex_bytes(prop.uuid.bytes, sizeof(prop.uuid.bytes)));
  print_value("luid", hex_bytes(prop.luid, sizeof(prop.luid)));
  print_value("luidDeviceNodeMask", prop.luidDeviceNodeMask);
  print_value("totalGlobalMem", prop.totalGlobalMem);
  print_value("sharedMemPerBlock", prop.sharedMemPerBlock);
  print_value("regsPerBlock", prop.regsPerBlock);
  print_value("warpSize", prop.warpSize);
  print_value("memPitch", prop.memPitch);
  print_value("maxThreadsPerBlock", prop.maxThreadsPerBlock);
  print_array("maxThreadsDim", prop.maxThreadsDim);
  print_array("maxGridSize", prop.maxGridSize);
  print_value("totalConstMem", prop.totalConstMem);
  print_value("major", prop.major);
  print_value("minor", prop.minor);
  print_value("textureAlignment", prop.textureAlignment);
  print_value("texturePitchAlignment", prop.texturePitchAlignment);
  print_value("multiProcessorCount", prop.multiProcessorCount);
  print_value("integrated", prop.integrated);
  print_value("canMapHostMemory", prop.canMapHostMemory);
  print_value("maxTexture1D", prop.maxTexture1D);
  print_value("maxTexture1DMipmap", prop.maxTexture1DMipmap);
  print_array("maxTexture2D", prop.maxTexture2D);
  print_array("maxTexture2DMipmap", prop.maxTexture2DMipmap);
  print_array("maxTexture2DLinear", prop.maxTexture2DLinear);
  print_array("maxTexture2DGather", prop.maxTexture2DGather);
  print_array("maxTexture3D", prop.maxTexture3D);
  print_array("maxTexture3DAlt", prop.maxTexture3DAlt);
  print_value("maxTextureCubemap", prop.maxTextureCubemap);
  print_array("maxTexture1DLayered", prop.maxTexture1DLayered);
  print_array("maxTexture2DLayered", prop.maxTexture2DLayered);
  print_array("maxTextureCubemapLayered", prop.maxTextureCubemapLayered);
  print_value("maxSurface1D", prop.maxSurface1D);
  print_array("maxSurface2D", prop.maxSurface2D);
  print_array("maxSurface3D", prop.maxSurface3D);
  print_array("maxSurface1DLayered", prop.maxSurface1DLayered);
  print_array("maxSurface2DLayered", prop.maxSurface2DLayered);
  print_value("maxSurfaceCubemap", prop.maxSurfaceCubemap);
  print_array("maxSurfaceCubemapLayered", prop.maxSurfaceCubemapLayered);
  print_value("surfaceAlignment", prop.surfaceAlignment);
  print_value("concurrentKernels", prop.concurrentKernels);
  print_value("ECCEnabled", prop.ECCEnabled);
  print_value("pciBusID", prop.pciBusID);
  print_value("pciDeviceID", prop.pciDeviceID);
  print_value("pciDomainID", prop.pciDomainID);
  print_value("tccDriver", prop.tccDriver);
  print_value("asyncEngineCount", prop.asyncEngineCount);
  print_value("unifiedAddressing", prop.unifiedAddressing);
  print_value("memoryBusWidth", prop.memoryBusWidth);
  print_value("l2CacheSize", prop.l2CacheSize);
  print_value("persistingL2CacheMaxSize", prop.persistingL2CacheMaxSize);
  print_value("maxThreadsPerMultiProcessor", prop.maxThreadsPerMultiProcessor);
  print_value("streamPrioritiesSupported", prop.streamPrioritiesSupported);
  print_value("globalL1CacheSupported", prop.globalL1CacheSupported);
  print_value("localL1CacheSupported", prop.localL1CacheSupported);
  print_value("sharedMemPerMultiprocessor", prop.sharedMemPerMultiprocessor);
  print_value("regsPerMultiprocessor", prop.regsPerMultiprocessor);
  print_value("managedMemory", prop.managedMemory);
  print_value("isMultiGpuBoard", prop.isMultiGpuBoard);
  print_value("multiGpuBoardGroupID", prop.multiGpuBoardGroupID);
  print_value("hostNativeAtomicSupported", prop.hostNativeAtomicSupported);
  print_value("pageableMemoryAccess", prop.pageableMemoryAccess);
  print_value("concurrentManagedAccess", prop.concurrentManagedAccess);
  print_value("computePreemptionSupported", prop.computePreemptionSupported);
  print_value("canUseHostPointerForRegisteredMem",
              prop.canUseHostPointerForRegisteredMem);
  print_value("cooperativeLaunch", prop.cooperativeLaunch);
  print_value("sharedMemPerBlockOptin", prop.sharedMemPerBlockOptin);
  print_value("pageableMemoryAccessUsesHostPageTables",
              prop.pageableMemoryAccessUsesHostPageTables);
  print_value("directManagedMemAccessFromHost",
              prop.directManagedMemAccessFromHost);
  print_value("maxBlocksPerMultiProcessor", prop.maxBlocksPerMultiProcessor);
  print_value("accessPolicyMaxWindowSize", prop.accessPolicyMaxWindowSize);
  print_value("reservedSharedMemPerBlock", prop.reservedSharedMemPerBlock);
  print_value("hostRegisterSupported", prop.hostRegisterSupported);
  print_value("sparseCudaArraySupported", prop.sparseCudaArraySupported);
  print_value("hostRegisterReadOnlySupported",
              prop.hostRegisterReadOnlySupported);
  print_value("timelineSemaphoreInteropSupported",
              prop.timelineSemaphoreInteropSupported);
  print_value("memoryPoolsSupported", prop.memoryPoolsSupported);
  print_value("gpuDirectRDMASupported", prop.gpuDirectRDMASupported);
  print_value("gpuDirectRDMAFlushWritesOptions",
              prop.gpuDirectRDMAFlushWritesOptions);
  print_value("gpuDirectRDMAWritesOrdering", prop.gpuDirectRDMAWritesOrdering);
  print_value("memoryPoolSupportedHandleTypes",
              prop.memoryPoolSupportedHandleTypes);
  print_value("deferredMappingCudaArraySupported",
              prop.deferredMappingCudaArraySupported);
  print_value("ipcEventSupported", prop.ipcEventSupported);
  print_value("clusterLaunch", prop.clusterLaunch);
  print_value("unifiedFunctionPointers", prop.unifiedFunctionPointers);
  print_value("deviceNumaConfig", prop.deviceNumaConfig);
  print_value("deviceNumaId", prop.deviceNumaId);
  print_value("mpsEnabled", prop.mpsEnabled);
  print_value("hostNumaId", prop.hostNumaId);
  print_value("gpuPciDeviceID", prop.gpuPciDeviceID);
  print_value("gpuPciSubsystemID", prop.gpuPciSubsystemID);
  print_value("hostNumaMultinodeIpcSupported",
              prop.hostNumaMultinodeIpcSupported);
  print_array("reserved", prop.reserved);
}

} // namespace

int main() {
  int runtime_version = 0;
  int driver_version = 0;
  int device_count = 0;

  check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

  std::println("CUDA runtime version: {}",
               cuda_version_string(runtime_version));
  std::println("CUDA driver version : {}", cuda_version_string(driver_version));
  std::println("CUDA device count   : {}", device_count);

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    print_device_properties(device, prop);
  }

  return EXIT_SUCCESS;
}
