#include "src/chunk_plan.h"

#include <algorithm>

namespace tp {

std::vector<Chunk> make_chunks(std::size_t size, std::size_t stream_count) {
  if (size == 0) {
    return {};
  }
  const std::size_t streams = std::max<std::size_t>(1, std::min(stream_count, size));
  const std::size_t chunk_size = (size + streams - 1) / streams;
  std::vector<Chunk> chunks;
  for (std::size_t begin = 0; begin < size; begin += chunk_size) {
    chunks.push_back({begin, std::min(begin + chunk_size, size)});
  }
  return chunks;
}

} // namespace tp
