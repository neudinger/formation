#include "src/chunk_plan.h"

#include <algorithm>

namespace tp {

std::vector<Chunk> make_chunks(std::size_t size, std::size_t stream_count) {
  if (size == 0) {
    return {};
  }
  const std::size_t streams = std::max<std::size_t>(1, std::min(stream_count, size));
  std::vector<Chunk> chunks;
  std::size_t begin = 0;
  for (std::size_t i = 0; i < streams; ++i) {
    const auto count = size / streams + (i < size % streams ? 1 : 0);
    chunks.push_back({begin, begin + count});
    begin += count;
  }
  return chunks;
}

} // namespace tp
