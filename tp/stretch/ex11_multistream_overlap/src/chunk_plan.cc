#include "src/chunk_plan.h"

namespace tp {

std::vector<Chunk> make_chunks(std::size_t size, std::size_t stream_count) {
  // TODO: Return balanced contiguous chunks for the requested stream count.
  (void)stream_count;
  if (size == 0) {
    return {};
  }
  return {{0, size}};
}

} // namespace tp
