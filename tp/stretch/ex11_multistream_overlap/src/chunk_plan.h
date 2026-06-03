#pragma once

#include <cstddef>
#include <vector>

namespace tp {

struct Chunk {
  std::size_t begin;
  std::size_t end;
};

std::vector<Chunk> make_chunks(std::size_t size, std::size_t stream_count);

} // namespace tp
