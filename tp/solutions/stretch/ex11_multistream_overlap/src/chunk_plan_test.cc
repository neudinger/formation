#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "test failure: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

#include "src/chunk_plan.h"

int main() {
  const auto chunks = tp::make_chunks(10, 4);
  require(chunks.size() == 4, "four chunks");
  require(chunks[0].begin == 0 && chunks[0].end == 3, "chunk 0");
  require(chunks[1].begin == 3 && chunks[1].end == 6, "chunk 1");
  require(chunks[2].begin == 6 && chunks[2].end == 8, "chunk 2");
  require(chunks[3].begin == 8 && chunks[3].end == 10, "chunk 3");
  require(tp::make_chunks(0, 4).empty(), "empty input");
  for (const std::size_t size : {1, 3, 5, 10, 1003}) {
    for (const std::size_t streams : {0, 1, 2, 4, 8}) {
      const auto plan = tp::make_chunks(size, streams);
      const auto count = streams == 0 ? 1 : (streams < size ? streams : size);
      require(plan.size() == count, "requested number of nonempty chunks");
      std::size_t end = 0;
      for (const auto chunk : plan) {
        require(chunk.begin == end && chunk.end > chunk.begin, "contiguous chunks");
        const auto length = chunk.end - chunk.begin;
        require(length == size / count || length == size / count + 1, "balanced chunks");
        end = chunk.end;
      }
      require(end == size, "cover entire input");
    }
  }
}
