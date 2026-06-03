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
  require(chunks[2].begin == 6 && chunks[2].end == 9, "chunk 2");
  require(chunks[3].begin == 9 && chunks[3].end == 10, "chunk 3");
}
