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

#include "src/packaging_metadata.h"

int main() {
  require(tp::image_repo_tag() == "tp-gpu-info:latest", "repo tag");
  require(tp::packaging_metadata_complete(), "metadata marked complete");
}
