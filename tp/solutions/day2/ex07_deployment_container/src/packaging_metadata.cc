#include "src/packaging_metadata.h"

namespace tp {

std::string_view image_repo_tag() {
  return "tp-gpu-info:latest";
}

bool packaging_metadata_complete() {
  return true;
}

} // namespace tp
