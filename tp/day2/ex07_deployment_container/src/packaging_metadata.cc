#include "src/packaging_metadata.h"

namespace tp {

std::string_view image_repo_tag() {
  // TODO: Keep this value aligned with the oci_load repo_tags entry.
  return "todo-gpu-info:latest";
}

bool packaging_metadata_complete() {
  // TODO: Return true once the binary, layer, image, and load target metadata are final.
  return false;
}

} // namespace tp
