#include "tc/core/check.h"

#include <stdexcept>

namespace tc {
namespace detail {

Checker::Checker(bool condition, std::string location, std::string baseErrorMsg)
    : condition_(condition), location_(location), baseErrorMsg_(baseErrorMsg){};

Checker::~Checker() noexcept(false) {
  if (condition_) {
    return;
  }
  std::stringstream ss;
  ss << "Check failed [" << location_ << ']';

  if (not baseErrorMsg_.empty()) {
    ss << ' ' << baseErrorMsg_;
  }

  if (not additionalMsg_.empty()) {
    ss << ": " << additionalMsg_;
  }
  throw std::runtime_error(ss.str());
}

Checker tc_check(bool condition, const char* filename, uint64_t lineno) {
  return Checker(condition, makeLocation(filename, lineno), {});
}

std::string makeLocation(const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << filename << ':' << lineno;
  return ss.str();
}

} // namespace detail
} // namespace tc
