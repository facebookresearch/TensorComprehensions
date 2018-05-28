/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <chrono>

namespace tc {
struct Duration {
 private:
  Duration() : val_(0) {}
  explicit Duration(size_t us) : val_(us) {}
  explicit Duration(std::chrono::microseconds us) : val_(us) {}

 public:
  Duration(const Duration& d) : val_(d.val_) {}

  static inline Duration since(
      std::chrono::time_point<std::chrono::system_clock> begin) {
    auto end = std::chrono::system_clock::now();
    return Duration(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin));
  }
  static inline Duration fromMicroSeconds(size_t us) {
    return Duration(us);
  }
  inline size_t toMicroSeconds() {
    return val_.count();
  }
  static inline Duration zero() {
    return Duration(0);
  }
  static inline Duration max() {
    return Duration(std::chrono::microseconds::max());
  }

  friend inline Duration operator-(Duration lhs, const Duration& rhs) {
    lhs.val_ -= rhs.val_;
    return lhs;
  }
  friend inline Duration operator+(Duration lhs, const Duration& rhs) {
    lhs.val_ += rhs.val_;
    return lhs;
  }
  friend inline Duration operator/(Duration lhs, uint32_t rhs) {
    return Duration(lhs.val_ / rhs);
  }
  friend inline Duration operator*(Duration lhs, uint32_t rhs) {
    return Duration(lhs.val_ * rhs);
  }
  friend inline bool operator>=(const Duration& lhs, const Duration& rhs) {
    return lhs.val_ >= rhs.val_;
  }
  friend inline bool operator<(const Duration& lhs, const Duration& rhs) {
    return lhs.val_ < rhs.val_;
  }
  friend inline bool operator==(const Duration& lhs, const Duration& rhs) {
    return lhs.val_ == rhs.val_;
  }

  friend inline bool operator!=(const Duration& lhs, const Duration& rhs) {
    return lhs.val_ != rhs.val_;
  }

 private:
  std::chrono::microseconds val_;
};

struct ProfilingInfo {
  Duration cpuOverhead;
  Duration kernelRuntime;
};
} // namespace tc
