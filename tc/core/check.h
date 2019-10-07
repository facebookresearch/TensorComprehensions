// Copyright (c) 2017-present, Facebook, Inc.
// #
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// #
//     http://www.apache.org/licenses/LICENSE-2.0
// #
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ##############################################################################
#pragma once

#include <cstdint>
#include <iosfwd>
#include <sstream>
#include <stdexcept>
#include <string>

#include "tc/core/utils/type_traits.h"

/*
 * Each TC_CHECK(_*) macro checks for a condition and throws an exception if
 * the condition does not hold
 *
 *
 *Additional information can be passed through operator<< and is
 *included in the exception's error message, e.g.:
 *TC_CHECK_EQ(x, 42) << "x is not the answer";
 *
 *
 * The message in the throw exception is:
 * Check failed [filename:line_number] error(: info)
 *
 * filename: the name of the file where TC_CHECK(_*) was used
 * lineno:   the number of the line in which TC_CHECK(_*) was used
 * error:    this shows what failed, (e.g, "1 is not equal to 42")
 * info:     if operator<< was called then the information passed to it is info
 *
 *
 *
 * WARNING/CORNER CASE:
 *
 * Checker's destructor throws. This means that if another exception is thrown
 * before a fully constructed Checker object is destroyed then the program will
 * std::terminate. here is one unavoidable corner case:
 *
 * TC_CHECK(foo) << bar;
 *
 * If bar is a function/constructor call and throws then the program will
 * std::terminate (because when Checker's destructor runs it will throw a
 * second exception).
 *
 *
 * Exception type:
 * The default exception type is std::runtime_error, a different type can be
 * specified by passing an extra argument, e.g.:
 * TC_CHECK(i, whatever.size(), std::out_of_range);
 *
 *
 * List of check macros:
 * TC_CHECK(condition) //checks if condition is true
 * TC_CHECK_EQ(x,y)    //checks if x == y
 * TC_CHECK_NE(x,y)    //checks if x != y
 * TC_CHECK_LT(x,y)    //checks if x < y
 * TC_CHECK_GT(x,y)    //checks if x > y
 * TC_CHECK_LE(x,y)    //checks if x <= y
 * TC_CHECK_GE(x,y)    //checks if x >= y
 */

// condition should either be a bool or convertible to bool
#define TC_CHECK_IMPL(condition, exception_type) \
  tc::detail::tc_check<exception_type>(          \
      static_cast<bool>(condition), __FILE__, __LINE__)
// checks if x == y
#define TC_CHECK_EQ_IMPL(x, y, exception_type) \
  tc::detail::tc_check_eq<exception_type>(x, y, __FILE__, __LINE__)
// checks if x != y
#define TC_CHECK_NE_IMPL(x, y, exception_type) \
  tc::detail::tc_check_ne<exception_type>(x, y, __FILE__, __LINE__)
// checks if x < y
#define TC_CHECK_LT_IMPL(x, y, exception_type) \
  tc::detail::tc_check_lt<exception_type>(x, y, __FILE__, __LINE__)
// checks if x > y
#define TC_CHECK_GT_IMPL(x, y, exception_type) \
  tc::detail::tc_check_gt<exception_type>(x, y, __FILE__, __LINE__)
// checks if x <= y
#define TC_CHECK_LE_IMPL(x, y, exception_type) \
  tc::detail::tc_check_le<exception_type>(x, y, __FILE__, __LINE__)
// checks if x >= y
#define TC_CHECK_GE_IMPL(x, y, exception_type) \
  tc::detail::tc_check_ge<exception_type>(x, y, __FILE__, __LINE__)

#define TC_CHECK_DEFAULT(condition) TC_CHECK_IMPL(condition, std::runtime_error)
#define TC_CHECK_EQ_DEFAULT(x, y, ...) \
  TC_CHECK_EQ_IMPL(x, y, std::runtime_error)
#define TC_CHECK_NE_DEFAULT(x, y, ...) \
  TC_CHECK_NE_IMPL(x, y, std::runtime_error)
#define TC_CHECK_LT_DEFAULT(x, y, ...) \
  TC_CHECK_LT_IMPL(x, y, std::runtime_error)
#define TC_CHECK_GT_DEFAULT(x, y, ...) \
  TC_CHECK_GT_IMPL(x, y, std::runtime_error)
#define TC_CHECK_LE_DEFAULT(x, y, ...) \
  TC_CHECK_LE_IMPL(x, y, std::runtime_error)
#define TC_CHECK_GE_DEFAULT(x, y, ...) \
  TC_CHECK_GE_IMPL(x, y, std::runtime_error)

#define TC_GET_MACRO12(_1, _2, NAME, ...) NAME
#define TC_GET_MACRO23(_1, _2, _3, NAME, ...) NAME

#define TC_CHECK(...)                                          \
  TC_GET_MACRO12(__VA_ARGS__, TC_CHECK_IMPL, TC_CHECK_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_EQ(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_EQ_IMPL, TC_CHECK_EQ_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_NE(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_NE_IMPL, TC_CHECK_NE_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_LT(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_LT_IMPL, TC_CHECK_LT_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_GT(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_GT_IMPL, TC_CHECK_GT_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_LE(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_LE_IMPL, TC_CHECK_LE_DEFAULT) \
  (__VA_ARGS__)

#define TC_CHECK_GE(...)                                             \
  TC_GET_MACRO23(__VA_ARGS__, TC_CHECK_GE_IMPL, TC_CHECK_GE_DEFAULT) \
  (__VA_ARGS__)

namespace tc {

namespace detail {
template <typename ExceptionType>
class Checker {
 public:
  Checker(bool condition, std::string location, std::string baseErrorMsg)
      : condition_(condition),
        location_(location),
        baseErrorMsg_(baseErrorMsg){};
  ~Checker() noexcept(false) {
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
    throw ExceptionType(ss.str());
  }

  template <typename T>
  typename std::enable_if<!tc::is_std_container<T>::value, Checker&>::type
  operator<<(const T& msg) {
    try {
      std::stringstream ss;
      ss << additionalMsg_ << msg;
      additionalMsg_ = ss.str();
    } catch (...) {
      // If the above throws and we don't catch the exception then the
      // destructor will throw a second one and the program will terminate.
    }
    return *this;
  }

  template <typename C>
  typename std::enable_if<tc::is_std_container<C>::value, Checker&>::type
  operator<<(const C& msg) {
    try {
      std::stringstream ss;
      ss << additionalMsg_;
      for (const auto& x : msg) {
        ss << x << ',';
      }
      additionalMsg_ = ss.str();
      if (msg.begin() != msg.end()) {
        additionalMsg_.pop_back();
      }
    } catch (...) {
      // If the above throws and we don't catch the exception then the
      // destructor will throw a second one and the program will terminate.
    }
    return *this;
  }

 private:
  bool condition_;
  std::string location_;
  std::string baseErrorMsg_;
  std::string additionalMsg_;
}; // namespace detail

inline std::string makeLocation(const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << filename << ':' << lineno;
  return ss.str();
}

template <typename ExceptionType>
Checker<ExceptionType>
tc_check(bool condition, const char* filename, uint64_t lineno) {
  return Checker<ExceptionType>(condition, makeLocation(filename, lineno), {});
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_eq(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " not equal to " << y;
  return Checker<ExceptionType>(
      x == y, makeLocation(filename, lineno), ss.str());
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_ne(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " equal to " << y;
  return Checker<ExceptionType>(
      x != y, makeLocation(filename, lineno), ss.str());
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_lt(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " not less than " << y;
  return Checker<ExceptionType>(
      x < y, makeLocation(filename, lineno), ss.str());
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_gt(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " not greater than " << y;
  return Checker<ExceptionType>(
      x > y, makeLocation(filename, lineno), ss.str());
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_le(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " not less than or equal to " << y;
  return Checker<ExceptionType>(
      x <= y, makeLocation(filename, lineno), ss.str());
}

template <typename ExceptionType, typename X, typename Y>
Checker<ExceptionType>
tc_check_ge(const X& x, const Y& y, const char* filename, uint64_t lineno) {
  std::stringstream ss;
  ss << x << " not greater than or equal to " << y;
  return Checker<ExceptionType>(
      x >= y, makeLocation(filename, lineno), ss.str());
}

} // namespace detail
} // namespace tc
