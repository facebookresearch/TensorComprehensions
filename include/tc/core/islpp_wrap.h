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

#include "tc/core/utils/function_traits.h"

#include <functional>
#include <tuple>
#include <type_traits>

using tc::function_traits;

namespace isl {

//
// A bunch of wrappers with Frankenmonster-style lamda as to function pointers
// to pass C++ lambdas that capture to C functions (PPCG mostly where there is
// no C++ interface).
// This must die as we go to PPCGPP but is useful for bootstrapping.
//

template <typename T>
T null() {
  return T();
}

inline isl::union_pw_multi_aff /*isl_schedule_node_*/ get_subtree_contraction(
    isl::schedule_node node) {
  return isl::manage(isl_schedule_node_get_subtree_contraction(
      const_cast<isl_schedule_node*>(node.get())));
}

inline isl::union_set /*isl_union_set_*/ preimage_union_pw_multi_aff(
    isl::union_set uset,
    isl::union_pw_multi_aff upma) {
  return isl::manage(isl_union_set_preimage_union_pw_multi_aff(
      uset.release(), upma.release()));
}

inline isl::schedule_node /*isl_schedule_node_*/ _delete(
    isl::schedule_node node) {
  return isl::manage(isl_schedule_node_delete(node.release()));
}

namespace detail {
// Some uglyness to wrap isl functions calls and pass lambdas with captures.
// Could go C++17 and use std::apply function to tuple and avoid some
// indirections and manual integer unpacking with gens.
template <typename R, typename... Args>
std::function<R(Args...)> makeFunction(R (*fun)(Args... args)) {
  return [fun](Args... args) -> R { return fun(args...); };
}

template <size_t N, typename T1, typename T2>
struct tuple_equality_helper {
  constexpr static bool value = tuple_equality_helper<N - 1, T1, T2>::value &&
      std::is_same<typename std::tuple_element<N, T1>::type,
                   typename std::tuple_element<N, T2>::type>::value;
};

template <typename T1, typename T2>
struct tuple_equality_helper<0, T1, T2> {
  constexpr static bool value = true;
};

template <typename T1, typename T2>
struct tuple_equal {
  constexpr static bool value =
      (std::tuple_size<T1>::value == std::tuple_size<T2>::value) &&
      tuple_equality_helper<std::tuple_size<T1>::value - 1, T1, T2>::value;
};

template <int...>
struct seq {};

template <int N, int... S>
struct gens : gens<N - 1, N - 1, S...> {};

template <int... S>
struct gens<0, S...> {
  typedef seq<S...> type;
};

template <typename Lambda>
struct Wrap {
  Lambda& fptr;
  void* usr;
  Wrap(Lambda& f, void* u) : fptr(f), usr(u) {}
};

template <typename R, typename Fun, typename... Args, int... S>
R applyImpl(Fun fun, std::tuple<Args...> params, seq<S...>) {
  return fun(std::get<S>(params)...);
}

template <typename R, typename Fun, typename... Args>
R apply(Fun fun, std::tuple<Args...> params) {
  return applyImpl<R>(fun, params, typename gens<sizeof...(Args)>::type());
}

template <typename R, typename Fun, typename Tuple, typename... Args>
R callFun(Fun fun, Tuple t, Args... args) {
  return apply<R>(fun, std::tuple_cat(t, std::make_tuple(args...)));
}

} // namespace detail

// It is possible to avoid copy-pasted macros with more macro ugliness, in
// particular something like REPEAT from boostPP + concatentation.

#define AS_FUN_TYPE(name) name##AsFun

#define MAKE_LAMBDA_0_WRAPPER(typen, name)                               \
  using AS_FUN_TYPE(name) = function_traits<typen>;                      \
  auto name = [](void* usr) -> typename AS_FUN_TYPE(name)::result_type { \
    auto w = static_cast<detail::Wrap<typen>*>(usr);                     \
    return (w->fptr)(w->usr);                                            \
  };

#define MAKE_LAMBDA_1_WRAPPER(typen, name)                                  \
  using AS_FUN_TYPE(name) = function_traits<typen>;                         \
  auto name = [](                                                           \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0, void* usr) -> \
      typename AS_FUN_TYPE(name)::result_type {                             \
    auto w = static_cast<detail::Wrap<typen>*>(usr);                        \
    return (w->fptr)(arg0, w->usr);                                         \
  };

#define MAKE_LAMBDA_2_WRAPPER(typen, name)                    \
  using AS_FUN_TYPE(name) = function_traits<typen>;           \
  auto name = [](                                             \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0, \
      typename AS_FUN_TYPE(name)::template arg<1>::type arg1, \
      void* usr) -> typename AS_FUN_TYPE(name)::result_type { \
    auto w = static_cast<detail::Wrap<typen>*>(usr);          \
    return (w->fptr)(arg0, arg1, w->usr);                     \
  };

#define MAKE_LAMBDA_3_WRAPPER(typen, name)                    \
  using AS_FUN_TYPE(name) = function_traits<typen>;           \
  auto name = [](                                             \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0, \
      typename AS_FUN_TYPE(name)::template arg<1>::type arg1, \
      typename AS_FUN_TYPE(name)::template arg<2>::type arg2, \
      void* usr) -> typename AS_FUN_TYPE(name)::result_type { \
    auto w = static_cast<detail::Wrap<typen>*>(usr);          \
    return (w->fptr)(arg0, arg1, arg2, w->usr);               \
  };

#define MAKE_LAMBDA_4_WRAPPER(typen, name)                    \
  using AS_FUN_TYPE(name) = function_traits<typen>;           \
  auto name = [](                                             \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0, \
      typename AS_FUN_TYPE(name)::template arg<1>::type arg1, \
      typename AS_FUN_TYPE(name)::template arg<2>::type arg2, \
      typename AS_FUN_TYPE(name)::template arg<3>::type arg3, \
      void* usr) -> typename AS_FUN_TYPE(name)::result_type { \
    auto w = static_cast<detail::Wrap<typen>*>(usr);          \
    return (w->fptr)(arg0, arg1, arg2, arg3, w->usr);         \
  };

#define MAKE_LAMBDA_5_WRAPPER(typen, name)                    \
  using AS_FUN_TYPE(name) = function_traits<typen>;           \
  auto name = [](                                             \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0, \
      typename AS_FUN_TYPE(name)::template arg<1>::type arg1, \
      typename AS_FUN_TYPE(name)::template arg<2>::type arg2, \
      typename AS_FUN_TYPE(name)::template arg<3>::type arg3, \
      typename AS_FUN_TYPE(name)::template arg<4>::type arg4, \
      void* usr) -> typename AS_FUN_TYPE(name)::result_type { \
    auto w = static_cast<detail::Wrap<typen>*>(usr);          \
    return (w->fptr)(arg0, arg1, arg2, arg3, arg4, w->usr);   \
  };

#define MAKE_LAMBDA_6_WRAPPER(typen, name)                        \
  using AS_FUN_TYPE(name) = function_traits<typen>;               \
  auto name = [](                                                 \
      typename AS_FUN_TYPE(name)::template arg<0>::type arg0,     \
      typename AS_FUN_TYPE(name)::template arg<1>::type arg1,     \
      typename AS_FUN_TYPE(name)::template arg<2>::type arg2,     \
      typename AS_FUN_TYPE(name)::template arg<3>::type arg3,     \
      typename AS_FUN_TYPE(name)::template arg<4>::type arg4,     \
      typename AS_FUN_TYPE(name)::template arg<5>::type arg5,     \
      void* usr) -> typename AS_FUN_TYPE(name)::result_type {     \
    auto w = static_cast<detail::Wrap<typen>*>(usr);              \
    return (w->fptr)(arg0, arg1, arg2, arg3, arg4, arg5, w->usr); \
  };

// This struct template serves exclusively to check static assertions when
// instatiating isl_wrap_* function templates.
template <typename CFunType, typename Lambda, typename... ArgsCFun>
struct check_type_match {
  typedef typename function_traits<Lambda>::c_function_type lambda_type;
  static_assert(
      detail::tuple_equal<
          std::tuple<ArgsCFun..., lambda_type, void*>,
          typename function_traits<CFunType>::packed_args_type>::value,
      "Types of tuple arguments do not match the types of C function arguments");
  typedef typename function_traits<CFunType>::template arg<sizeof...(
      ArgsCFun)>::type callback_type;
  static_assert(
      detail::tuple_equal<
          typename function_traits<Lambda>::packed_args_type,
          typename function_traits<callback_type>::packed_args_type>::value,
      "Lambda argument types do not match the types of C callback arguments");
};

/////////////////////////////////////////////////////////////////////
// Helper methods to pass capturing lambdas into C functions that  //
// accept a C function pointer with last argument void*.           //
/////////////////////////////////////////////////////////////////////

// NOTE: the number N in isl_wrap_N corresponds to the number of
//       CALLBACK arguments WITHOUT the trailing void*.
//
// The functions that accept callbacks are expected to have a void* argument
// immediately after each function pointer argument.  This argument is supposed
// to be passed as the last argument of the callback function without change.
// We use this argument to pass a wrapper object that contains an actual lambda
// and the original void* pointer provided by the caller. We pass a simple
// non-capturing lambda that as C function pointer, which unwraps the wrapper
// object and calls the stored potentially capturing lambda.
//
// NOTE: the wrapper is created as a local stack-allocated variable of the
//       isl_wrap_N function, and a pointer to it is passed to the C function.
//       Therefore isl_wrap_N are NOT SUITABLE for DELAYED CALLBACKS, i.e.
//       callbacks that may be stored and called after isl_wrap_N returns.
//
// The tuple argument contains the leading arguments of the C function that are
// passed unmodified.  We enforce strict type matching to avoid hard-to-debug
// type conversion problems in template+macro code.  The caller of isl_wrap_N
// MUST explicitly convert all arguments to the types expected by the C
// function, even when implicit conversion is possible, including for void*
// arguments.

template <typename CFunType, typename... ArgsCFun, typename Lambda>
auto isl_wrap_0(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda f,
    void* user = nullptr) ->
    typename decltype(detail::makeFunction(cfun))::result_type {
  check_type_match<CFunType, Lambda, ArgsCFun...>();
  MAKE_LAMBDA_0_WRAPPER(Lambda, l);

  // Could go C++17 and use std::apply function to tuple and avoid indirection
  // and manual integer unpacking with gens
  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda> w(f, user);
  return detail::callFun<R>(cfun, argscfun, l, &w);
}

template <typename CFunType, typename... ArgsCFun, typename Lambda>
auto isl_wrap_1(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda f,
    void* user = nullptr) ->
    typename decltype(detail::makeFunction(cfun))::result_type {
  check_type_match<CFunType, Lambda, ArgsCFun...>();
  MAKE_LAMBDA_1_WRAPPER(Lambda, l);

  // Could go C++17 and use std::apply function to tuple and avoid indirection
  // and manual integer unpacking with gens
  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda> w(f, user);
  return detail::callFun<R>(cfun, argscfun, l, &w);
}

template <typename CFunType, typename... ArgsCFun, typename Lambda>
auto isl_wrap_2(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda f,
    void* user = nullptr) ->
    typename decltype(detail::makeFunction(cfun))::result_type {
  check_type_match<CFunType, Lambda, ArgsCFun...>();
  MAKE_LAMBDA_2_WRAPPER(Lambda, l);

  // Could go C++17 and use std::apply function to tuple and avoid indirection
  // and manual integer unpacking with gens
  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda> w(f, user);
  return detail::callFun<R>(cfun, argscfun, l, &w);
}

template <typename CFunType, typename... ArgsCFun, typename Lambda>
auto isl_wrap_3(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda f,
    void* user = nullptr) ->
    typename decltype(detail::makeFunction(cfun))::result_type {
  check_type_match<CFunType, Lambda, ArgsCFun...>();
  MAKE_LAMBDA_3_WRAPPER(Lambda, l);

  // Could go C++17 and use std::apply function to tuple and avoid indirection
  // and manual integer unpacking with gens
  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda> w(f, user);
  return detail::callFun<R>(cfun, argscfun, l, &w);
}

template <typename CFunType, typename... ArgsCFun, typename Lambda>
auto isl_wrap_4(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda f,
    void* user = nullptr) ->
    typename decltype(detail::makeFunction(cfun))::result_type {
  check_type_match<CFunType, Lambda, ArgsCFun...>();
  MAKE_LAMBDA_4_WRAPPER(Lambda, l);

  // Could go C++17 and use std::apply function to tuple and avoid indirection
  // and manual integer unpacking with gens
  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda> w(f, user);
  return detail::callFun<R>(cfun, argscfun, l, &w);
}

template <
    typename CFunType,
    typename... ArgsCFun,
    typename Lambda1,
    typename Lambda2,
    typename Lambda3>
auto isl_wrap_6_5_4(
    CFunType cfun,
    std::tuple<ArgsCFun...> argscfun,
    Lambda1 f1,
    void* user1,
    Lambda2 f2,
    void* user2,
    Lambda3 f3,
    void* user3) -> typename decltype(detail::makeFunction(cfun))::result_type {
  typedef typename function_traits<Lambda1>::c_function_type lambda1_type;
  typedef typename function_traits<Lambda2>::c_function_type lambda2_type;
  typedef typename function_traits<Lambda3>::c_function_type lambda3_type;
  static_assert(
      detail::tuple_equal<
          std::tuple<
              ArgsCFun...,
              lambda1_type,
              void*,
              lambda2_type,
              void*,
              lambda3_type,
              void*>,
          typename function_traits<CFunType>::packed_args_type>::value,
      "Types of tuple arguments do not match the types of C function arguments");

  typedef typename function_traits<CFunType>::template arg<sizeof...(
      ArgsCFun)>::type callback1_type;
  static_assert(
      detail::tuple_equal<
          typename function_traits<Lambda1>::packed_args_type,
          typename function_traits<callback1_type>::packed_args_type>::value,
      "Lambda1 argument types do not match the types of C function arguments");
  typedef typename function_traits<CFunType>::template arg<
      sizeof...(ArgsCFun) + 2>::type callback2_type;
  static_assert(
      detail::tuple_equal<
          typename function_traits<Lambda2>::packed_args_type,
          typename function_traits<callback2_type>::packed_args_type>::value,
      "Lambda2 argument types do not match the types of C function arguments");
  typedef typename function_traits<CFunType>::template arg<
      sizeof...(ArgsCFun) + 4>::type callback3_type;
  static_assert(
      detail::tuple_equal<
          typename function_traits<Lambda3>::packed_args_type,
          typename function_traits<callback3_type>::packed_args_type>::value,
      "Lambda3 argument types do not match the types of C function arguments");

  MAKE_LAMBDA_6_WRAPPER(Lambda1, l1);
  MAKE_LAMBDA_5_WRAPPER(Lambda2, l2);
  MAKE_LAMBDA_4_WRAPPER(Lambda3, l3);

  using R = typename decltype(detail::makeFunction(cfun))::result_type;
  detail::Wrap<Lambda1> w1(f1, user1);
  detail::Wrap<Lambda2> w2(f2, user2);
  detail::Wrap<Lambda3> w3(f3, user3);
  return detail::callFun<R>(cfun, argscfun, l1, &w1, l2, &w2, l3, &w3);
}

#undef AS_FUN_TYPE
#undef MAKE_LAMBDA_0_WRAPPER
#undef MAKE_LAMBDA_1_WRAPPER
#undef MAKE_LAMBDA_2_WRAPPER
#undef MAKE_LAMBDA_3_WRAPPER
#undef MAKE_LAMBDA_4_WRAPPER
#undef MAKE_LAMBDA_5_WRAPPER
#undef MAKE_LAMBDA_6_WRAPPER
} // namespace isl
