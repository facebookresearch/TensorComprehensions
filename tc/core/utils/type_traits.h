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
#include <type_traits>

namespace tc {
// WG21 N3911 2.3 Implementation workaround
// (Make sure template argument is always used.)
template <class...>
struct voider {
  using type = void;
};
template <class... T0toN>
using void_t = typename voider<T0toN...>::type;

template <typename T, typename = void>
struct is_std_container : std::false_type {};

template <typename T>
struct is_std_container<
    T,
    void_t<
        decltype(std::declval<T&>().begin()),
        decltype(std::declval<T&>().end()),
        typename T::value_type>> : std::true_type {};
} // namespace tc
