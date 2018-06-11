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
