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

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>

#include <isl/cpp.h>

#include "tc/core/islpp_wrap.h"

namespace isl {

//
// A bunch of isl utility functions and helpers that have not yet graduated to
// the official ISL C++ bindings.
//

template <typename T>
inline T operator-(T a, T b) {
  return a.sub(b);
}

inline isl::val operator*(isl::val l, isl::val r) {
  return l.mul(r);
}

inline isl::val operator*(isl::val v, long i) {
  return v.mul(isl::val(v.get_ctx(), i));
}

inline isl::val operator*(long i, isl::val v) {
  return v * i;
}

inline isl::val operator+(isl::val l, isl::val r) {
  return l.add(r);
}

inline isl::val operator+(isl::val v, long i) {
  return v.add(isl::val(v.get_ctx(), i));
}

inline isl::val operator+(long i, isl::val v) {
  return v + i;
}

inline isl::val operator-(isl::val v, long i) {
  return v.sub(isl::val(v.get_ctx(), i));
}

inline isl::val operator-(long i, isl::val v) {
  return isl::val(v.get_ctx(), i).sub(v);
}

inline bool operator<(isl::val l, isl::val r) {
  return l.lt(r);
}

inline bool operator<=(isl::val l, isl::val r) {
  return l.le(r);
}

inline bool operator>(isl::val l, isl::val r) {
  return l.gt(r);
}

inline bool operator>=(isl::val l, isl::val r) {
  return l.ge(r);
}

inline bool operator==(isl::val v, long i) {
  return v.eq(isl::val(v.get_ctx(), i));
}

inline bool operator==(long i, isl::val v) {
  return v == i;
}

inline bool operator==(isl::val v1, isl::val v2) {
  return v1.eq(v2);
}

inline bool operator!=(isl::val v, long i) {
  return !(v == i);
}

inline bool operator!=(long i, isl::val v) {
  return !(v == i);
}

inline bool operator!=(isl::val v1, isl::val v2) {
  return !(v1 == v2);
}

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::aff to perform arithmetic and create/combine with sets
///////////////////////////////////////////////////////////////////////////////
isl::aff operator*(int i, isl::aff A);
isl::aff operator*(isl::aff A, int i);
isl::aff operator*(isl::aff A, isl::val V);
isl::aff operator*(isl::val V, isl::aff A);

isl::aff operator/(isl::aff A, int i);

isl::aff operator+(int i, isl::aff A);
isl::aff operator+(isl::aff A, isl::aff B);
isl::aff operator+(isl::aff A, int i);
isl::aff operator+(isl::aff A, isl::val v);
isl::aff operator+(isl::val v, isl::aff A);

isl::aff operator-(isl::aff A, int i);
isl::aff operator-(int i, isl::aff A);

// Thin wrapper around aff to disambiguate types for operators and avoid case
// where return type overloading occurs
struct aff_set {
  isl::aff aff;
  aff_set(isl::aff a) : aff(a) {}
};

isl::set operator>=(isl::aff_set A, int i);
isl::set operator>=(isl::aff_set A, isl::val v);
isl::set operator>=(int i, isl::aff_set A);
isl::set operator>=(isl::aff_set A, isl::aff B);
isl::set operator>=(isl::aff A, isl::aff_set B);
isl::set operator>(isl::aff_set A, int i);
isl::set operator>(int i, isl::aff_set A);
isl::set operator>(isl::aff_set A, isl::aff B);
isl::set operator>(isl::aff A, isl::aff_set B);

isl::set operator<=(isl::aff_set A, int i);
isl::set operator<=(isl::aff_set A, isl::val v);
isl::set operator<=(int i, isl::aff_set A);
isl::set operator<=(isl::aff_set A, isl::aff B);
isl::set operator<=(isl::aff_set A, isl::aff_set B);
isl::set operator<(isl::aff_set A, int i);
isl::set operator<(isl::aff_set A, isl::val v);
isl::set operator<(int i, isl::aff_set A);
isl::set operator<(isl::aff_set A, isl::aff B);
isl::set operator<(isl::aff_set A, isl::aff_set B);

isl::set operator==(isl::aff_set A, int i);
isl::set operator==(int i, isl::aff_set A);
isl::set operator==(isl::aff_set A, isl::aff B);
isl::set operator==(isl::aff A, isl::aff_set B);

// Thin wrapper around aff to disambiguate types for operators and avoid case
// where return type overloading occurs
struct aff_map {
  isl::aff aff;
  aff_map(isl::aff a) : aff(a) {}
};

isl::map operator>=(isl::aff_map A, isl::aff B);
isl::map operator<=(isl::aff_map A, isl::aff B);
isl::map operator>(isl::aff_map A, isl::aff B);
isl::map operator<(isl::aff_map A, isl::aff B);

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::union_set
///////////////////////////////////////////////////////////////////////////////
isl::set operator&(isl::set S1, isl::set S2);
isl::union_set operator&(isl::union_set S1, isl::set S2);
isl::union_set operator&(isl::set S1, isl::union_set S2);
isl::union_set operator&(isl::union_set S1, isl::union_set S2);

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::point
///////////////////////////////////////////////////////////////////////////////
isl::set operator&(isl::set S1, isl::point P2);
isl::set operator&(isl::point P1, isl::set S2);

inline isl::map operator&(isl::map m1, isl::map m2) {
  return m1.intersect(m2);
}

inline isl::map operator|(isl::map m1, isl::map m2) {
  return m1.unite(m2);
}

namespace detail {

inline isl::set extractFromUnion(isl::union_set uset, isl::space dim) {
  return uset.extract_set(dim);
}

inline isl::map extractFromUnion(isl::union_map umap, isl::space dim) {
  return umap.extract_map(dim);
}

inline int nElement(isl::union_set uset) {
  return uset.n_set();
}

inline int nElement(isl::union_map umap) {
  return umap.n_map();
}

inline void foreachElement(
    const std::function<void(isl::set)>& fun,
    isl::union_set uset) {
  uset.foreach_set(fun);
}

inline void foreachElement(
    const std::function<void(isl::map)>& fun,
    isl::union_map umap) {
  umap.foreach_map(fun);
}

inline isl::union_set addElement(isl::union_set uset, isl::set set) {
  return uset.add_set(set);
}

inline isl::union_map addElement(isl::union_map umap, isl::map map) {
  return umap.add_map(map);
}
} // namespace detail

template <typename Composite>
struct UnionAsVector
    : std::vector<decltype(
          detail::extractFromUnion(Composite(), isl::space()))> {
  using Element = decltype(detail::extractFromUnion(Composite(), isl::space()));
  UnionAsVector() {}
  UnionAsVector(Composite composite) {
    this->reserve(detail::nElement(composite));
    detail::foreachElement(
        [&](Element e) -> void { this->push_back(e); }, composite);
  }
  Composite asUnion() {
    Composite res((*this)[0]);
    for (int i = 1; i < this->size(); ++i) {
      res = detail::addElement(res, (*this)[i]);
    }
    return res;
  }
};

struct IslIdIslHash {
  size_t operator()(const isl::id& id) const {
    return isl_id_get_hash(id.get());
  }
};

inline bool operator==(const isl::id& id1, const isl::id& id2) {
  // isl_id objects can be compared by pointer value.
  return id1.get() == id2.get();
}

inline bool operator!=(const isl::id& id1, const isl::id& id2) {
  return id1.get() != id2.get();
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T dropDimsPreserveTuple(T t, isl::dim_type type, int from, int length) {
  auto id = t.get_tuple_id(type);
  t = t.drop_dims(type, from, length);
  return t.set_tuple_id(type, id);
}

// Given a space and a list of values, this returns the corresponding multi_val.
template <typename T>
isl::multi_val makeMultiVal(isl::space s, const std::vector<T>& vals) {
  isl::multi_val mv = isl::multi_val::zero(s);
  CHECK_EQ(vals.size(), s.dim(isl::dim_type::set));
  for (size_t i = 0; i < vals.size(); ++i) {
    mv = mv.set_val(i, isl::val(s.get_ctx(), vals[i]));
  }
  return mv;
}

// Takes a space of parameters, a range of (ids, extent)-pairs and returns
// the set such that:
// 1. the space is paramSpace extended by all the ids (enforced to not be
//    present in the original space)
// 2. each new parameter dimension p(i) is bounded to be in [0, e(i) - 1]
// 3. if e (i) == 0 then no constraint is set on the corresponding id(i)
template <typename IterPair>
inline isl::set makeParameterContext(
    isl::space paramSpace,
    const IterPair begin,
    const IterPair end) {
  for (auto it = begin; it != end; ++it) {
    paramSpace = paramSpace.add_param(it->first);
  }
  isl::set res(isl::set::universe(paramSpace));
  for (auto it = begin; it != end; ++it) {
    isl::aff a(isl::aff::param_on_domain_space(paramSpace, it->first));
    res = res & (isl::aff_set(a) >= 0) & (isl::aff_set(a) < it->second);
  }
  return res;
}

// Given a space and values for parameters, this function creates the set
// that ties the space parameter to the values.
//
template <typename T>
inline isl::set makeSpecializationSet(
    isl::space space,
    const std::unordered_map<std::string, T>& paramValues) {
  auto ctx = space.get_ctx();
  for (auto kvp : paramValues) {
    space = space.add_param(isl::id(ctx, kvp.first));
  }
  auto set = isl::set::universe(space);
  for (auto kvp : paramValues) {
    auto id = isl::id(ctx, kvp.first);
    isl::aff affParam(isl::aff::param_on_domain_space(space, id));
    set = set & (isl::aff_set(affParam) == kvp.second);
  }
  return set;
}

template <typename T>
inline isl::set makeSpecializationSet(
    isl::space space,
    std::initializer_list<std::pair<const std::string, T>> paramValues) {
  std::unordered_map<std::string, T> map(paramValues);
  return makeSpecializationSet(space, map);
}

template <typename T>
inline isl::set makeSpecializationSet(
    isl::space space,
    std::initializer_list<std::pair<isl::id, T>> paramValues) {
  std::unordered_map<std::string, T> map;
  for (auto kvp : paramValues) {
    map.emplace(kvp.first.get_name(), kvp.second);
  }
  return makeSpecializationSet(space, map);
}

namespace detail {

// Helper class used to support range-based for loops on isl::*_list types.
// It only provides the minimal required implementation.
// "L" is the list type, "E" is the corresponding element type.
template <typename E, typename L>
struct ListIter {
  ListIter(L& list, int pos) : list_(list), pos_(pos) {}
  bool operator!=(const ListIter& other) {
    return pos_ != other.pos_;
  }
  void operator++() {
    pos_++;
  }
  E operator*() {
    return list_.get(pos_);
  }

 private:
  L& list_;
  int pos_;
};

template <typename L>
auto begin(L& list) -> ListIter<decltype(list.get(0)), L> {
  return ListIter<decltype(list.get(0)), L>(list, 0);
}

template <typename L>
auto end(L& list) -> ListIter<decltype(list.get(0)), L> {
  return ListIter<decltype(list.get(0)), L>(list, list.n());
}

} // namespace detail

// The begin() and end() functions used for range-based for loops
// need to be available in the same namespace as that of the class
// of the object to which it is applied.
using detail::begin;
using detail::end;

} // namespace isl

namespace isl {
namespace with_exceptions {

enum struct IslCtxOption : int { Uninitialized = -1, Default = 0 };

struct CtxUPtrDeleter {
  void operator()(ctx* c) {
    isl_ctx_free(c->release());
    delete c;
  }
};
typedef std::unique_ptr<ctx, CtxUPtrDeleter> CtxUPtr;

// C++11 thread-safe init
isl::ctx globalIslCtx(IslCtxOption options = IslCtxOption::Default);

} // namespace with_exceptions
} // namespace isl

#include "tc/external/detail/islpp-inl.h"
