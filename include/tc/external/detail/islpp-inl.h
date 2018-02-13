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
#include "tc/external/isl.h"

namespace isl {

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::aff to perform arithmetic and create/combine with sets
///////////////////////////////////////////////////////////////////////////////
inline isl::aff operator*(int i, isl::aff A) {
  isl::val V(isl::val(A.get_ctx(), i));
  return A * V;
}

inline isl::aff operator*(isl::aff A, int i) {
  return i * A;
}

inline isl::aff operator*(isl::val V, isl::aff A) {
  isl::aff AV(A.get_local_space().domain(), V);
  return A.mul(AV);
}

inline isl::aff operator*(isl::aff A, isl::val V) {
  return V * A;
}

inline isl::aff operator/(isl::aff A, int i) {
  isl::ctx ctx = A.get_ctx();
  isl::aff T(isl::local_space(A.get_space().domain()), isl::val(ctx, i));
  return A.div(T);
}

inline isl::aff operator+(int i, isl::aff A) {
  isl::ctx ctx = A.get_ctx();
  return A + isl::val(ctx, i);
}

inline isl::aff operator+(isl::aff A, isl::val v) {
  isl::aff T(isl::local_space(A.get_space().domain()), v);
  return A.add(T);
}

inline isl::aff operator+(isl::val v, isl::aff A) {
  return A + v;
}

inline isl::aff operator+(isl::aff A, isl::aff B) {
  return A.add(B);
}

inline isl::aff operator+(isl::aff A, int i) {
  return i + A;
}

inline isl::aff operator-(isl::aff A, int i) {
  return A + (-i);
}

inline isl::aff operator-(int i, isl::aff A) {
  return (A + (-i)).neg();
}

inline isl::set operator>=(isl::aff_set A, isl::val v) {
  auto T = isl::aff(isl::local_space(A.aff.get_space().domain()), v);
  return A.aff.ge_set(T);
}

inline isl::set operator>=(isl::aff_set A, int i) {
  auto ctx = A.aff.get_ctx();
  return A >= isl::val(ctx, i);
}

inline isl::set operator>=(int i, isl::aff_set A) {
  return A.aff.neg() >= -i;
}

inline isl::set operator>=(isl::aff_set A, isl::aff B) {
  return A.aff.ge_set(B);
}

inline isl::set operator>=(isl::aff A, isl::aff_set B) {
  return A.ge_set(B.aff);
}

inline isl::set operator>(isl::aff_set A, int i) {
  return A >= (i + 1);
}

inline isl::set operator>(int i, isl::aff_set A) {
  return (i - 1) >= A;
}

inline isl::set operator>(isl::aff_set A, isl::aff B) {
  return A >= (B + 1);
}

inline isl::set operator>(isl::aff A, isl::aff_set B) {
  return (A - 1) >= B;
}

inline isl::set operator<=(isl::aff_set A, int i) {
  return A.aff.neg() >= -i;
}

inline isl::set operator<=(isl::aff_set A, isl::val v) {
  return A.aff.neg() >= v.neg();
}

inline isl::set operator<=(int i, isl::aff_set A) {
  return A >= i;
}

inline isl::set operator<=(isl::aff_set A, isl::aff B) {
  return A.aff.le_set(B);
}

inline isl::set operator<=(isl::aff A, isl::aff_set B) {
  return A.le_set(B.aff);
}

inline isl::set operator<(isl::aff_set A, int i) {
  return A <= i - 1;
}

inline isl::set operator<(int i, isl::aff_set A) {
  return i + 1 <= A;
}

inline isl::set operator<(isl::aff_set A, isl::val v) {
  return A <= v - 1;
}

inline isl::set operator<(isl::aff_set A, isl::aff B) {
  return A <= B - 1;
}

inline isl::set operator<(isl::aff A, isl::aff_set B) {
  return A + 1 <= B;
}

inline isl::set operator==(isl::aff_set A, int i) {
  return (A <= i) & (A >= i);
}

inline isl::set operator==(int i, isl::aff_set A) {
  return A == i;
}

inline isl::set operator==(isl::aff_set A, isl::aff B) {
  return (A <= B) & (A >= B);
}

inline isl::set operator==(isl::aff A, isl::aff_set B) {
  return (A <= B) & (A >= B);
}

inline isl::map operator>=(isl::aff_map A, isl::aff B) {
  return A > B - 1;
}

inline isl::map operator>(isl::aff_map A, isl::aff B) {
  auto pwA = isl::pw_aff(A.aff);
  auto pwB = isl::pw_aff(B);
  return pwA.gt_map(pwB);
}

inline isl::map operator<(isl::aff_map A, isl::aff B) {
  auto pwA = isl::pw_aff(A.aff);
  auto pwB = isl::pw_aff(B);
  return pwA.lt_map(pwB);
}

inline isl::map operator<=(isl::aff_map A, isl::aff B) {
  return A < B + 1;
}

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::union_set
///////////////////////////////////////////////////////////////////////////////
inline isl::set operator&(isl::set S1, isl::set S2) {
  return S1.intersect(S2);
}

inline isl::union_set operator&(isl::union_set S1, isl::set S2) {
  return S1.intersect(isl::union_set(S2));
}

inline isl::union_set operator&(isl::set S1, isl::union_set S2) {
  return S2 & S1;
}

inline isl::union_set operator&(isl::union_set S1, isl::union_set S2) {
  return S1.intersect(S2);
}

///////////////////////////////////////////////////////////////////////////////
// Operations on isl::set and isl::point
///////////////////////////////////////////////////////////////////////////////
inline isl::set operator&(isl::set S1, isl::point P2) {
  return S1.intersect(isl::set(P2));
}

inline isl::set operator&(isl::point P1, isl::set S2) {
  return S2 & P1;
}

inline isl::set makeUniverseSet(
    const isl::ctx& ctx,
    std::vector<const char*> pNames) {
  auto s = isl::set::universe(isl::space(ctx, pNames.size()));
  int idx = 0;
  for (auto n : pNames) {
    s = isl::manage(isl_set_set_dim_name(s.release(), isl_dim_param, idx++, n));
  }
  return s;
}

// Better if we had isl::set::align(s) a member
inline isl::set makeAlignedSet(isl::set orig, isl::set s) {
  return isl::manage(
      isl_set_align_params(orig.copy(), isl_set_get_space(s.copy())));
}

inline isl::point makePoint(
    isl::space s,
    std::vector<const char*> names,
    std::vector<long> vals) {
  isl::point pt(s);
  int idx = 0;
  for (auto n : names) {
    int pos = isl_space_find_dim_by_name(s.get(), isl_dim_param, n);
    assert(pos >= 0);
    if (vals[idx] >= 0) {
      pt = isl::manage(
          isl_point_add_ui(pt.release(), isl_dim_param, pos, vals[idx]));
    } else {
      pt = isl::manage(
          isl_point_sub_ui(pt.release(), isl_dim_param, pos, -vals[idx]));
    }
    idx++;
  }
  return pt;
}

inline isl::point makePoint(
    isl::space s,
    std::unordered_map<std::string, long> nameMap) {
  std::vector<long> vals;
  std::vector<const char*> names;
  for (const auto& kvp : nameMap) {
    names.push_back(kvp.first.c_str());
    vals.push_back(kvp.second);
  }
  return makePoint(s, names, vals);
}

inline long evalIntegerAt(isl::aff a, isl::point pt) {
  // Parametric only
  assert(isl::set(pt).dim(isl::dim_type::in) == 0);
  assert(
      (unsigned)isl_aff_dim(a.get(), isl_dim_in) ==
      isl::set(pt).dim(isl::dim_type::in));
  assert(
      (unsigned)isl_aff_dim(a.get(), isl_dim_param) ==
      isl::set(pt).dim(isl::dim_type::param));
  auto aff_map = isl::manage(isl_map_from_aff(a.copy()));
  auto pt_map = isl::manage(isl_map_from_domain(isl::set(pt).release()));
  auto m = pt_map.apply_domain(aff_map);
  assert(m.is_single_valued());
  // Project out all parameters and only keep the value
  m = m.project_out(
      isl::dim_type::param, 0, isl::set(pt).dim(isl::dim_type::param));
  auto v = isl::manage(isl_map_plain_get_val_if_fixed(m.get(), isl_dim_in, 0));
  assert(isl_val_get_den_si(v.get()) == 1); // not a rational
  return isl_val_get_num_si(v.get());
}

inline isl::schedule_node MoveDownToMark(
    isl::schedule_node node,
    std::string name) {
  while (node.get()) {
    isl_schedule_node_type type = isl_schedule_node_get_type(node.get());
    if (type == isl_schedule_node_mark) {
      isl_id* id = isl_schedule_node_mark_get_id(node.get());
      std::string mark_name{isl_id_get_name(id)};
      isl_id_free(id);
      if (mark_name == name)
        return node;
    } else if (
        type == isl_schedule_node_sequence || type == isl_schedule_node_set) {
      throw isl::with_exceptions::islpp_error(
          "cannot unambiguously descend past sequence/set node");
    }
    node = node.child(0);
  }

  throw isl::with_exceptions::islpp_error(
      "no child mark node with the name \"" + name + "\"");
}
} // namespace isl
