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
#include "tc/core/polyhedral/schedule_tree.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <glog/logging.h>

#include "tc/external/isl.h"

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/functional.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

using namespace std;

namespace tc {
namespace polyhedral {
namespace detail {

using namespace isl;

namespace {

// Returns the list of positions in [current, parent(target)] to find the node
// As a byproduct, the special-case current == target is the only case where
// findDescendant may return {} (takes care of finding the root ScheduleTree).
deque<size_t> findDescendant(
    const ScheduleTree* current,
    const ScheduleTree* target,
    size_t iteration = 0) {
  if (current == target && iteration == 0) {
    // This special case is only meant to catch initial call of
    // current == target (e.g. for root or equality).
    // Otherwise, returning {} is an indication of failure to find.
    return {};
  }
  for (size_t i = 0; i < current->numChildren(); ++i) {
    if (current->child({i}) == target) {
      return {i};
    }
  }
  for (size_t i = 0; i < current->numChildren(); ++i) {
    deque<size_t> res =
        findDescendant(current->child({i}), target, iteration + 1);
    if (res.size() > 0) {
      res.push_front(i);
      return res;
    }
  }
  return {};
}

vector<size_t> positionRelativeToSubtree(
    const ScheduleTree* relativeRoot,
    const ScheduleTree* target) {
  TC_CHECK(relativeRoot != target)
      << "Need a strict relative root to find position";
  auto res = findDescendant(relativeRoot, target);
  return vector<size_t>{res.begin(), res.end()};
}

vector<const ScheduleTree*> constAncestorsInSubTree(
    const ScheduleTree* relativeRoot,
    const ScheduleTree* target) {
  if (relativeRoot == target) {
    return vector<const ScheduleTree*>();
  }
  vector<size_t> cp(positionRelativeToSubtree(relativeRoot, target));
  if (cp.size() == 0) {
    // Special case, this must be the root
    TC_CHECK_EQ(relativeRoot, target);
    return {};
  }
  vector<const ScheduleTree*> res(cp.size() + 1, nullptr);
  // always return the root as first element
  res[0] = relativeRoot;
  for (size_t i = 0; i < cp.size(); ++i) {
    res[i + 1] = res[i]->child({cp[i]});
  }
  // Check last element is self for consistency
  TC_CHECK_EQ(res.back(), target)
      << "Could not find " << *target << " under " << *relativeRoot << "\n";
  // Resize to drop self, and check again for consistency
  res.resize(cp.size());
  TC_CHECK_NE(res.back(), target);
  return res;
}

vector<ScheduleTree*> ancestorsInSubTree(
    ScheduleTree* relativeRoot,
    ScheduleTree* target) {
  const auto rr = relativeRoot;
  const auto t = target;
  auto tmp = constAncestorsInSubTree(rr, t);
  vector<ScheduleTree*> res;
  res.reserve(tmp.size());
  for (auto v : tmp) {
    res.push_back(const_cast<ScheduleTree*>(v));
  }
  return res;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////
//                        ScheduleTree
////////////////////////////////////////////////////////////////////////////////
ScheduleTree::~ScheduleTree() {}

ScheduleTree::ScheduleTree(const ScheduleTree& st)
    : ctx_(st.ctx_), children_(), type_(st.type_) {
  children_.reserve(st.children_.size());
  for (const auto& child : st.children()) {
    children_.push_back(ScheduleTree::makeScheduleTree(*child));
  }
}

ScheduleTreeUPtr ScheduleTree::makeScheduleTree(const ScheduleTree& tree) {
  return tree.clone();
}

ScheduleTree* ScheduleTree::child(const vector<size_t>& positions) {
  const auto& st = *this;
  return const_cast<ScheduleTree*>(st.child(positions));
}

const ScheduleTree* ScheduleTree::child(const vector<size_t>& positions) const {
  auto st = this;
  for (auto pos : positions) {
    TC_CHECK_LE(0u, pos) << "Reached a leaf";
    TC_CHECK_GT(st->children_.size(), pos) << "Out of children bounds";
    st = st->children_[pos].get();
  }
  return st;
}

ScheduleTree* ScheduleTree::ancestor(
    ScheduleTree* relativeRoot,
    size_t generations) {
  const auto& st = *this;
  return const_cast<ScheduleTree*>(st.ancestor(relativeRoot, generations));
}

const ScheduleTree* ScheduleTree::ancestor(
    const ScheduleTree* relativeRoot,
    size_t generations) const {
  TC_CHECK_LT(0u, generations) << "Nonpositive ancestor generation";
  auto as = constAncestorsInSubTree(relativeRoot, this);
  TC_CHECK_GE(as.size(), generations) << "Out of ancestors bounds";
  return as[as.size() - generations];
}

vector<ScheduleTree*> ScheduleTree::ancestors(ScheduleTree* relativeRoot) {
  return ancestorsInSubTree(relativeRoot, this);
}

vector<const ScheduleTree*> ScheduleTree::ancestors(
    const ScheduleTree* relativeRoot) const {
  return constAncestorsInSubTree(relativeRoot, this);
}

vector<size_t> ScheduleTree::positionRelativeTo(
    const ScheduleTree* relativeRoot) const {
  return positionRelativeToSubtree(relativeRoot, this);
}

size_t ScheduleTree::scheduleDepth(const ScheduleTree* relativeRoot) const {
  size_t depth = 0;
  for (auto const& anc : ancestors(relativeRoot)) {
    auto bandElem = anc->as<ScheduleTreeBand>();
    if (!bandElem) {
      continue;
    }
    depth += bandElem->nMember();
  }
  return depth;
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeBand(
    isl::MultiUnionPwAff<Statement, Band> mupa,
    std::vector<ScheduleTreeUPtr>&& children) {
  std::vector<bool> coincident(mupa.size(), false);
  std::vector<bool> unroll(mupa.size(), false);
  auto res = ScheduleTreeBand::make(
      mupa, false, coincident, unroll, std::move(children));
  return res;
}

ScheduleTreeUPtr ScheduleTree::makeEmptyBand(const ScheduleTree* root) {
  auto domain = root->as<ScheduleTreeDomain>();
  TC_CHECK(domain);
  auto space = domain->domain_.get_space().add_unnamed_tuple_ui<Band>(0);
  auto mv = isl::MultiVal<Band>::zero(space);
  auto zero = isl::MultiUnionPwAff<Statement, Band>(domain->domain_, mv);
  return ScheduleTree::makeBand(zero);
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeDomain(
    isl::union_set domain,
    std::vector<ScheduleTreeUPtr>&& children) {
  return ScheduleTreeDomain::make(domain, std::move(children));
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeContext(
    isl::Set<Prefix> context,
    std::vector<ScheduleTreeUPtr>&& children) {
  return ScheduleTreeContext::make(context, std::move(children));
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeFilter(
    isl::union_set filter,
    std::vector<ScheduleTreeUPtr>&& children) {
  return ScheduleTreeFilter::make(filter, std::move(children));
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeMappingUnsafe(
    const std::vector<mapping::MappingId>& mappedIds,
    isl::UnionPwAffListOn<Statement> mappedAffs,
    std::vector<ScheduleTreeUPtr>&& children) {
  TC_CHECK_EQ(mappedIds.size(), static_cast<size_t>(mappedAffs.size()))
      << "expected as many mapped ids as affs";
  ScheduleTreeMapping::Mapping mapping;
  for (size_t i = 0, n = mappedAffs.size(); i < n; ++i) {
    mapping.emplace(mappedIds.at(i), mappedAffs.get_at(i));
  }
  TC_CHECK_GE(mapping.size(), 1u) << "empty mapping";
  TC_CHECK_EQ(mappedIds.size(), mapping.size())
      << "some id is used more than once in the mapping";
  auto ctx = mappedIds[0].get_ctx();
  return ScheduleTreeMapping::make(ctx, mapping, std::move(children));
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeExtension(
    isl::union_map extension,
    std::vector<ScheduleTreeUPtr>&& children) {
  return ScheduleTreeExtension::make(extension, std::move(children));
}

std::unique_ptr<ScheduleTree> ScheduleTree::makeThreadSpecificMarker(
    isl::ctx ctx,
    std::vector<ScheduleTreeUPtr>&& children) {
  return ScheduleTreeThreadSpecificMarker::make(ctx, std::move(children));
}

////////////////////////////////////////////////////////////////////////////////
//                        Collector member functions
////////////////////////////////////////////////////////////////////////////////
namespace {

vector<ScheduleTree*> unconst(vector<const ScheduleTree*>&& v) {
  vector<ScheduleTree*> res;
  res.reserve(v.size());
  for (const auto sptr : v) {
    res.push_back(const_cast<ScheduleTree*>(sptr));
  }
  return res;
}

} // namespace

vector<ScheduleTree*> ScheduleTree::collectDFSPostorder(ScheduleTree* tree) {
  const ScheduleTree* t = tree;
  return unconst(collectDFSPostorder(t));
}
vector<ScheduleTree*> ScheduleTree::collectDFSPreorder(ScheduleTree* tree) {
  const ScheduleTree* t = tree;
  return unconst(collectDFSPreorder(t));
}
vector<ScheduleTree*> ScheduleTree::collectDFSPostorder(
    ScheduleTree* tree,
    detail::ScheduleTreeType type) {
  const ScheduleTree* t = tree;
  return unconst(collectDFSPostorder(t, type));
}
vector<ScheduleTree*> ScheduleTree::collectDFSPreorder(
    ScheduleTree* tree,
    detail::ScheduleTreeType type) {
  const ScheduleTree* t = tree;
  return unconst(collectDFSPreorder(t, type));
}

vector<const ScheduleTree*> ScheduleTree::collectDFSPostorder(
    const ScheduleTree* tree) {
  vector<const ScheduleTree*> res;
  for (const auto& c : tree->children_) {
    auto tmp = ScheduleTree::collectDFSPostorder(c.get());
    res.insert(res.end(), tmp.begin(), tmp.end());
  }
  res.insert(res.end(), tree);
  return res;
}

vector<const ScheduleTree*> ScheduleTree::collectDFSPostorder(
    const ScheduleTree* tree,
    detail::ScheduleTreeType type) {
  auto filterType = [type](const ScheduleTree* t) { return t->type_ == type; };
  return functional::Filter(filterType, collectDFSPostorder(tree));
}

vector<const ScheduleTree*> ScheduleTree::collectDFSPreorder(
    const ScheduleTree* tree) {
  vector<const ScheduleTree*> res{tree};
  for (const auto& c : tree->children_) {
    auto tmp = ScheduleTree::collectDFSPreorder(c.get());
    res.insert(res.end(), tmp.begin(), tmp.end());
  }
  return res;
}

vector<const ScheduleTree*> ScheduleTree::collectDFSPreorder(
    const ScheduleTree* tree,
    detail::ScheduleTreeType type) {
  auto filterType = [type](const ScheduleTree* t) { return t->type_ == type; };
  return functional::Filter(filterType, collectDFSPreorder(tree));
}

bool ScheduleTree::operator==(const ScheduleTree& other) const {
  // ctx_ cmp ?
  if (type_ != other.type_) {
    return false;
  }
  if (children_.size() != other.children_.size()) {
    return false;
  }
  if (!elemEquals(this, &other, type_)) {
    return false;
  }
  TC_CHECK(!other.as<ScheduleTreeSet>())
      << "NYI: ScheduleTreeType::Set comparison";
  for (size_t i = 0; i < children_.size(); ++i) {
    if (*children_[i] != *other.children_[i]) {
      return false;
    }
  }
  return true;
}

} // namespace detail
} // namespace polyhedral
} // namespace tc
