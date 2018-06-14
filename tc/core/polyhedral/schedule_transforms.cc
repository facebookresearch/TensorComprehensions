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
#include "tc/core/polyhedral/schedule_transforms.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <glog/logging.h>

#include "tc/external/isl.h"

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/polyhedral/functional.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//                        Transformation functions, out-of-class
////////////////////////////////////////////////////////////////////////////////

namespace tc {
namespace polyhedral {
using namespace detail;

isl::union_map extendSchedule(
    const ScheduleTree* node,
    isl::union_map schedule) {
  if (auto bandElem = node->elemAs<ScheduleTreeElemBand>()) {
    if (bandElem->nMember() > 0) {
      schedule =
          schedule.flat_range_product(isl::union_map::from(bandElem->mupa_));
    }
  } else if (auto filterElem = node->elemAs<ScheduleTreeElemFilter>()) {
    schedule = schedule.intersect_domain(filterElem->filter_);
  } else if (auto extensionElem = node->elemAs<ScheduleTreeElemExtension>()) {
    // FIXME: we may need to restrict the range of reversed extension map to
    // schedule values that correspond to active domain elements at this
    // point.
    schedule = schedule.unite(
        extensionElem->extension_.reverse().intersect_range(schedule.range()));
  }

  return schedule;
}

namespace {
isl::union_map partialScheduleImpl(
    const ScheduleTree* root,
    const ScheduleTree* node,
    bool useNode) {
  auto schedule = isl::null<isl::union_map>();
  auto nodes = node->ancestors(root);
  if (useNode) {
    nodes.push_back(node);
  }
  for (auto anc : nodes) {
    if (auto domainElem = anc->elemAs<ScheduleTreeElemDomain>()) {
      schedule = isl::union_map::from_domain(domainElem->domain_);
    } else {
      schedule = extendSchedule(anc, schedule);
    }
  }
  return schedule;
}
} // namespace

isl::union_map prefixSchedule(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return partialScheduleImpl(root, node, false);
}

isl::union_map partialSchedule(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return partialScheduleImpl(root, node, true);
}

namespace {
/*
 * If "node" is any filter, then intersect "domain" with that filter.
 */
isl::union_set applyFilter(isl::union_set domain, const ScheduleTree* node) {
  if (auto filterElem = node->elemAs<ScheduleTreeElemFilter>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

/*
 * If "node" is a mapping, then intersect "domain" with its filter.
 */
isl::union_set applyMapping(isl::union_set domain, const ScheduleTree* node) {
  if (auto filterElem = node->elemAs<ScheduleTreeElemMapping>()) {
    return domain.intersect(filterElem->filter_);
  }
  return domain;
}

// Get the set of domain elements that are active below
// the given branch of nodes, filtered using "filter".
//
// Domain elements are introduced by the root domain node.  Some nodes
// refine this set of elements based on "filter".  Extension nodes
// are considered to introduce additional domain points.
isl::union_set collectDomain(
    const ScheduleTree* root,
    const vector<const ScheduleTree*>& nodes,
    isl::union_set (*filter)(isl::union_set domain, const ScheduleTree* node)) {
  auto domainElem = root->elemAs<ScheduleTreeElemDomain>();
  TC_CHECK(domainElem) << "root must be a Domain node" << *root;

  auto domain = domainElem->domain_;

  for (auto anc : nodes) {
    domain = filter(domain, anc);
    if (auto extensionElem = anc->elemAs<ScheduleTreeElemExtension>()) {
      auto parentSchedule = prefixSchedule(root, anc);
      auto extension = extensionElem->extension_;
      TC_CHECK(parentSchedule) << "missing root domain node";
      parentSchedule = parentSchedule.intersect_domain(domain);
      domain = domain.unite(parentSchedule.range().apply(extension));
    }
  }
  return domain;
}

// Get the set of domain elements that are active below
// the given branch of nodes.
isl::union_set activeDomainPointsHelper(
    const ScheduleTree* root,
    const vector<const ScheduleTree*>& nodes) {
  return collectDomain(root, nodes, &applyFilter);
}

} // namespace

isl::union_set prefixMappingFilter(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return collectDomain(root, node->ancestors(root), &applyMapping);
}

isl::union_set activeDomainPoints(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  return activeDomainPointsHelper(root, node->ancestors(root));
}

isl::union_set activeDomainPointsBelow(
    const ScheduleTree* root,
    const ScheduleTree* node) {
  auto ancestors = node->ancestors(root);
  ancestors.emplace_back(node);
  return activeDomainPointsHelper(root, ancestors);
}

vector<ScheduleTree*> collectScheduleTreesPath(
    std::function<ScheduleTree*(ScheduleTree*)> next,
    ScheduleTree* start) {
  vector<ScheduleTree*> res{start};
  auto n = start;
  while ((n = next(n)) != nullptr) {
    res.push_back(n);
  }
  return res;
}

vector<const ScheduleTree*> collectScheduleTreesPath(
    std::function<const ScheduleTree*(const ScheduleTree*)> next,
    const ScheduleTree* start) {
  vector<const ScheduleTree*> res{start};
  auto n = start;
  while ((n = next(n)) != nullptr) {
    res.push_back(n);
  }
  return res;
}

// Replace "tree" in the list of its parent's children with newTree.
// Returns the pointer to newTree for call chaining purposes.
ScheduleTree* swapSubtree(
    ScheduleTree* relativeRoot,
    ScheduleTree* tree,
    ScheduleTreeUPtr& newTree) {
  TC_CHECK(relativeRoot != tree) << "Need a strict relative root to graft";
  auto cpos = tree->positionRelativeTo(relativeRoot).back();
  auto parent = tree->ancestor(relativeRoot, 1);
  auto rawPtr = newTree.get();
  parent->swapChild(cpos, newTree);
  return rawPtr;
}

namespace {

/*
 * If the child of the band node "st" is also a band node,
 * then combine the two band nodes into a single band node
 * at the position of "st" and set "moveChildren" to true.
 * The coincident fields corresponding to the band members
 * that come from the nested band are reset, because the coincident
 * members of that nested band are only known to be coincident
 * within the outer band.
 */
ScheduleTree* joinBandsHelper(ScheduleTree* st, bool& moveChildren) {
  moveChildren = false;
  TC_CHECK(st->elemAs<ScheduleTreeElemBand>());
  if (st->numChildren() != 1) {
    return st;
  }

  auto eb = st->elemAs<ScheduleTreeElemBand>();
  auto ebChild = st->child({0})->elemAs<ScheduleTreeElemBand>();
  if (!ebChild) {
    return st;
  }

  auto& partialSchedule = eb->mupa_;
  auto& partialScheduleChild = ebChild->mupa_;
  partialSchedule = partialSchedule.flat_range_product(partialScheduleChild);
  eb->coincident_.resize(
      eb->coincident_.size() + ebChild->coincident_.size(), false);
  eb->unroll_.insert(
      eb->unroll_.end(), ebChild->unroll_.begin(), ebChild->unroll_.end());

  moveChildren = true;
  return st;
}

} // namespace

ScheduleTree* joinBands(ScheduleTree* st, bool permutable) {
  bool moveChildren;
  st = joinBandsHelper(st, moveChildren);
  // Stupid private access hack, remove when moving to unique_ptr
  if (moveChildren) {
    // Just overwrite children and let shared pointers go out of scope
    auto children = st->detachChildren();
    TC_CHECK_EQ(1u, children.size()) << "expected a sequence of bands";
    st->appendChildren(children[0]->detachChildren());
  }
  st->elemAs<ScheduleTreeElemBand>()->permutable_ = permutable;
  return st;
}

ScheduleTree* joinBandsIterative(ScheduleTree* st, bool permutable) {
  bool moveChildren = true;
  while (moveChildren) {
    st = joinBandsHelper(st, moveChildren);
    // Stupid private access hack, remove when moving to unique_ptr
    if (moveChildren) {
      auto children = st->detachChildren();
      TC_CHECK_EQ(1u, children.size()) << "expected a sequence of bands";
      st->appendChildren(children[0]->detachChildren());
    }
  }
  st->elemAs<ScheduleTreeElemBand>()->permutable_ = permutable;
  return st;
}

using TileOptionsType = std::underlying_type<TileOptions>::type;

bool operator&(TileOptions actual, TileOptions wanted) {
  return static_cast<TileOptionsType>(actual) &
      static_cast<TileOptionsType>(wanted);
}

TileOptions operator|(TileOptions actual, TileOptions wanted) {
  return static_cast<TileOptions>(
      static_cast<TileOptionsType>(actual) |
      static_cast<TileOptionsType>(wanted));
}

// Note that by-reference ctx has only a semantic meaning: context will be
// changed by this call.
void applyTileOptions(isl::ctx& ctx, TileOptions tileOptions) {
  isl_options_set_tile_scale_tile_loops(
      ctx.get(), (tileOptions & TileOptions::ScaleTileLoops) ? 1 : 0);
  isl_options_set_tile_shift_point_loops(
      ctx.get(), (tileOptions & TileOptions::ShiftPointLoops) ? 1 : 0);
}

ScheduleTree*
bandSplit(ScheduleTree* relativeRoot, ScheduleTree* tree, size_t pos) {
  TC_CHECK(tree->elemAs<ScheduleTreeElemBand>()) << "Not a band:\n" << *tree;
  auto band = tree->elemAs<ScheduleTreeElemBand>();
  size_t n = band->nMember();
  TC_CHECK_LT(0u, n) << "no bands to split";
  TC_CHECK_LE(0u, pos) << "position out of bounds";
  TC_CHECK_GE(n, pos) << "position out of bounds";

  // Detach and reattach children to avoid making copies.
  auto children = tree->detachChildren();
  auto newChild = ScheduleTree::makeScheduleTree(*tree);
  newChild->appendChildren(std::move(children));
  auto newChildBand = newChild->elemAs<ScheduleTreeElemBand>();
  newChildBand->drop(0, pos);

  tree->appendChild(std::move(newChild));
  band->drop(pos, n - pos);
  return tree;
}

ScheduleTree*
bandSplitOut(ScheduleTree* relativeRoot, ScheduleTree* tree, size_t pos) {
  auto band = tree->elemAs<ScheduleTreeElemBand>();
  TC_CHECK(band);
  auto size = band->nMember();
  if (pos != size - 1) {
    tree = bandSplit(relativeRoot, tree, pos + 1);
  }
  if (pos != 0) {
    tree = bandSplit(relativeRoot, tree, pos);
    tree = tree->child({0});
  }
  return tree;
}

namespace {

template <typename T>
std::ostream& operator<<(ostream& os, const vector<T>& v) {
  for (auto vv : v) {
    os << vv << " ";
  }
  return os;
}
} // namespace

ScheduleTree* bandTile(
    ScheduleTree* st,
    const vector<size_t>& tileSizes,
    TileOptions tileOptions) {
  auto eb = st->elemAs<ScheduleTreeElemBand>();
  TC_CHECK(eb) << "Not a band: " << *st;

  if (tileSizes.size() == 0) {
    return st;
  }
  auto& band = *eb;
  TC_CHECK(band.permutable_) << "Can't tile an non-permutable band" << band;

  auto ts = tileSizes;
  if (band.nMember() > ts.size()) {
    ts.resize(band.nMember(), 0);
  }
  if (band.nMember() < ts.size()) {
    LOG(WARNING) << "Resizing tile sizes to " << band.nMember()
                 << " entries: " << ts;
    ts.resize(band.nMember());
  }
  TC_CHECK_EQ(band.nMember(), ts.size()) << "NYI: incorrect sizes: " << ts;
  // TODO: adapt size
  // TODO: imperfectly nested loop tiling

  // Create a child, copy of st before outer tiling
  ScheduleTreeUPtr childUPtr = ScheduleTree::makeScheduleTree(*st);

  for (size_t i = 0;
       i < std::min(static_cast<size_t>(band.nMember()), ts.size());
       ++i) {
    auto upa = band.mupa_.get_union_pw_aff(i);
    if (ts[i]) {
      upa = upa.scale_down(isl::val(st->ctx_, ts[i])).floor();
      if (tileOptions & TileOptions::ScaleTileLoops) {
        upa = upa.scale_val(isl::val(st->ctx_, ts[i]));
      }
    } else {
      upa = upa.scale_val(isl::val(st->ctx_, ts[i]));
    }
    band.mupa_ = band.mupa_.set_union_pw_aff(i, upa);
  }

  auto ebChild = childUPtr->elemAs<ScheduleTreeElemBand>();
  TC_CHECK(ebChild) << "Not a band: " << *childUPtr;
  auto& childBand = *ebChild;
  // No need for isl_schedule_band_point, it's almost done
  if (tileOptions & TileOptions::ShiftPointLoops) {
    auto mupa = band.mupa_;
    if (!(tileOptions & TileOptions::ScaleTileLoops)) {
      mupa = mupa.scale(makeMultiVal(mupa.get_space(), ts));
    }
    childBand.mupa_ = childBand.mupa_.sub(mupa);
  }

  st->detachChildren(); // let 'em die
  st->appendChild(std::move(childUPtr));

  return st;
}

ScheduleTree* bandScale(ScheduleTree* tree, const vector<size_t>& scales) {
  auto eb = tree->elemAs<ScheduleTreeElemBand>();
  TC_CHECK(eb) << "Not a band: " << *tree;
  auto& band = *eb;

  // This mimics the behavior of bandTile...
  auto s = scales;
  if (s.size() < band.nMember()) {
    s.resize(band.nMember(), 0);
  }
  if (band.nMember() < s.size()) {
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "Resizing scales to " << band.nMember() << " entries: " << s;
    s.resize(band.nMember());
  }
  auto& mupa = band.mupa_;
  auto space = mupa.get_space();
  mupa = mupa.scale(isl::makeMultiVal(space, s));
  return tree;
}

namespace {

template <typename T>
vector<T> reversed(const vector<T>& vec) {
  vector<T> result;
  result.reserve(vec.size());
  result.insert(result.begin(), vec.rbegin(), vec.rend());
  return result;
}

template <typename T>
vector<const ScheduleTree*> filterType(const vector<const ScheduleTree*>& vec) {
  vector<const ScheduleTree*> result;
  for (auto e : vec) {
    if (e->elemAs<T>()) {
      result.push_back(e);
    }
  }
  return result;
}

template <typename T, typename Func>
T foldl(const vector<const ScheduleTree*> vec, Func op, T init = T()) {
  T value = init;
  for (auto st : vec) {
    value = op(st, value);
  }
  return value;
}

template <typename... Args>
ostream& operator<<(ostream& os, const vector<Args...>& v) {
  os << "[";
  bool first = true;
  for (auto const& ve : v) {
    if (!first) {
      os << ", ";
    }
    os << ve;
    first = true;
  }
  os << "]";
  return os;
}
} // namespace

isl::multi_union_pw_aff infixScheduleMupa(
    const ScheduleTree* root,
    const ScheduleTree* relativeRoot,
    const ScheduleTree* tree) {
  auto domainElem = root->elemAs<ScheduleTreeElemDomain>();
  TC_CHECK(domainElem);
  auto domain = domainElem->domain_.universe();
  auto zero = isl::multi_val::zero(domain.get_space().set_from_params());
  auto prefix = isl::multi_union_pw_aff(domain, zero);
  prefix = foldl(
      filterType<ScheduleTreeElemBand>(tree->ancestors(relativeRoot)),
      [](const ScheduleTree* st, isl::multi_union_pw_aff pref) {
        auto mupa = st->elemAs<ScheduleTreeElemBand>()->mupa_;
        return pref.flat_range_product(mupa);
      },
      prefix);
  return prefix;
}

isl::multi_union_pw_aff prefixScheduleMupa(
    const ScheduleTree* root,
    const ScheduleTree* tree) {
  return infixScheduleMupa(root, root, tree);
}

isl::multi_union_pw_aff partialScheduleMupa(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree) {
  auto prefix = prefixScheduleMupa(root, tree);
  auto band = tree->elemAs<ScheduleTreeElemBand>();
  return band ? prefix.flat_range_product(band->mupa_) : prefix;
}

void updateTopLevelContext(detail::ScheduleTree* root, isl::set context) {
  if (!matchOne(tc::polyhedral::domain(tc::polyhedral::context(any())), root)) {
    root->appendChild(ScheduleTree::makeContext(
        isl::set::universe(context.get_space()), root->detachChildren()));
  }
  auto contextElem = const_cast<detail::ScheduleTreeElemContext*>(
      root->child({0})->elemAs<detail::ScheduleTreeElemContext>());
  TC_CHECK(contextElem) << "Expected domain(context(any()))";
  contextElem->context_ = contextElem->context_ & context;
}

namespace {

// In a tree starting at "root", insert a sequence node with
// as only child the node identified by "tree"
// within the subtree at "relativeRoot".
ScheduleTree* insertSequenceAbove(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* tree) {
  auto parent = tree->ancestor(relativeRoot, 1);
  auto childPos = tree->positionInParent(parent);
  auto filter = activeDomainPoints(root, tree).universe();
  parent->insertChild(
      childPos,
      ScheduleTree::makeSequence(
          ScheduleTree::makeFilter(filter, parent->detachChild(childPos))));
  return parent->child({childPos});
}

} // namespace

ScheduleTree* insertSequenceAbove(ScheduleTree* root, ScheduleTree* tree) {
  return insertSequenceAbove(root, root, tree);
}

void insertSequenceBelow(
    const detail::ScheduleTree* root,
    detail::ScheduleTree* tree) {
  auto numChildren = tree->numChildren();
  TC_CHECK_LE(numChildren, 1u);
  auto filter = activeDomainPointsBelow(root, tree).universe();
  auto node = ScheduleTree::makeFilter(filter, tree->detachChildren());
  tree->appendChild(ScheduleTree::makeSequence(std::move(node)));
}

ScheduleTree* insertExtensionAbove(
    ScheduleTree* relativeRoot,
    ScheduleTree* tree,
    isl::union_map extension) {
  auto parent = tree->ancestor(relativeRoot, 1);
  auto childPos = tree->positionInParent(parent);
  auto child = parent->detachChild(childPos);
  parent->insertChild(
      childPos, ScheduleTree::makeExtension(extension, std::move(child)));
  return parent->child({childPos});
}

namespace {
/*
 * Insert an empty extension node above "st" in a tree with the given root and
 * return a pointer to the inserted extension node.
 * The modification is performed within the subtree at "relativeRoot".
 */
detail::ScheduleTree* insertEmptyExtensionAbove(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* st) {
  auto domain = root->elemAs<ScheduleTreeElemDomain>();
  TC_CHECK(domain);
  auto space = domain->domain_.get_space();
  auto extension = isl::union_map::empty(space);
  return insertExtensionAbove(relativeRoot, st, extension);
}

/*
 * Construct an extension map for a zero-dimensional statement
 * with the given identifier.
 */
isl::map labelExtension(ScheduleTree* root, ScheduleTree* tree, isl::id id) {
  auto prefix = prefixScheduleMupa(root, tree);
  auto scheduleSpace = prefix.get_space();
  auto space = scheduleSpace.params().named_set_from_params_id(id, 0);
  auto extensionSpace = scheduleSpace.map_from_domain_and_range(space);
  return isl::map::universe(extensionSpace);
}

/*
 * Construct a filter node for a zero-dimensional extension statement
 * with the given extension map.
 */
ScheduleTreeUPtr labelFilterFromExtension(isl::map extension) {
  return detail::ScheduleTree::makeFilter(extension.range());
}

/*
 * Given a sequence node in the schedule tree, insert
 * an extension with the given extension map and extension filter node
 * before the child at position "pos".
 * If "pos" is equal to the number of children, then
 * the statement is added after the last child.
 * The modification is performed within the subtree at "relativeRoot".
 */
void insertExtensionAt(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* seqNode,
    size_t pos,
    isl::union_map extension,
    ScheduleTreeUPtr&& filterNode) {
  auto extensionTree = seqNode->ancestor(relativeRoot, 1);
  auto extensionNode =
      extensionTree->elemAs<detail::ScheduleTreeElemExtension>();
  if (!extensionNode) {
    extensionTree = insertEmptyExtensionAbove(root, relativeRoot, seqNode);
    extensionNode = extensionTree->elemAs<detail::ScheduleTreeElemExtension>();
  }
  TC_CHECK(extensionNode);
  TC_CHECK(seqNode->elemAs<detail::ScheduleTreeElemSequence>());
  extensionNode->extension_ = extensionNode->extension_.unite(extension);
  seqNode->insertChild(pos, std::move(filterNode));
}
} // namespace

void insertExtensionBefore(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* tree,
    isl::union_map extension,
    ScheduleTreeUPtr&& filterNode) {
  size_t pos;
  auto parent = tree->ancestor(relativeRoot, 1);
  ScheduleTree* seqTree;
  if (tree->elemAs<detail::ScheduleTreeElemExtension>()) {
    tree = tree->child({0});
    parent = tree;
  }
  if (tree->elemAs<detail::ScheduleTreeElemSequence>()) {
    seqTree = tree;
    pos = 0;
  } else if (
      parent->elemAs<detail::ScheduleTreeElemFilter>() &&
      parent->ancestor(root, 1)->elemAs<detail::ScheduleTreeElemSequence>()) {
    seqTree = parent->ancestor(relativeRoot, 1);
    pos = parent->positionInParent(seqTree);
  } else {
    seqTree = insertSequenceAbove(root, relativeRoot, tree);
    pos = 0;
  }
  insertExtensionAt(
      root, relativeRoot, seqTree, pos, extension, std::move(filterNode));
}

void insertExtensionAfter(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* tree,
    isl::union_map extension,
    ScheduleTreeUPtr&& filterNode) {
  size_t pos;
  auto parent = tree->ancestor(relativeRoot, 1);
  ScheduleTree* seqTree;
  if (tree->elemAs<detail::ScheduleTreeElemExtension>()) {
    tree = tree->child({0});
    parent = tree;
  }
  if (tree->elemAs<detail::ScheduleTreeElemSequence>()) {
    seqTree = tree;
    pos = tree->numChildren();
  } else if (
      parent->elemAs<detail::ScheduleTreeElemFilter>() &&
      parent->ancestor(root, 1)->elemAs<detail::ScheduleTreeElemSequence>()) {
    seqTree = parent->ancestor(relativeRoot, 1);
    pos = parent->positionInParent(seqTree) + 1;
  } else {
    seqTree = insertSequenceAbove(root, relativeRoot, tree);
    pos = 1;
  }
  insertExtensionAt(
      root, relativeRoot, seqTree, pos, extension, std::move(filterNode));
}

void insertExtensionLabelAt(
    ScheduleTree* root,
    ScheduleTree* seqNode,
    size_t pos,
    isl::id id) {
  auto extension = labelExtension(root, seqNode, id);
  auto filterNode = labelFilterFromExtension(extension);
  insertExtensionAt(root, root, seqNode, pos, extension, std::move(filterNode));
}

void insertExtensionLabelBefore(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::id id) {
  auto extension = labelExtension(root, tree, id);
  auto filterNode = labelFilterFromExtension(extension);
  insertExtensionBefore(root, root, tree, extension, std::move(filterNode));
}

void insertExtensionLabelAfter(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::id id) {
  auto extension = labelExtension(root, tree, id);
  auto filterNode = labelFilterFromExtension(extension);
  insertExtensionAfter(root, root, tree, extension, std::move(filterNode));
}

namespace {

/*
 * Simplify the given tree inside the given context.
 *
 * In particular, simplify filters and the domains
 * of band node partial schedules.
 * Elements of a sequence that end up with an empty filter are removed.
 */
void gist(ScheduleTree* tree, isl::union_set context) {
  if (auto bandElem = tree->elemAs<ScheduleTreeElemBand>()) {
    bandElem->mupa_ = bandElem->mupa_.gist(context);
  } else if (auto filterElem = tree->elemAs<ScheduleTreeElemMapping>()) {
    filterElem->filter_ = filterElem->filter_.gist(context);
  } else if (auto filterElem = tree->elemAs<ScheduleTreeElemFilter>()) {
    filterElem->filter_ = filterElem->filter_.gist(context);
    if (filterElem->filter_.is_empty()) {
      tree->detachChildren();
    }
  }
  for (auto child : tree->children()) {
    gist(child, context);
  }
  if (tree->elemAs<ScheduleTreeElemSequence>()) {
    for (auto i = tree->numChildren(); i > 0; --i) {
      auto child = tree->child({i - 1});
      if (auto filterElem = child->elemAs<ScheduleTreeElemFilter>()) {
        if (filterElem->filter_.is_empty()) {
          tree->detachChild(i - 1);
        }
      }
    }
  }
}

/*
 * Create a filter node with the given filter and single child node,
 * after simplifying the child node in the context of the filter.
 */
ScheduleTreeUPtr gistedFilter(isl::union_set filter, ScheduleTreeUPtr child) {
  gist(child.get(), filter);
  return ScheduleTree::makeFilter(filter, std::move(child));
}

/*
 * Given a partition of the (active) domain elements into "first" and "second",
 * is it possible to order the "first" elements before the "second"
 * without violating any of the (active) "dependences"?
 */
bool canOrder(
    isl::union_set first,
    isl::union_set second,
    isl::union_map dependences) {
  if (first.is_empty() || second.is_empty()) {
    return true;
  }
  // Create an ordering schedule function first -> 0; second -> 1.
  auto ctx = dependences.get_ctx();
  auto space = isl::space(ctx, 0).unnamed_set_from_params(1);
  auto zero = isl::multi_val::zero(space);
  auto one = zero.set_val(0, isl::val::one(ctx));
  auto order = isl::multi_union_pw_aff(first, zero);
  order = order.union_add(isl::multi_union_pw_aff(second, one));

  // Check that this ordering preserves all dependences.
  auto preserved = dependences.lex_lt_at(order).unite(dependences.eq_at(order));
  return dependences.is_subset(preserved);
}

} // namespace

bool canOrderBefore(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::union_set filter,
    isl::union_map dependences) {
  auto other = activeDomainPoints(root, tree).subtract(filter);
  return canOrder(filter, other, dependences);
}

bool canOrderAfter(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::union_set filter,
    isl::union_map dependences) {
  auto other = activeDomainPoints(root, tree).subtract(filter);
  return canOrder(other, filter, dependences);
}

void orderBefore(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::union_set filter) {
  auto other = activeDomainPoints(root, tree).subtract(filter);
  auto seq = ScheduleTree::makeSequence(
      gistedFilter(filter, ScheduleTree::makeScheduleTree(*tree)));
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  seq->appendChild(gistedFilter(other, parent->detachChild(childPos)));
  parent->insertChild(childPos, std::move(seq));
}

void orderAfter(ScheduleTree* root, ScheduleTree* tree, isl::union_set filter) {
  auto other = activeDomainPoints(root, tree).subtract(filter);
  auto seq = ScheduleTree::makeSequence(
      gistedFilter(filter, ScheduleTree::makeScheduleTree(*tree)));
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  seq->insertChild(0, gistedFilter(other, parent->detachChild(childPos)));
  parent->insertChild(childPos, std::move(seq));
}

/*
 * Extract a mapping from the domain elements active at "tree"
 * to identifiers "ids", where all branches in "tree"
 * are assumed to have been mapped to these identifiers.
 * The result lives in a space of the form "tupleId"["ids"...].
 */
isl::multi_union_pw_aff extractDomainToIds(
    const detail::ScheduleTree* root,
    const detail::ScheduleTree* tree,
    const std::vector<mapping::MappingId>& ids,
    isl::id tupleId) {
  using namespace polyhedral::detail;

  auto space = isl::space(tree->ctx_, 0);
  auto empty = isl::union_set::empty(space);
  space = space.named_set_from_params_id(tupleId, ids.size());
  auto zero = isl::multi_val::zero(space);
  auto domainToIds = isl::multi_union_pw_aff(empty, zero);

  for (auto mapping : tree->collect(tree, ScheduleTreeType::Mapping)) {
    auto mappingNode = mapping->elemAs<ScheduleTreeElemMapping>();
    auto list = isl::union_pw_aff_list(tree->ctx_, ids.size());
    for (auto id : ids) {
      if (mappingNode->mapping.count(id) == 0) {
        break;
      }
      auto idMap = mappingNode->mapping.at(id);
      list = list.add(idMap);
    }
    // Ignore this node if it does not map to all required ids.
    if (static_cast<size_t>(list.n()) != ids.size()) {
      continue;
    }
    auto nodeToIds = isl::multi_union_pw_aff(space, list);
    auto active = activeDomainPoints(root, mapping);
    TC_CHECK(active.intersect(domainToIds.domain()).is_empty())
        << "conflicting mappings; are the filters in the tree disjoint?";
    nodeToIds = nodeToIds.intersect_domain(active);
    domainToIds = domainToIds.union_add(nodeToIds);
  }

  auto active = activeDomainPoints(root, tree);
  TC_CHECK(active.is_subset(domainToIds.domain()))
      << "not all domain points of\n"
      << active << "\nwere mapped to the required ids";

  return domainToIds;
}

} // namespace polyhedral
} // namespace tc
