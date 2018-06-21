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
#include "tc/core/functional.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/schedule_utils.h"
#include "tc/core/scope_guard.h"
#include "tc/external/isl.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//                        Transformation functions, out-of-class
////////////////////////////////////////////////////////////////////////////////

namespace tc {
namespace polyhedral {
using namespace detail;

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
  TC_CHECK(st->as<ScheduleTreeBand>());
  if (st->numChildren() != 1) {
    return st;
  }

  auto eb = st->as<ScheduleTreeBand>();
  auto ebChild = st->child({0})->as<ScheduleTreeBand>();
  if (!ebChild) {
    return st;
  }

  auto& partialSchedule = eb->mupa_;
  auto& partialScheduleChild = ebChild->mupa_;
  partialSchedule =
      partialSchedule.flat_range_product<Band>(partialScheduleChild);
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
  st->as<ScheduleTreeBand>()->permutable_ = permutable;
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
  st->as<ScheduleTreeBand>()->permutable_ = permutable;
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
  TC_CHECK(tree->as<ScheduleTreeBand>()) << "Not a band:\n" << *tree;
  auto band = tree->as<ScheduleTreeBand>();
  size_t n = band->nMember();
  TC_CHECK_LT(0u, n) << "no bands to split";
  TC_CHECK_LE(0u, pos) << "position out of bounds";
  TC_CHECK_GE(n, pos) << "position out of bounds";

  // Detach and reattach children to avoid making copies.
  auto children = tree->detachChildren();
  auto newChild = ScheduleTree::makeScheduleTree(*tree);
  newChild->appendChildren(std::move(children));
  auto newChildBand = newChild->as<ScheduleTreeBand>();
  newChildBand->drop(0, pos);

  tree->appendChild(std::move(newChild));
  band->drop(pos, n - pos);
  return tree;
}

ScheduleTree*
bandSplitOut(ScheduleTree* relativeRoot, ScheduleTree* tree, size_t pos) {
  auto band = tree->as<ScheduleTreeBand>();
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
  auto eb = st->as<ScheduleTreeBand>();
  TC_CHECK(eb) << "Not a band: " << *st;

  auto& band = *eb;
  TC_CHECK(band.permutable_) << "Can't tile a non-permutable band" << band;

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

  for (size_t i = 0; i < band.nMember(); ++i) {
    auto upa = band.mupa_.get_union_pw_aff(i);
    if (ts[i]) {
      upa = upa.scale_down(isl::val(st->ctx_, ts[i])).floor();
      if (tileOptions & TileOptions::ScaleTileLoops) {
        upa = upa.scale(isl::val(st->ctx_, ts[i]));
      }
    } else {
      upa = upa.scale(isl::val(st->ctx_, ts[i]));
    }
    band.mupa_ = band.mupa_.set_union_pw_aff(i, upa);
  }

  auto ebChild = childUPtr->as<ScheduleTreeBand>();
  TC_CHECK(ebChild) << "Not a band: " << *childUPtr;
  auto& childBand = *ebChild;
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
  auto eb = tree->as<ScheduleTreeBand>();
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

ScheduleTree* insertTopLevelEmptyBand(ScheduleTree* root) {
  auto node = root;
  if (node->numChildren() > 0 &&
      node->child({0})->as<detail::ScheduleTreeContext>()) {
    node = node->child({0});
  }
  return insertNodeBelow(node, ScheduleTree::makeEmptyBand(root));
}

void updateTopLevelContext(
    detail::ScheduleTree* root,
    isl::Set<Prefix> context) {
  if (!matchOne(tc::polyhedral::domain(tc::polyhedral::context(any())), root)) {
    root->appendChild(
        ScheduleTree::makeContext(context, root->detachChildren()));
  }
  auto contextElem = root->child({0})->as<detail::ScheduleTreeContext>();
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
  auto domain = root->as<ScheduleTreeDomain>();
  TC_CHECK(domain);
  auto space = domain->domain_.get_space();
  auto extension = isl::union_map::empty(space);
  return insertExtensionAbove(relativeRoot, st, extension);
}

/*
 * Construct an extension map for a zero-dimensional statement
 * with the given identifier.
 */
isl::Map<Prefix, Statement>
labelExtension(ScheduleTree* root, ScheduleTree* tree, isl::id id) {
  auto prefix = prefixScheduleMupa<Prefix>(root, tree);
  auto scheduleSpace = prefix.get_space();
  auto space = scheduleSpace.params().add_named_tuple_id_ui<Statement>(id, 0);
  auto extensionSpace = scheduleSpace.map_from_domain_and_range(space);
  return isl::Map<Prefix, Statement>::universe(extensionSpace);
}

/*
 * Construct a filter node for a zero-dimensional extension statement
 * with the given extension map.
 */
ScheduleTreeUPtr labelFilterFromExtension(
    isl::Map<Prefix, Statement> extension) {
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
    isl::UnionMap<Prefix, Statement> extension,
    ScheduleTreeUPtr&& filterNode) {
  auto extensionTree = seqNode->ancestor(relativeRoot, 1);
  auto extensionNode = extensionTree->as<detail::ScheduleTreeExtension>();
  if (!extensionNode) {
    extensionTree = insertEmptyExtensionAbove(root, relativeRoot, seqNode);
    extensionNode = extensionTree->as<detail::ScheduleTreeExtension>();
  }
  TC_CHECK(extensionNode);
  TC_CHECK(seqNode->as<detail::ScheduleTreeSequence>());
  extensionNode->extension_ = extensionNode->extension_.unite(extension);
  seqNode->insertChild(pos, std::move(filterNode));
}
} // namespace

void insertExtensionBefore(
    const ScheduleTree* root,
    ScheduleTree* relativeRoot,
    ScheduleTree* tree,
    isl::UnionMap<Prefix, Statement> extension,
    ScheduleTreeUPtr&& filterNode) {
  size_t pos;
  auto parent = tree->ancestor(relativeRoot, 1);
  ScheduleTree* seqTree;
  if (tree->as<detail::ScheduleTreeExtension>()) {
    tree = tree->child({0});
    parent = tree;
  }
  if (tree->as<detail::ScheduleTreeSequence>()) {
    seqTree = tree;
    pos = 0;
  } else if (
      parent->as<detail::ScheduleTreeFilter>() &&
      parent->ancestor(root, 1)->as<detail::ScheduleTreeSequence>()) {
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
    isl::UnionMap<Prefix, Statement> extension,
    ScheduleTreeUPtr&& filterNode) {
  size_t pos;
  auto parent = tree->ancestor(relativeRoot, 1);
  ScheduleTree* seqTree;
  if (tree->as<detail::ScheduleTreeExtension>()) {
    tree = tree->child({0});
    parent = tree;
  }
  if (tree->as<detail::ScheduleTreeSequence>()) {
    seqTree = tree;
    pos = tree->numChildren();
  } else if (
      parent->as<detail::ScheduleTreeFilter>() &&
      parent->ancestor(root, 1)->as<detail::ScheduleTreeSequence>()) {
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
void gist(ScheduleTree* tree, isl::UnionSet<Statement> context) {
  if (auto bandElem = tree->as<ScheduleTreeBand>()) {
    bandElem->mupa_ = bandElem->mupa_.gist(context);
  } else if (auto filterElem = tree->as<ScheduleTreeMapping>()) {
    filterElem->filter_ = filterElem->filter_.gist(context);
  } else if (auto filterElem = tree->as<ScheduleTreeFilter>()) {
    filterElem->filter_ = filterElem->filter_.gist(context);
    if (filterElem->filter_.is_empty()) {
      tree->detachChildren();
    }
  }
  for (auto child : tree->children()) {
    gist(child, context);
  }
  if (tree->as<ScheduleTreeSequence>()) {
    for (auto i = tree->numChildren(); i > 0; --i) {
      auto child = tree->child({i - 1});
      if (auto filterElem = child->as<ScheduleTreeFilter>()) {
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
ScheduleTreeUPtr gistedFilter(
    isl::UnionSet<Statement> filter,
    ScheduleTreeUPtr child) {
  gist(child.get(), filter);
  return ScheduleTree::makeFilter(filter, std::move(child));
}

/*
 * Given a partition of the (active) domain elements into "first" and "second",
 * is it possible to order the "first" elements before the "second"
 * without violating any of the (active) "dependences"?
 */
bool canOrder(
    isl::UnionSet<Statement> first,
    isl::UnionSet<Statement> second,
    isl::UnionMap<Statement, Statement> dependences) {
  if (first.is_empty() || second.is_empty()) {
    return true;
  }
  // Create an ordering schedule function first -> 0; second -> 1.
  auto ctx = dependences.get_ctx();
  auto space = isl::Space<>(ctx, 0).add_unnamed_tuple_ui<isl::Anonymous>(1);
  auto zero = isl::MultiVal<isl::Anonymous>::zero(space);
  auto one = zero.set_val(0, isl::val::one(ctx));
  auto order = isl::MultiUnionPwAff<Statement, isl::Anonymous>(first, zero);
  order = order.union_add(
      isl::MultiUnionPwAff<Statement, isl::Anonymous>(second, one));

  // Check that this ordering preserves all dependences.
  auto preserved = dependences.lex_lt_at(order).unite(dependences.eq_at(order));
  return dependences.is_subset(preserved);
}

} // namespace

bool canOrderBefore(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::UnionSet<Statement> filter,
    isl::UnionMap<Statement, Statement> dependences) {
  auto active = activeDomainPoints(root, tree);
  auto other = active.subtract(filter);
  return canOrder(filter, other, dependences);
}

bool canOrderAfter(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::UnionSet<Statement> filter,
    isl::UnionMap<Statement, Statement> dependences) {
  auto active = activeDomainPoints(root, tree);
  auto other = active.subtract(filter);
  return canOrder(other, filter, dependences);
}

void orderBefore(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::UnionSet<Statement> filter) {
  auto active = activeDomainPoints(root, tree);
  auto other = active.subtract(filter);
  auto seq = ScheduleTree::makeSequence(
      gistedFilter(filter, ScheduleTree::makeScheduleTree(*tree)));
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  seq->appendChild(gistedFilter(other, parent->detachChild(childPos)));
  parent->insertChild(childPos, std::move(seq));
}

void orderAfter(
    ScheduleTree* root,
    ScheduleTree* tree,
    isl::UnionSet<Statement> filter) {
  auto active = activeDomainPoints(root, tree);
  auto other = active.subtract(filter);
  auto seq = ScheduleTree::makeSequence(
      gistedFilter(filter, ScheduleTree::makeScheduleTree(*tree)));
  auto parent = tree->ancestor(root, 1);
  auto childPos = tree->positionInParent(parent);
  seq->insertChild(0, gistedFilter(other, parent->detachChild(childPos)));
  parent->insertChild(childPos, std::move(seq));
}
} // namespace polyhedral
} // namespace tc
