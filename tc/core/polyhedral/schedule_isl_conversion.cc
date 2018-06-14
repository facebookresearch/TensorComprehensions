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
#include "tc/core/polyhedral/schedule_isl_conversion.h"

#include <memory>
#include <numeric>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/check.h"
#include "tc/core/flags.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/external/isl.h"

using namespace std;

namespace tc {
namespace polyhedral {
namespace detail {

namespace {

isl::schedule_node insertChild(isl::schedule_node node, const ScheduleTree* st);
isl::schedule_node extendChild(isl::schedule_node node, const ScheduleTree* st);

/*
 * Convert the branching (set or sequence) node "st" along with
 * its children at positions "pos" and their descendants
 * to nodes in an isl schedule tree and insert them at position "node".
 */
isl::schedule_node insertBranch(
    isl::schedule_node node,
    const ScheduleTree* st,
    const std::vector<size_t>& pos) {
  auto filters = isl::union_set_list(node.get_ctx(), st->numChildren());
  for (size_t i = 0; i < pos.size(); ++i) {
    auto filter = st->child({pos[i]})->elemAs<ScheduleTreeElemFilter>();
    TC_CHECK(filter);
    filters = filters.add(filter->filter_);
  }
  if (st->elemAs<ScheduleTreeElemSet>()) {
    node = node.insert_set(filters);
  } else {
    node = node.insert_sequence(filters);
  }
  for (size_t i = 0; i < pos.size(); ++i) {
    node = extendChild(node.child(i), st->child({pos[i]})).parent();
  }
  return node;
}

/*
 * Convert the branching (set or sequence) node "st" along with its descendants
 * to nodes in an isl schedule tree and insert them at position "node".
 */
isl::schedule_node insertBranch(
    isl::schedule_node node,
    const ScheduleTree* st) {
  std::vector<size_t> all(st->numChildren());
  std::iota(std::begin(all), std::end(all), 0);
  return insertBranch(node, st, all);
}

/*
 * Return the positions of children of sequence node "st"
 * with filters that intersect "domain".
 */
std::vector<size_t> findCorePositions(
    const ScheduleTree* st,
    isl::union_set domain) {
  std::vector<size_t> positions;
  TC_CHECK(st->elemAs<ScheduleTreeElemSequence>());
  for (size_t i = 0; i < st->numChildren(); ++i) {
    auto filter = st->child({i})->elemAs<ScheduleTreeElemFilter>();
    TC_CHECK(filter);
    if (!filter->filter_.intersect(domain).is_empty()) {
      positions.emplace_back(i);
    }
  }
  return positions;
}

/*
 * Construct an isl schedule tree from the subtree "st",
 * with a filter node as root, for grafting into another tree.
 * The created extension is the subset of "extension" that
 * corresponds to the filter.
 */
isl::schedule_node graftFromFilterSubtree(
    const ScheduleTree* st,
    isl::union_map extension) {
  auto filter = st->elemAs<ScheduleTreeElemFilter>();
  TC_CHECK(filter);
  auto filterExtension = extension.intersect_range(filter->filter_);
  auto extensionNode = isl::schedule_node::from_extension(filterExtension);
  return extendChild(extensionNode, st);
}

/*
 * Convert the extension node "st" along with its descendants
 * to nodes in an isl schedule tree and insert them at position "node".
 * The extension node is assumed to have a sequence node as child
 * with some children (exclusively) referring
 * to the original domain elements (those that reach "node").
 * Convert this subsequence first and then graft the intermediate
 * elements after the corresponding node in the converted sequence,
 * the earlier elements before the sequence and
 * the later elements after.
 */
isl::schedule_node insertExtension(
    isl::schedule_node node,
    const ScheduleTree* st) {
  auto depth0 = node.get_tree_depth();
  auto domain = node.get_universe_domain();
  auto child = st->child({0});
  auto corePos = findCorePositions(child, domain);
  TC_CHECK(!corePos.empty());
  node = insertBranch(node, child, corePos);

  auto extension = st->elemAs<ScheduleTreeElemExtension>()->extension_;
  for (size_t i = 0; i < corePos.size() - 1; ++i) {
    auto depth0 = node.get_tree_depth();
    node = node.child(i).child(0);
    for (auto j = corePos[i] + 1; j < corePos[i + 1]; ++j) {
      auto graft = graftFromFilterSubtree(child->child({j}), extension);
      node = node.graft_after(graft);
    }
    node = node.ancestor(node.get_tree_depth() - depth0);
  }
  for (size_t i = 0; i < corePos[0]; ++i) {
    auto graft = graftFromFilterSubtree(child->child({i}), extension);
    node = node.graft_before(graft);
  }
  for (auto i = child->numChildren(); i > corePos.back() + 1; --i) {
    auto graft = graftFromFilterSubtree(child->child({i - 1}), extension);
    node = node.graft_after(graft);
  }
  node = node.ancestor(node.get_tree_depth() - depth0);
  return node;
}

/*
 * Convert the internal "st" node along with its descendants
 * to nodes in an isl schedule tree and insert them at position "node".
 * Domain nodes are not supported because they can only appear
 * in the root node.
 * Expansion nodes are not supported yet because they require
 * some extra functionality in isl.
 */
isl::schedule_node insert(isl::schedule_node node, const ScheduleTree* st) {
  if (auto band = st->elemAs<ScheduleTreeElemBand>()) {
    node = node.insert_partial_schedule(band->mupa_);
    auto bandNode = node.as<isl::schedule_node_band>();
    bandNode = bandNode.set_permutable(band->permutable_);
    for (size_t i = 0; i < band->coincident_.size(); ++i) {
      bandNode = bandNode.member_set_coincident(i, band->coincident_[i]);
    }
    for (size_t i = 0; i < band->unroll_.size(); ++i) {
      if (band->unroll_[i]) {
        bandNode =
            bandNode.member_set_ast_loop_type(i, isl::ast_loop_type::unroll);
      }
    }
    node = bandNode;
  } else if (auto context = st->elemAs<ScheduleTreeElemContext>()) {
    node = node.insert_context(context->context_);
  } else if (auto filter = st->elemAs<ScheduleTreeElemFilter>()) {
    node = node.insert_filter(filter->filter_);
  } else if (auto filter = st->elemAs<ScheduleTreeElemMapping>()) {
    node = node.insert_filter(filter->filter_);
  } else if (
      st->elemAs<ScheduleTreeElemSet>() ||
      st->elemAs<ScheduleTreeElemSequence>()) {
    return insertBranch(node, st);
  } else if (st->elemAs<ScheduleTreeElemExtension>()) {
    return insertExtension(node, st);
  } else if (st->elemAs<ScheduleTreeElemThreadSpecificMarker>()) {
    return insertChild(node, st);
  } else {
    LOG(FATAL) << "NYI: insert type: " << *st;
  }
  return extendChild(node, st);
}

/*
 * Recursively add nodes corresponding to the descendants of "st"
 * at "node".
 * If "st" does not have any children, then no descendants need to be added.
 */
isl::schedule_node insertChild(
    isl::schedule_node node,
    const ScheduleTree* st) {
  if (st->numChildren() == 0) {
    return node;
  }

  return insert(node, st->child({0}));
}

/*
 * Recursively add nodes corresponding to the descendants of "st"
 * underneath "node".
 */
isl::schedule_node extendChild(
    isl::schedule_node node,
    const ScheduleTree* st) {
  return insertChild(node.child(0), st).parent();
}
} // namespace

/*
 * Construct an isl::schedule from "root".
 * Start with the root node itself, which is assumed to be a domain node, and
 * then recursively add nodes corresponding to the descendants of "root".
 */
isl::schedule toIslSchedule(const ScheduleTree* root) {
  auto domain = root->elemAs<ScheduleTreeElemDomain>();
  TC_CHECK(domain) << "Root node should be domain node" << *root;
  auto node = isl::schedule_node::from_domain(domain->domain_);
  node = extendChild(node, root);
  return node.get_schedule();
}

namespace {

/*
 * Create a ScheduleTree of the isl schedule subtree at "node".
 *
 * Leaves are not explicitly represented in a ScheduleTree,
 * so do not visit the children of "node" if there is only one and
 * if this single child node is a leaf.
 */
std::unique_ptr<ScheduleTree> fromIslScheduleNode(isl::schedule_node node) {
  unique_ptr<ScheduleTree> res(new ScheduleTree(node.get_ctx()));
  res->elem_ = ScheduleTreeElemBase::make(node);
  res->type_ = res->elem_->type();
  auto n = node.n_children();
  if (n == 1 && node.child(0).isa<isl::schedule_node_leaf>()) {
    return res;
  }
  for (int i = 0; i < n; ++i) {
    res->appendChild(fromIslScheduleNode(node.child(i)));
  }
  return res;
}
} // namespace

std::unique_ptr<ScheduleTree> fromIslSchedule(isl::schedule schedule) {
  return fromIslScheduleNode(schedule.get_root());
}

// Note that the children of set and sequence nodes are always filters, so
// they cannot be replaced by empty trees.
bool validateSchedule(const ScheduleTree* st) {
  return *st == *fromIslSchedule(toIslSchedule(st));
}

bool validateSchedule(isl::schedule sc) {
  return validateSchedule(fromIslSchedule(sc).get());
}

} // namespace detail
} // namespace polyhedral
} // namespace tc
