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

#include "tc/core/flags.h"
#include "tc/external/isl.h"

using namespace std;

namespace tc {
namespace polyhedral {
namespace detail {

// static constexpr std::initializer_list need both definition and declaration
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemExtension::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemSequence::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemSet::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemDomain::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemFilter::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemMappingFilter::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemBand::NodeDerivedTypes;
constexpr std::initializer_list<detail::ScheduleTreeType>
    ScheduleTreeElemContext::NodeDerivedTypes;

namespace {

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
    auto filter = st->child({pos[i]})->elemAsBase<ScheduleTreeElemFilter>();
    CHECK(filter);
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
  CHECK(st->elemAs<ScheduleTreeElemSequence>());
  for (size_t i = 0; i < st->numChildren(); ++i) {
    auto filter = st->child({i})->elemAsBase<ScheduleTreeElemFilter>();
    CHECK(filter);
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
  auto filter = st->elemAsBase<ScheduleTreeElemFilter>();
  CHECK(filter);
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
  CHECK(!corePos.empty());
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
  for (auto i = corePos[0]; i >= 1; --i) {
    auto graft = graftFromFilterSubtree(child->child({i - 1}), extension);
    node = node.graft_before(graft);
  }
  for (auto i = corePos.back() + 1; i < child->numChildren(); ++i) {
    auto graft = graftFromFilterSubtree(child->child({i}), extension);
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
    for (int i = 0; i < band->coincident_.size(); ++i) {
      bandNode = bandNode.member_set_coincident(i, band->coincident_[i]);
    }
    for (int i = 0; i < band->unroll_.size(); ++i) {
      if (band->unroll_[i]) {
        bandNode =
            bandNode.member_set_ast_loop_type(i, isl::ast_loop_type::unroll);
      }
    }
    node = bandNode;
  } else if (auto context = st->elemAs<ScheduleTreeElemContext>()) {
    node = node.insert_context(context->context_);
  } else if (auto filter = st->elemAsBase<ScheduleTreeElemFilter>()) {
    node = node.insert_filter(filter->filter_);
  } else if (
      st->elemAs<ScheduleTreeElemSet>() ||
      st->elemAs<ScheduleTreeElemSequence>()) {
    return insertBranch(node, st);
  } else if (st->elemAs<ScheduleTreeElemExtension>()) {
    return insertExtension(node, st);
  } else {
    LOG(FATAL) << "NYI: insert type: " << *st;
  }
  return extendChild(node, st);
}

/*
 * Recursively add nodes corresponding to the descendants of "st"
 * underneath "node".
 * If "st" does not have any children, then no descendants need to be added.
 */
isl::schedule_node extendChild(
    isl::schedule_node node,
    const ScheduleTree* st) {
  if (st->numChildren() == 0) {
    return node;
  }

  return insert(node.child(0), st->child({0})).parent();
}
} // namespace

/*
 * Construct an isl::schedule from "root".
 * Start with the root node itself, which is assumed to be a domain node, and
 * then recursively add nodes corresponding to the descendants of "root".
 */
isl::schedule toIslSchedule(const ScheduleTree* root) {
  checkValidIslSchedule(root);
  auto domain = root->elemAs<ScheduleTreeElemDomain>();
  CHECK(domain) << "Root node should be domain node" << *root;
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
  for (size_t i = 0; i < n; ++i) {
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

namespace {

// Get the parametric space in which the relation contained by the given tree
// are defined.
isl::space definitionParamSpace(const ScheduleTree* node) {
  auto space = isl::space(node->ctx_, 0);
  switch (node->type_) {
    // mapping_filter is a filter for ISL
    // TODO: this switch is too ISL-ish and is not meant for subtyping
    // extensions. Replace by if (node->elemAsBase<T>(...))
    case detail::ScheduleTreeType::MappingFilter:
    case detail::ScheduleTreeType::Filter: {
      auto filterNode = node->elemAsBase<ScheduleTreeElemFilter>();
      space = filterNode->filter_.get_space().params();
      break;
    }
    case detail::ScheduleTreeType::Band: {
      auto bandNode = node->elemAs<ScheduleTreeElemBand>();
      space = bandNode->mupa_.get_space().params();
      break;
    }
    case detail::ScheduleTreeType::Extension: {
      auto extensionNode = node->elemAs<ScheduleTreeElemExtension>();
      space = extensionNode->extension_.get_space().params();
      break;
    }
    case detail::ScheduleTreeType::Domain: {
      auto domainElem = node->elemAs<ScheduleTreeElemDomain>();
      space = domainElem->domain_.get_space().params();
      break;
    }
    case detail::ScheduleTreeType::Context: {
      auto contextElem = node->elemAs<ScheduleTreeElemContext>();
      space = contextElem->context_.get_space().params();
      break;
    }

    // Other types of nodes do not have any potentially parametric expression.
    case detail::ScheduleTreeType::Any:
    case detail::ScheduleTreeType::None:
    case detail::ScheduleTreeType::Set:
    case detail::ScheduleTreeType::Sequence:
      break;
  }
  return space;
}

bool refersToUndefinedParameters(
    const ScheduleTree* relativeRoot,
    const ScheduleTree* node) {
  using namespace polyhedral::detail;

  // Assuming no parameters are introduced above root.
  if (node == relativeRoot) {
    return false;
  }

  // Domain and context can introduce new parameters.
  if (node->elemAs<ScheduleTreeElemDomain>() ||
      node->elemAs<ScheduleTreeElemContext>()) {
    return false;
  }

  // Collect all ancestors that are allowed to introduce new parameters, i.e.
  // domain and context nodes.  Collect the parameters they introduce in a
  // space.
  auto paramSpace = isl::null<isl::space>();
  for (auto anc : node->ancestors(relativeRoot)) {
    auto contextNode = anc->elemAs<ScheduleTreeElemContext>();
    auto domainNode = anc->elemAs<ScheduleTreeElemDomain>();
    if (!contextNode && !domainNode) {
      continue;
    }

    auto space = contextNode ? contextNode->context_.get_space()
                             : domainNode->domain_.get_space();
    space = space.params();
    paramSpace = paramSpace.get() ? paramSpace.align_params(space) : space;
  }

  CHECK(paramSpace.get()) << "no parent context or domain node found";
  if (!paramSpace.get()) {
    return true;
  }

  // The space in which tree's relations are defined should not involve
  // parameters that were not present in its domain or context ancestors. If
  // the tree has no relations, it cannot refer to undefined parameters.
  auto space = definitionParamSpace(node);
  if (space.dim(isl::dim_type::param) == 0) {
    return false;
  }

  // If space uses some parameters that are not avaialbe in paramSpace,
  // they will be introduced into paramSpace, making its dimension larger.
  int definedParams = paramSpace.dim(isl::dim_type::param);
  paramSpace = paramSpace.align_params(space);
  return paramSpace.dim(isl::dim_type::param) > definedParams;
}

// Note that this uses union_set as "optional" where nullptr value means
// there's no actual value, rather than an error.
isl::union_set nodeDomain(const ScheduleTree* node) {
  if (auto domainElem = node->elemAs<ScheduleTreeElemDomain>()) {
    return domainElem->domain_;
  } else if (auto bandElem = node->elemAs<ScheduleTreeElemBand>()) {
    return bandElem->mupa_.domain();
  } else if (auto filterElem = node->elemAsBase<ScheduleTreeElemFilter>()) {
    return filterElem->filter_;
  } else if (auto extensionElem = node->elemAs<ScheduleTreeElemExtension>()) {
    // FIXME: these are the points that are _introduced_...  they are inactive
    // until now...
    // does the extension have a domain?
    return extensionElem->extension_
        .range(); // FIXME: do we need to restrict its domain first?
  }
  return isl::null<isl::union_set>();
}
} // namespace

void checkValidIslSchedule(const ScheduleTree* root_) {
  using namespace polyhedral::detail;

  // 1. The root node is always of type domain or extension.
  auto domainRoot = root_->elemAs<ScheduleTreeElemDomain>();
  auto extensionRoot = root_->elemAs<ScheduleTreeElemDomain>();
  CHECK(domainRoot || extensionRoot)
      << "root must be a domain or an extension" << *root_;

  for (auto node : ScheduleTree::collect(root_)) {
    auto activeInstances = activeDomainPoints(root_, node);
    auto nodeDomainPoints = nodeDomain(node);

    // 2. Only set or sequence nodes are allowed to have multiple children and
    // these children must be filter nodes.
    auto nodeIsSet = node->elemAs<ScheduleTreeElemSet>() != nullptr;
    auto nodeIsSequence = node->elemAs<ScheduleTreeElemSequence>() != nullptr;
    auto nChildren = node->numChildren();

    // 4. Nodes should not refer to inactive domain points.
    if (nodeDomainPoints.get()) {
      if (!nodeDomainPoints.is_subset(activeInstances)) {
        LOG_IF(WARNING, tc::FLAGS_schedule_tree_verbose_validation)
            << "node refers to inactive domain points: active "
            << activeInstances << " found: " << nodeDomainPoints << " in\n"
            << *node;
      }
    }

    if (!nodeIsSet && !nodeIsSequence) {
      CHECK_LE(nChildren, 1)
          << "only sequence or set nodes can have multiple children" << *node;
    } else {
      auto filters = isl::null<isl::union_set>();
      for (auto child : node->children()) {
        auto filterElem = child->elemAsBase<ScheduleTreeElemFilter>();
        auto childIsFilter = filterElem != nullptr;
        CHECK(childIsFilter)
            << "only filter nodes allowed as children of sequence or set node"
            << *child;

        filters = filters.get() ? filters.unite(filterElem->filter_)
                                : filterElem->filter_;
      }
      CHECK(filters.get()) << "set/sequence node must have at least one child"
                           << *node;

      // 5. The union of filters must cover all active domain points.
      CHECK(activeInstances.is_subset(filters))
          << "filters must cover all active domain points; active "
          << activeInstances << " filtered: " << filters << " in\n"
          << *node;
    }

    // 3. Nodes should not refer to parameters that are not declared in context
    // nodes above.
    bool usesUndefinedParams = refersToUndefinedParameters(root_, node);
    CHECK(!usesUndefinedParams)
        << "non-context node introduces new parameters" << *node;

    // 7. Band schedules should be total on all active domain points.
    // Only check if an actual domain is specified.
    // In particular, 0D bands do not necessarily specify a domain.
    if (auto bandElem = node->elemAs<ScheduleTreeElemBand>()) {
      auto scheduleDefinitionDomain = bandElem->mupa_.domain();
      if (!scheduleDefinitionDomain.is_params()) {
        CHECK(activeInstances.is_subset(scheduleDefinitionDomain))
            << "schedule should be total on the active domain points: active"
            << activeInstances
            << " schedule defined over: " << scheduleDefinitionDomain << " in\n"
            << *node;
      }
    }

    // 8. Extension nodes should not introduce any elements that are already
    // active domain elements.
    // 10. Anchored nodes match the flattened space of the outer bands.
    if (auto extensionElem = node->elemAs<ScheduleTreeElemExtension>()) {
      auto introducedInstances =
          extensionElem->extension_
              .range(); // FIXME: restrict domain to the partial schedule??
      CHECK(introducedInstances.intersect(activeInstances).is_empty())
          << "extension node should not introduce elements that are already "
             "active domain elements: active "
          << activeInstances << " introduced: " << introducedInstances
          << " in\n"
          << *node;
      // FIXME: we probably want to check dim_ids for an exact match
      auto depth = node->scheduleDepth(root_);
      auto extension = extensionElem->extension_;
      for (auto const& e : isl::UnionAsVector<isl::union_map>(extension)) {
        CHECK_EQ(depth, e.dim(isl::dim_type::in))
            << "the input dimensionality of the extension map should "
               "correspond to the schedule depth"
            << *node;
      }
    }

    // 10. Anchored nodes match the flattened space of the outer bands.
    if (auto contextElem = node->elemAs<ScheduleTreeElemContext>()) {
      auto depth = node->scheduleDepth(root_);
      CHECK_EQ(depth, contextElem->context_.dim(isl::dim_type::set))
          << "the dimensionality of the context should correspond "
             "to the schedule depth"
          << *node;
    }
  }
}
} // namespace detail
} // namespace polyhedral
} // namespace tc
