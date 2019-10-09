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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tc/core/functional.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/options.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
////////////////////////////////////////////////////////////////////////////////
//                        Transformation functions, out-of-class
////////////////////////////////////////////////////////////////////////////////
// Joins 2 perfectly nested bands into a single band.
// This is a structural transformation but it is not necessarily correct
// semantically. In particular, the user is responsible for setting the
// permutability of the band since it is generally required to perform
// dependence analysis to determine permutability.
// The coincident fields corresponding to members of the inner band are cleared.
detail::ScheduleTree* joinBands(
    detail::ScheduleTree* tree,
    bool permutable = false);

// Iteratively joins perfectly nested bands into a single band.
// This is a structural transformation but it is not necessarily correct
// semantically. In particular, the user is responsible for setting the
// permutability of the band since it is generally required to perform
// dependence analysis to determine permutability.
// The coincident fields corresponding to members of inner bands are cleared.
detail::ScheduleTree* joinBandsIterative(
    detail::ScheduleTree* tree,
    bool permutable = false);

// Split tree rooted under relativeRoot into two nested trees,
// one with the first "pos" dimensions and one with the remaining dimensions.
// The schedules of the two bands live in anonymous spaces.
// This updates the current ScheduleTree and returns it so we can chain
// expressions.
detail::ScheduleTree* bandSplit(
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    size_t pos);
// Split band rooted under relativeRoot into at most three nested band
// such that the band member at position "pos" is isolated
// into a single-member band.
// The schedules of the split bands live in anonymous spaces.
// Update the current ScheduleTree and return
// a pointer to band containing the isolated member.
detail::ScheduleTree* bandSplitOut(
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    size_t pos);

// The semantics for this function is somewhat richer than the ISL C semantics.
// Since tiling is implemented as a simple band.mupa_ tranformation we can
// just complete it with 0 on the unspecified dimensions.
// This has the effect of pushing the non-tiled outer-loop inside the tile.
//   i.e. for i, j, k -> for i, j, ii, jj, k
//
// On the contrary if you want to keep the non-tiled outer-loop outside the
// tile, you can just specify tile size of 1 which, similarly to the current
// ISL behavior, will make it so.
//   i.e. for i, j, k -> for i, j, k, ii, jj, kk where range(kk)=[0, 1]
//
// This will automatically drop innermost sizes in excess of band->nMember()
//
// Modifies tree in place and returns it for call chaining purposes
//
// TODO: Support imperfectly nested tiling
detail::ScheduleTree* bandTile(
    detail::ScheduleTree* tree,
    const std::vector<size_t>& tileSizes,
    TileOptions tileOptions);

// Change the partial schedule of the band in place by multiplying it with the
// given scales.  The size of the "scales" vector must correspond to the number
// of band members.
//
// This will automatically drop innermost sizes in excess of band->nMember()
detail::ScheduleTree* bandScale(
    detail::ScheduleTree* tree,
    const std::vector<size_t>& scales);

// Insert an empty band node below "root" or below the only child of "root" if
// the child is a context node.
detail::ScheduleTree* insertTopLevelEmptyBand(detail::ScheduleTree* root);

// Update the top-level context node by intersecting it with "context".  The
// top-level context node must be located directly under the root of the tree.
// If there is no such node, insert one first.
void updateTopLevelContext(
    detail::ScheduleTree* root,
    isl::Set<Prefix> context);

// In a tree starting at "root", insert a sequence node with
// as only child the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted sequence node
// for call chaining purposes.
detail::ScheduleTree* insertSequenceAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree);

// In a tree starting at "root", insert a sequence node underneath "tree".
// "tree" is assumed to have at most one child.
//
// The tree is modified in place.
void insertSequenceBelow(
    const detail::ScheduleTree* root,
    detail::ScheduleTree* tree);

// In a tree starting at a "relativeRoot", insert an extension node with the
// given extension above the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted extension node
// for call chaining purposes.
detail::ScheduleTree* insertExtensionAbove(
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    isl::union_map extension);

// In a tree starting at a (relative) "root", insert the given node
// above the node identified by "tree".
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted node
// for call chaining purposes.
inline detail::ScheduleTree* insertNodeAbove(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node);

// Insert the given node below node "tree", which is assumed to have at
// most one child.
//
// The tree is modified in place.
// Return a non-owning pointer to the inserted node
// for call chaining purposes.
inline detail::ScheduleTree* insertNodeBelow(
    detail::ScheduleTree* tree,
    ScheduleTreeUPtr&& node);

// Insert an extension with the given extension map and extension filter node
// before node "tree".
// If "tree" is a sequence node, an extension node with a sequence child,
// or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
// The modification is performed within the subtree at "relativeRoot".
void insertExtensionBefore(
    const detail::ScheduleTree* root,
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    isl::UnionMap<Prefix, Statement> extension,
    ScheduleTreeUPtr&& filterNode);

// Insert an extension with the given extension map and extension filter node
// after node "tree".
// If "tree" is a sequence node, an extension node with a sequence child,
// or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
// The modification is performed within the subtree at "relativeRoot".
void insertExtensionAfter(
    const detail::ScheduleTree* root,
    detail::ScheduleTree* relativeRoot,
    detail::ScheduleTree* tree,
    isl::UnionMap<Prefix, Statement> extension,
    ScheduleTreeUPtr&& filterNode);

// Given a sequence node in the schedule tree, insert
// a zero-dimensional extension statement with the given identifier
// before the child at position "pos".
// If "pos" is equal to the number of children, then
// the statement is added after the last child.
void insertExtensionLabelAt(
    detail::ScheduleTree* root,
    detail::ScheduleTree* seqNode,
    size_t pos,
    isl::id id);

// Insert a zero-dimensional extension statement with the given identifier
// before node "tree".
// If "tree" is a sequence node, an extension node with a sequence child,
// or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
void insertExtensionLabelBefore(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::id id);

// Insert a zero-dimensional extension statement with the given identifier
// after node "tree".
// If "tree" is a sequence node, an extension node with a sequence child,
// or a grandchild of a sequence node,
// then the new statement is inserted in the right position
// of that sequence node.
// Otherwise, a new sequence node is inserted.
void insertExtensionLabelAfter(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::id id);

// Is it possible to order the elements in the given filter
// before the other active elements without violating
// any of the given dependences?
bool canOrderBefore(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::UnionSet<Statement> filter,
    isl::UnionMap<Statement, Statement> dependences);
// Is it possible to order the elements in the given filter
// after the other active elements without violating
// any of the given dependences?
bool canOrderAfter(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::UnionSet<Statement> filter,
    isl::UnionMap<Statement, Statement> dependences);

// Insert a sequence to ensure that the active domain elements
// in the given filter are executed before the other active domain elements.
void orderBefore(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::UnionSet<Statement> filter);
// Insert a sequence to ensure that the active domain elements
// in the given filter are executed after the other active domain elements.
void orderAfter(
    detail::ScheduleTree* root,
    detail::ScheduleTree* tree,
    isl::UnionSet<Statement> filter);

} // namespace polyhedral
} // namespace tc

#include "tc/core/polyhedral/schedule_transforms-inl.h"
