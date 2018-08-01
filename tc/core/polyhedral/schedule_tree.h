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

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tc/core/check.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/mapping_types.h"
#include "tc/core/polyhedral/options.h"
#include "tc/core/utils/vararg.h"
#include "tc/external/isl.h"

#include "glog/logging.h"

namespace tc {
namespace polyhedral {
namespace detail {

// Internal representation of a polyhedral schedule information, wrapping a
// ScheduleTree, convertible to and from isl::schedule.
//
struct ScheduleTree;
struct ScheduleTreeSet;
struct ScheduleTreeSequence;
struct ScheduleTreeMapping;

enum class ScheduleTreeType {
  None,
  Band,
  Context,
  Domain,
  Extension,
  Filter,
  Sequence,
  Set,
  Mapping,
  ThreadSpecificMarker,
  Any,
};

} // namespace detail

using ScheduleTreeUPtr = std::unique_ptr<detail::ScheduleTree>;

namespace detail {

//////////////////////////////////////////////////////////////////////////////
//
// Schedule Trees
//
// Memory model: tree uniquely owns its children, user owns the root,
// traversals are non-owning.
//
// ScheduleTree is a data store of the ScheduleXYZ API.  It implements a
// mutable tree data structure, each ScheduleTree having a potentially empty
// list of children, with the following ownership semantics.  A ScheduleTree
// object owns its children.  When a child is added to or removed from the
// tree, the ownership is transferred from or to the caller.  Users of
// ScheduleTree own the root of the tree and may apply any ownership policies,
// e.g. share the ownership.  Users are guaranteed that, if they own a
// ScheduleTree, it is a root of a tree.  Ownership rules are enforced through
// unique_ptr and move semantics.  In particular, users are not expected to
// manipulate ScheduleTree objects by value.
//
// New ScheduleTree objects of various types or deep copies of the existing
// objects can be constructed using static factory functions, which transfer
// the ownership of the constructed object to the caller.  These functions
// optionally take a list of subtrees that will become children of the newly
// constructed tree, which takes ownership.
//
// Tree structure can be changed by appending, inserting, detaching or swapping
// the subtrees.  Only trees owned by the user can be attached, inserted or
// swapped with, in which case the ownership is transferred to the parent tree
// object.  The user is expected to own the root of the tree and the inserted
// tree, but not the insertion point. The ownership of the detached or swapped
// tree is transferred to the caller.
//
// ScheduleTrees are not supposed to have null children, which is checked in
// the construction/child manipulation in debug builds.
//
//
// Internal structure: single-linked tree (no parent pointer).
//
// Because the caller must own the root of the tree, it is always possible to
// find the parent or any ancestor tree by traversing the tree from the root.
// Subtrees are ordered and are identified by their position in the parent
// tree.
//
// Trees can be traversed, inspected and modified through raw non-owning
// pointers.  PreorderDFS, PostorderDFS and BFS traversals are provided.  Tree
// modification is in place and does not require the caller to own the
// ScheduleTree object.
//
// Tree modification functions are external to the ScheduleTree class and
// should only rely on the exposed API to avoid breaking the ownership and
// non-null guarantees.  For the sake of consistency, modification functions
// should take a raw pointer to the root of the tree as the first argument,
// even if they do not use it, and a raw pointer to the subtree being
// manipulated.  Transformation functions should account for the root pointer
// being relative, i.e. not being the actual root pointer owned by the caller,
// but rather some ancestor of the given node, above which the transformation
// has no effect (think of C++ standard library with begin/end iterators).
//
//
// Well-formedness guarantees: non-null subtrees.
//
// ScheduleTree does NOT impose any structure requirements on the tree, e.g.
// those of ISL.  A tree with a null child is ill-formed.
//////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, const ScheduleTree& tree);

template <typename T>
void flattenSequenceOrSet(ScheduleTree* tree);

struct ScheduleTree {
  friend std::ostream& tc::polyhedral::detail::operator<<(
      std::ostream&,
      const tc::polyhedral::detail::ScheduleTree&);

 private:
  ScheduleTree() = delete;

 protected:
  ScheduleTree(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children,
      detail::ScheduleTreeType type)
      : ctx_(ctx), type_(type) {
    appendChildren(std::move(children));
  }

  // Copy constructor for internal use only.
  // Note that this does not account for a specific subclass of ScheduleTree.
  // All callers should use makeScheduleTree(const ScheduleTree&) instead,
  // which dispatches the copying to subclasses as well as keeps the memory
  // management scheme consistent.
  ScheduleTree(const ScheduleTree& st);

 public:
  virtual ~ScheduleTree();

  bool operator==(const ScheduleTree& other) const;
  bool operator!=(const ScheduleTree& other) const {
    return !(*this == other);
  }

  // Swap a tree with with the given tree.
  void swapChild(size_t pos, ScheduleTreeUPtr& swappee) {
    TC_CHECK_GE(pos, 0u) << "position out of children bounds";
    TC_CHECK_LE(pos, children_.size()) << "position out of children bounds";
    TC_CHECK(swappee.get()) << "Cannot swap in a null tree";
    std::swap(children_[pos], swappee);
  }

  // Child accessors (only in-place modification allowed)
  ScheduleTree* child(const std::vector<size_t>& positions);
  const ScheduleTree* child(const std::vector<size_t>& positions) const;
  size_t numChildren() const {
    return children_.size();
  };

  // Manipulators for the list of children.
  void insertChildren(size_t pos, std::vector<ScheduleTreeUPtr>&& children) {
    TC_CHECK_GE(pos, 0u) << "position out of children bounds";
    TC_CHECK_LE(pos, children_.size()) << "position out of children bounds";
    for (const auto& c : children) {
      TC_CHECK(c.get()) << "inserting null or moved-from child";
    }

    children_.insert(
        children_.begin() + pos,
        std::make_move_iterator(children.begin()),
        std::make_move_iterator(children.end()));
  }

  void insertChild(size_t pos, ScheduleTreeUPtr&& child) {
    // One cannot move from an initializer_list, so need an actual temporary
    // object here.
    insertChildren(pos, vectorFromArgs(std::move(child)));
  }

  void appendChildren(std::vector<ScheduleTreeUPtr>&& children) {
    insertChildren(children_.size(), std::move(children));
  }

  void appendChild(ScheduleTreeUPtr&& child) {
    insertChild(children_.size(), std::move(child));
  }

  ScheduleTreeUPtr detachChild(size_t pos) {
    TC_CHECK_GE(pos, 0u) << "position out of children bounds";
    TC_CHECK_LT(pos, children_.size()) << "position out of children bounds";

    ScheduleTreeUPtr child = std::move(children_[pos]);
    children_.erase(children_.begin() + pos);
    return child;
  }

  std::vector<ScheduleTreeUPtr> detachChildren() {
    std::vector<ScheduleTreeUPtr> tmpChildren;
    std::swap(tmpChildren, children_);
    return tmpChildren;
  }

  std::vector<ScheduleTreeUPtr> replaceChildren(
      std::vector<ScheduleTreeUPtr>&& children) {
    auto oldChildren = detachChildren();
    appendChildren(std::move(children));
    return oldChildren;
  }

  ScheduleTreeUPtr replaceChild(size_t pos, ScheduleTreeUPtr&& child) {
    TC_CHECK_GE(pos, 0u) << "position out of children bounds";
    TC_CHECK_LT(pos, children_.size()) << "position out of children bounds";

    ScheduleTreeUPtr oldChild = std::move(children_[pos]);
    children_[pos] = std::move(child);
    return oldChild;
  }

  // Helper to avoid calling collect + filter for this common case
  std::vector<ScheduleTree*> children() {
    std::vector<ScheduleTree*> res;
    res.reserve(children_.size());
    for (auto& p : children_) {
      res.push_back(p.get());
    }
    return res;
  };
  std::vector<const ScheduleTree*> children() const {
    std::vector<const ScheduleTree*> res;
    res.reserve(children_.size());
    for (const auto& p : children_) {
      res.push_back(p.get());
    }
    return res;
  };

  ScheduleTree* ancestor(ScheduleTree* relativeRoot, size_t generation);
  const ScheduleTree* ancestor(
      const ScheduleTree* relativeRoot,
      size_t generation) const;
  // Returns the ancestors up to relativeRoot in a vector. The first element
  // of the result is relativeRoot, the last element of the result is the
  // father of the "this" ScheduleTree.
  // If relativeRoot is equal to "this" ScheduleTree, then the result is empty.
  std::vector<ScheduleTree*> ancestors(ScheduleTree* relativeRoot);
  std::vector<const ScheduleTree*> ancestors(
      const ScheduleTree* relativeRoot) const;

  std::vector<size_t> positionRelativeTo(
      const ScheduleTree* relativeRoot) const;

  inline size_t positionInParent(const ScheduleTree* parent) const {
    auto p = positionRelativeTo(parent);
    TC_CHECK_EQ(1u, p.size()) << *parent << " is not the parent of " << *this;
    return p[0];
  }

  size_t scheduleDepth(const ScheduleTree* relativeRoot) const;

  //
  // Factory functions
  //
  static ScheduleTreeUPtr makeBand(
      isl::MultiUnionPwAff<Statement, Band> mupa,
      std::vector<ScheduleTreeUPtr>&& children = {});

  // Return a zero-dimensional band for use in a tree with the given root.
  static ScheduleTreeUPtr makeEmptyBand(const ScheduleTree* root);

  static ScheduleTreeUPtr makeDomain(
      isl::union_set domain,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeContext(
      isl::Set<Prefix> context,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeFilter(
      isl::union_set filter,
      std::vector<ScheduleTreeUPtr>&& children = {});

  template <typename MappingIdType>
  static inline ScheduleTreeUPtr makeMapping(
      const std::vector<MappingIdType>& mappedIds,
      isl::UnionPwAffListOn<Statement> mappedAffs,
      std::vector<ScheduleTreeUPtr>&& children = {}) {
    static_assert(
        std::is_base_of<mapping::MappingId, MappingIdType>::value,
        "can only map to ids that are subclasses of MappingId");
    std::vector<mapping::MappingId> ids;
    ids.reserve(mappedIds.size());
    for (const auto& id : mappedIds) {
      ids.push_back(id);
    }
    return makeMappingUnsafe(ids, mappedAffs, std::move(children));
  }

 private:
  // Internal type-unsafe function to construct mappings.
  static ScheduleTreeUPtr makeMappingUnsafe(
      const std::vector<mapping::MappingId>& mappedIds,
      isl::UnionPwAffListOn<Statement> mappedAffs,
      std::vector<ScheduleTreeUPtr>&& children);

 public:
  static ScheduleTreeUPtr makeExtension(
      isl::union_map extension,
      std::vector<ScheduleTreeUPtr>&& children = {});

  static ScheduleTreeUPtr makeThreadSpecificMarker(
      isl::ctx ctx,
      std::vector<ScheduleTreeUPtr>&& children = {});

  template <typename... Args>
  static ScheduleTreeUPtr makeBand(
      isl::MultiUnionPwAff<Statement, Band> mupa,
      Args&&... args) {
    return makeBand(
        mupa, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeDomain(isl::union_set domain, Args&&... args) {
    return makeDomain(
        domain, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeContext(isl::Set<> context, Args&&... args) {
    return makeContext(
        context, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeFilter(isl::union_set filter, Args&&... args) {
    return makeFilter(
        filter, vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename MappingIdType, typename... Args>
  static inline ScheduleTreeUPtr makeMapping(
      isl::union_set filter,
      const std::unordered_set<MappingIdType, typename MappingIdType::Hash>&
          mappingIds,
      Args&&... args) {
    return makeMapping(
        filter,
        mappingIds,
        vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeExtension(
      isl::union_map extension,
      Args&&... args) {
    return makeExtension(
        extension,
        vectorFromArgs<ScheduleTreeUPtr>(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeSet(Args&&... args) {
    return fromList<ScheduleTreeSet>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static ScheduleTreeUPtr makeSequence(Args&&... args) {
    return fromList<ScheduleTreeSequence>(std::forward<Args>(args)...);
  }

  // disallow empty lists in syntax
  template <typename T, typename Arg, typename... Args>
  static ScheduleTreeUPtr fromList(Arg&& arg, Args&&... args) {
    static_assert(
        std::is_same<ScheduleTreeSet, T>::value ||
            std::is_same<ScheduleTreeSequence, T>::value,
        "Can only construct Set or Sequence elements");
    static_assert(
        std::is_same<
            typename std::remove_reference<Arg>::type,
            ScheduleTreeUPtr>::value,
        "Arguments must be rvalue references to ScheduleTreeUPtr");

    auto ctx = arg->ctx_;
    auto res = T::make(ctx);
    flattenSequenceOrSet(res.get());
    res->appendChildren(
        vectorFromArgs(std::forward<Arg>(arg), std::forward<Args>(args)...));
    return res;
  }

  // Make a (deep) copy of "tree".
  static ScheduleTreeUPtr makeScheduleTree(const ScheduleTree& tree);

  // Collect the nodes of "tree" in some arbitrary order.
  template <typename T>
  static std::vector<T> collect(T tree) {
    return collectDFSPreorder(tree);
  }
  // Collect the nodes of "tree" of the given type in some arbitrary order.
  template <typename T>
  static std::vector<T> collect(T tree, detail::ScheduleTreeType type) {
    return collectDFSPreorder(tree, type);
  }

  static std::vector<ScheduleTree*> collectDFSPostorder(ScheduleTree* tree);
  static std::vector<ScheduleTree*> collectDFSPreorder(ScheduleTree* tree);
  static std::vector<ScheduleTree*> collectDFSPostorder(
      ScheduleTree* tree,
      detail::ScheduleTreeType type);
  static std::vector<ScheduleTree*> collectDFSPreorder(
      ScheduleTree* tree,
      detail::ScheduleTreeType type);

  static std::vector<const ScheduleTree*> collectDFSPostorder(
      const ScheduleTree* tree);
  static std::vector<const ScheduleTree*> collectDFSPreorder(
      const ScheduleTree* tree);
  static std::vector<const ScheduleTree*> collectDFSPostorder(
      const ScheduleTree* tree,
      detail::ScheduleTreeType type);
  static std::vector<const ScheduleTree*> collectDFSPreorder(
      const ScheduleTree* tree,
      detail::ScheduleTreeType type);

  // View this tree node as the specified type.
  // Returns nullptr if this is not the proper type.
  // Inline impl for now, does not justify an extra -inl.h file
  template <typename T>
  T* as() {
    const ScheduleTree* t = this;
    return const_cast<T*>(t->as<const T>());
  }
  template <typename T>
  const T* as() const {
    static_assert(
        std::is_base_of<ScheduleTree, T>::value,
        "Must call with a class derived from ScheduleTree");
    if (type_ != T::NodeType) {
      return nullptr;
    }
    return static_cast<const T*>(this);
  }

  // Write the textual representation of the node to "os".
  // Note that this function does _not_ output the child trees.
  virtual std::ostream& write(std::ostream& os) const = 0;

  // Clone the current node.
  // Note that this function does _not_ clone the child trees.
  virtual ScheduleTreeUPtr clone() const = 0;

  //
  // Data members
  //
 public:
  mutable isl::ctx ctx_;

 private:
  std::vector<ScheduleTreeUPtr> children_{};

 public:
  const detail::ScheduleTreeType type_{detail::ScheduleTreeType::None};
};

// Flatten nested nodes of the same type.
template <typename T>
inline void flattenSequenceOrSet(T* tree) {
  static_assert(
      std::is_same<ScheduleTreeSet, T>::value ||
          std::is_same<ScheduleTreeSequence, T>::value,
      "Only Sequence or Set node can be flattened");

  // Iterate over the changing list of children. If a child has the same list
  // type as a parent, replace it with grandchildren and traverse them too.
  for (size_t i = 0; i < tree->numChildren(); ++i) {
    if (tree->child({i})->type_ != tree->type_) {
      continue;
    }
    auto grandChildren = tree->child({i})->detachChildren();
    tree->detachChild(i);
    tree->insertChildren(i, std::move(grandChildren));
    --i;
  }
}

} // namespace detail
} // namespace polyhedral
} // namespace tc
