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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/core/check.h"
#include "tc/core/constants.h"
#include "tc/core/halide2isl.h"
#include "tc/core/mapping_options.h"
#include "tc/core/polyhedral/body.h"
#include "tc/core/polyhedral/domain_types.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"
#include "tc/external/isl.h"
#include "tc/utils/compiler_options.h"

namespace tc {
namespace polyhedral {

// Reduction dims must be properly ordered
using ReductionDimSet = std::set<std::string>;
class TensorReferenceGroup;

struct Scop {
 private:
  Scop() {}

 public:
  // Should be reserved for internal use and unit testing.
  static std::unique_ptr<Scop> makeScop(
      isl::ctx ctx,
      const tc2halide::HalideComponents& components);

  // Preferred points of entry, given a TC string or a treeRef,
  // Halide IR is constructed and made a member by setting halideComponents.
  // These operations are grouped and scheduled in a halide::Stmt which becomes
  // the unit from which the scop is constructed.
  static std::unique_ptr<Scop> makeScop(
      isl::ctx ctx,
      const std::string& tc,
      const CompilerOptions& compilerOptions = CompilerOptions());

  static std::unique_ptr<Scop> makeScop(
      isl::ctx ctx,
      const lang::TreeRef& treeRef,
      const CompilerOptions& compilerOptions = CompilerOptions());

  // Clone a Scop
  static std::unique_ptr<Scop> makeScop(const Scop& scop) {
    auto res = std::unique_ptr<Scop>(new Scop());
    res->parameterValues = scop.parameterValues;
    res->halide = scop.halide;
    res->body = scop.body;
    res->dependences = scop.dependences;
    res->scheduleTreeUPtr =
        detail::ScheduleTree::makeScheduleTree(*scop.scheduleTreeUPtr);
    res->treeSyncUpdateMap = scop.treeSyncUpdateMap;
    res->defaultReductionInitMap = scop.defaultReductionInitMap;
    res->groupCounts_ = scop.groupCounts_;
    res->promotedDecls_ = scop.promotedDecls_;
    res->activePromotions_ = scop.activePromotions_;
    return res;
  }

  // Return a context encapsulating all known information about
  // the parameters.  In particular, all parameters are known
  // to be non-negative and the parameters fixed by fixParameters
  // have a known value.
  // This context lives in a parameter space.
  // The scop is not necessarily specialized to its context.
  // Call specializeToContext to perform this specialization.
  // The schedule tree of the scop does not necessarily have
  // a context node.  Call updateTopLevelContext on the schedule tree
  // to introduce or refine such a context node.
  isl::Set<> context() const {
    auto ctx = domain().get_ctx();
    auto context = halide2isl::makeParamContext(ctx, halide.params);
    return context.intersect(makeContext(parameterValues));
  }

  // Specialize a Scop by fixing the given parameters to the given sizes.
  // If you want to intersect the support domain with the
  // resulting context then you need to do it explicitly.
  // Otherwise ambiguities will ensue.
  // TODO: this is still subject to interpretation but intersecting seems a
  // bit final here so probably we're right not doing it.
  template <typename T>
  static std::unique_ptr<Scop> makeSpecializedScop(
      const Scop& scop,
      const std::unordered_map<std::string, T>& sizes) {
    auto res = makeScop(scop);
    res->fixParameters(sizes);
    // **WARNING** if called before scheduling, this could result in a
    // (partially) specialized schedule, i.e. force
    // strategy.proto.fix_parameters_before_scheduling to true.
    // If you want to fix the parameters in the support domain,
    // then you need to do it explicitly.
    // TODO: this is still subject to interpretation but intersecting seems
    // final here so probably we're right not doing it.
    // res->specializeToContext();
    return res;
  }

  // Specialize the Scop with respect to its context.
  void specializeToContext() {
    auto globalParameterContext = context();
    domainRef() = domain().intersect_params(globalParameterContext);
    body.specialize(globalParameterContext);
  }

  // Returns a set that specializes the named scop's subset of
  // parameter space to the integer values passed to the function.
  template <typename T>
  isl::Set<> makeContext(
      const std::unordered_map<std::string, T>& sizes =
          std::unordered_map<std::string, T>()) const {
    auto s = domain().get_space();
    return makeSpecializationSet(s, sizes);
  }

  // Returns a set that specializes the named scop's subset of
  // parameter space to the integer values passed to the function.
  template <typename T>
  isl::set makeContext(
      std::initializer_list<std::pair<const std::string, T>> sizes) {
    auto s = domain().get_space().params();
    return makeSpecializationSet(s, sizes);
  }

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values by keeping track of them
  // in parameterValues.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    CHECK(parameterValues.size() == 0);
    for (const auto& kvp : sizes) {
      parameterValues.emplace(kvp.first, kvp.second);
    }
  }

  // Return the list of parameter values in the same
  // order as codegen places them in the function signature, i.e. following the
  // order of scop.params.
  std::vector<long> getParameterValues() const;

  isl::id nextGroupIdForTensor(isl::id tensorId) {
    auto ctx = domain().get_ctx();
    std::stringstream ss;
    ss << "_" << tensorId.get_name() << "_" << groupCounts_[tensorId]++;
    return isl::id(ctx, ss.str());
  }

  // Assuming redPoint is a reduction candidate node with
  // the given reduction update statement identifier,
  // add an extension node for a reduction init and
  // a reduction update statement and insert the new
  // statements before and after (the children of) redPoint.
  // If redPoint is a sequence node, then the new node are inserted
  // inside that sequence node.  Otherwise, a new sequence node is created.
  //
  // The transformed shape is:
  //
  // *extension(     <- extension
  //   sequence(
  //     *filter()   <- red_init in new or existing sequence
  //     redPoint
  //     *filter()   <- red_update in new or existing sequence
  //   )
  // )
  //
  // This tree structure typically appears when one does not include the
  // innermost loop as part of an n-D tiling and mapping scheme but rather
  // does (n-K)D tiling and placement and then another level of placement
  // inside that.
  isl::id insertReductionSync1D(
      detail::ScheduleTree* redPoint,
      isl::id updateId);

  // The different level of synchronization.
  enum class SyncLevel : int { None = 0, Warp = 1, Block = 2 };

  // Given a sequence node in the schedule tree, insert
  // synchronization before the child at position "pos".
  // If "pos" is equal to the number of children, then
  // the synchronization is added after the last child.
  void insertSync(
      detail::ScheduleTree* seqNode,
      size_t pos,
      SyncLevel level = SyncLevel::Block) {
    insertExtensionLabelAt(scheduleRoot(), seqNode, pos, makeSyncId(level));
  }

  // Insert synchronization after the given subtree,
  // creating a sequence node if needed.
  void insertSyncAfter(
      detail::ScheduleTree* tree,
      SyncLevel level = SyncLevel::Block) {
    insertExtensionLabelAfter(scheduleRoot(), tree, makeSyncId(level));
  }

  size_t reductionUID() const {
    static size_t count = 0;
    return count++;
  }
  size_t syncUID() const {
    static size_t count = 0;
    return count++;
  }
  size_t warpSyncUID() const {
    static size_t count = 0;
    return count++;
  }

  // Make the synchronization id corresponding to the synchronization level.
  // The level should not be None.
  isl::id makeSyncId(SyncLevel level) {
    switch (level) {
      case SyncLevel::Warp:
        return makeWarpSyncId();
        break;
      case SyncLevel::Block:
        return makeSyncId();
        break;
      default:
        TC_CHECK(level != SyncLevel::None);
        return isl::id();
    }
  }

  isl::id makeSyncId() const {
    auto ctx = domain().get_ctx();
    return isl::id(ctx, std::string(kSyncIdPrefix) + std::to_string(syncUID()));
  }

  isl::id makeWarpSyncId() const {
    auto ctx = domain().get_ctx();
    return isl::id(
        ctx, std::string(kWarpSyncIdPrefix) + std::to_string(warpSyncUID()));
  }

  // Check if the id has a name with the expected prefix, followed by a long
  // integer.
  static bool isIdWithExpectedPrefix(
      isl::id id,
      const std::string& expectedPrefix) {
    auto name = id.get_name();
    if (name.find(expectedPrefix) != 0) {
      return false;
    }
    name = name.substr(expectedPrefix.size());
    char* end;
    std::strtol(name.c_str(), &end, 10);
    if (name.c_str() + name.size() != end) {
      return false;
    }
    return true;
  }

  static bool isSyncId(isl::id id) {
    return isIdWithExpectedPrefix(id, kSyncIdPrefix);
  }

  static bool isWarpSyncId(isl::id id) {
    return isIdWithExpectedPrefix(id, kWarpSyncIdPrefix);
  }

  static isl::id makeRefId(isl::ctx ctx) {
    static thread_local size_t count = 0;
    return isl::id(ctx, std::string("__tc_ref_") + std::to_string(count++));
  }

  std::pair<isl::id, isl::id> makeReductionSpecialIds(isl::id updateId) {
    auto uid = reductionUID();
    auto treeSyncId = isl::id(
        domain().get_ctx(), std::string("red_update") + std::to_string(uid));
    auto reductionInitId = isl::id(
        domain().get_ctx(), std::string("red_init") + std::to_string(uid));
    TC_CHECK_EQ(0u, treeSyncUpdateMap.count(treeSyncId));
    TC_CHECK_EQ(0u, defaultReductionInitMap.count(treeSyncId));

    treeSyncUpdateMap.emplace(treeSyncId, updateId);
    defaultReductionInitMap.emplace(treeSyncId, reductionInitId);
    return std::make_pair(treeSyncId, reductionInitId);
  }

  bool isTreeSyncId(isl::id id) const {
    return treeSyncUpdateMap.count(id) == 1;
  }

  bool isDefaultReductionInitId(isl::id id) const {
    for (const auto& p : defaultReductionInitMap) {
      if (p.second == id) {
        return true;
      }
    }
    return false;
  }

  isl::id getReductionUpdateForDefaultInit(isl::id id) const {
    for (const auto& p : defaultReductionInitMap) {
      if (p.second == id) {
        return treeSyncUpdateMap.at(p.first);
      }
    }
    TC_CHECK(false) << "not found";
    return id;
  }

  bool isReductionUpdate(isl::id id) const {
    for (const auto& kvp : treeSyncUpdateMap) {
      if (id == kvp.second) {
        return true;
      }
    }
    return false;
  }

  size_t reductionUpdatePos(isl::id id) const {
    size_t pos = 0;
    TC_CHECK(isReductionUpdate(id));
    for (const auto& kvp : treeSyncUpdateMap) {
      if (id == kvp.second) {
        return pos;
      }
      pos++;
    }
    return -1;
  }

  void promoteEverythingAt(std::vector<size_t> pos);

  struct PromotedDecl {
    enum class Kind { SharedMem, Register };

    isl::id tensorId;
    std::vector<size_t> sizes;
    Kind kind;
  };

  struct PromotionInfo {
    std::shared_ptr<TensorReferenceGroup> group;
    isl::union_map outerSchedule;
    isl::id groupId;
  };

  const std::unordered_map<isl::id, PromotedDecl, isl::IslIdIslHash>&
  promotedDecls() const {
    return promotedDecls_;
  }

  // Return the promoted declaration information associated to
  // the given identifier of a promoted tensor reference group.
  const PromotedDecl& promotedDecl(isl::id groupId) const {
    if (promotedDecls().count(groupId) != 1) {
      std::stringstream ss;
      ss << "promoted group " << groupId << " has no declaration";
      throw std::logic_error(ss.str());
    }
    return promotedDecls().at(groupId);
  }

  const std::vector<std::pair<isl::union_set, PromotionInfo>>&
  activePromotions() const {
    return activePromotions_;
  }

  detail::ScheduleTree* scheduleRoot() {
    return scheduleTreeUPtr.get();
  }

  const detail::ScheduleTree* scheduleRoot() const {
    return scheduleTreeUPtr.get();
  }

  // Create a Scop scheduled with a given scheduling strategy.
  static std::unique_ptr<Scop> makeScheduled(
      const Scop& scop,
      const SchedulerOptionsView& schedulerOptions);
  // Return the outermost band in the schedule tree with the given root.
  // If there is no single outermost band, then insert a (permutable)
  // zero-dimensional band and return that.
  detail::ScheduleTree* obtainOuterBand();
  // Tile the outermost band.
  // Splits the band into tile loop band and point loop band where point loops
  // have fixed trip counts specified in "tiling", and returns a pointer to the
  // tile loop band.
  detail::ScheduleTree* tileOuterBand(const TilingView& tiling);

  // Reschedule the schedule subtree rooted at "tree" with the
  // given scheduler options.
  void reschedule(
      detail::ScheduleTree* tree,
      const SchedulerOptionsView& schedulerOptions);

  // Find an input or an output argument given its name.
  // Assumes such argument exists.
  const Halide::OutputImageParam& findArgument(isl::id id) const;

  // Make an affine function from a Halide Expr that is defined
  // over the instance set of the statement with identifier "stmtId".
  // Return a null isl::aff if the expression is not affine.  Fail if any
  // of the variables does not correspond to a parameter or
  // an instance identifier of the statement.
  isl::AffOn<Statement> makeIslAffFromStmtExpr(
      isl::id stmtId,
      const Halide::Expr& e) const;

  // Promote a tensor reference group to a storage of a given "kind",
  // inserting the copy
  // statements below the given node.  Inserts an Extension node below the give
  // node, unless there is already another Extension node which introduces
  // copies.  The Extension node has a unique Sequence child, whose children
  // perform copies from global memory, then main computation using the
  // original nodes, then copies back to global memory.  The caller is in
  // charge of inserting the synchronization nodes.
  //
  // Creates the promoted array declaration in the internal list.
  // If "forceLastExtentOdd" is set, the last extent in the declaration is
  // incremented if it is even.  This serves as a simple heuristic to reduce
  // shared memory bank conflicts.
  void promoteGroup(
      PromotedDecl::Kind kind,
      isl::id tensorId,
      std::unique_ptr<TensorReferenceGroup>&& gr,
      detail::ScheduleTree* tree,
      isl::union_map schedule,
      bool forceLastExtentOdd = false);

  // Given a tree node under which the promotion copy statements were
  // introduced, insert syncthread statements before and after the copies.
  // The tree should match the structure:
  //   any(
  //     extension(
  //       sequence(
  //         // <-- sync will be inserted here
  //         filter(any()), // filter that refers to read
  //         ...
  //         // <-- sync will be inserted here if filter above exists
  //         filter(any()), // at least one filter that does not refer to
  //         ...            // read/write
  //         // <-- sync will be inserted here if filter below exists
  //         filter(any()), // filter that refers to write
  //         ...
  //         // <-- sync will be inserted here
  //         )))
  //
  void insertSyncsAroundCopies(detail::ScheduleTree* tree);

  // Given a sequence node, insert synchronizations before its first child node
  // and after its last child node.
  void insertSyncsAroundSeqChildren(detail::ScheduleTree* tree);

 private:
  // Compute a schedule satisfying the given schedule constraints and
  // taking into account the scheduler options.
  // Note that some of the scheduler options have already been
  // taken into account during the construction of the schedule constraints.
  static std::unique_ptr<detail::ScheduleTree> computeSchedule(
      isl::schedule_constraints constraints,
      const SchedulerOptionsView& schedulerOptions);

 public:
  // Do the simplest possible dependence analysis.
  // Compute all RAW, WAR, and WAW dependences, and save them in dependences.
  void computeAllDependences();
  // Return the set of dependences that are active
  // at the given position.
  isl::union_map activeDependences(detail::ScheduleTree* tree);

 public:
  // Halide stuff
  struct {
    halide2isl::ParameterVector params;
    std::vector<std::string> idx, reductionIdx;
    std::vector<Halide::ImageParam> inputs;
    std::vector<Halide::OutputImageParam> outputs;
    std::unordered_map<isl::id, Halide::Internal::Stmt, isl::IslIdIslHash>
        statements;
    std::unordered_map<const Halide::Internal::IRNode*, isl::id> accesses;
    halide2isl::IterationDomainMap domains;
  } halide;

  // Polyhedral IR
  //
  // The domain is collected from the root of the ScheduleTree; no redundant
  // state is kept.
  // By analogy with generalized functions, the domain is the "support" part
  // of the ScheduleTree "function".
 private:
  isl::union_set& domainRef();

 public:
  const isl::UnionSet<Statement> domain() const;
  // The parameter values of a specialized Scop.
  std::unordered_map<std::string, int> parameterValues;

  Body body;

  // RAW, WAR, and WAW dependences
  isl::union_map dependences;

 private:
  // By analogy with generalized functions, a ScheduleTree is a (piecewise
  // affine) function operating on a support.
  // The support is originally an isl::union_set corresponding to the union of
  // the iteration domains of the statements in the Scop.
  // The support must be the unique root node of the ScheduleTree and be of
  // type: ScheduleTreeDomain.
  std::unique_ptr<detail::ScheduleTree> scheduleTreeUPtr;

 public:
  // For reduction matching purposes we keep the following maps
  std::unordered_map<isl::id, isl::id, isl::IslIdIslHash> treeSyncUpdateMap;
  std::unordered_map<isl::id, isl::id, isl::IslIdIslHash>
      defaultReductionInitMap; // treeSyncId -> defaultInitId

 private:
  // Memory promotion stuff
  // tensorId -> number of mapped groups
  std::unordered_map<isl::id, size_t, isl::IslIdIslHash> groupCounts_;
  // groupId -> (tensorId, groupSizes)
  std::unordered_map<isl::id, PromotedDecl, isl::IslIdIslHash> promotedDecls_;
  // (domain, group, partial schedule, groupId)
  // Note that domain is a non-unique key, i.e. multiple groups can be listed
  // for the same domain, or for partially intersecting domains.
  std::vector<std::pair<isl::union_set, PromotionInfo>> activePromotions_;
};

std::ostream& operator<<(std::ostream& os, const Scop&);

} // namespace polyhedral
} // namespace tc
