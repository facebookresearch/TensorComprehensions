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
#include <exception>
#include <sstream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "tc/core/flags.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/external/isl.h"

using namespace isl::with_exceptions;
using namespace tc;

struct Isl : public ::testing::Test {
  isl::ctx ctx;
  Isl() : ctx(isl::with_exceptions::globalIslCtx()) {}
};

TEST_F(Isl, Context) {}

TEST_F(Isl, DropDims) {
  isl::multi_pw_aff mpa(
      ctx, " { [i, j] -> [ (j + i : i > 0), (2 * i : j > 0)] } ");
  isl::multi_union_pw_aff mupa(mpa);
  mupa = mupa.set_tuple_name(isl::dim_type::set, std::string("SOME_TUPLE"));
  mupa = dropDimsPreserveTuple(mupa, isl::dim_type::set, 0, 1);
}

TEST_F(Isl, DropDimsViaBand) {
  isl::multi_pw_aff mpa(
      ctx, " { [i, j] -> [ (j + i : i > 0), (2 * i : j > 0)] } ");
  isl::multi_union_pw_aff mupa(mpa);
  using namespace tc::polyhedral;
  using namespace tc::polyhedral::detail;
  auto b = ScheduleTreeElemBand::fromMultiUnionPwAff(mupa);
  b->drop(0, 1);
}

TEST_F(Isl, DropDimsViaSchedule) {
  isl::multi_pw_aff mpa(
      ctx, " { [i, j] -> [ (j + i : i > 0), (2 * i : j > 0)] } ");
  isl::multi_union_pw_aff mupa(mpa);
  using namespace tc::polyhedral;
  using namespace tc::polyhedral::detail;
  auto tree = ScheduleTree::makeBand(mupa);
  isl::set s(ctx, R"S([i, j] -> {  : 0 <= i <= 10 and 0 <= j <= 20 })S");
  auto root = ScheduleTree::makeDomain(isl::union_set(s), std::move(tree));
  // Note that tree was moved-from, so we have to access from the child
  // or store the raw pointer.
  bandSplit(root.get(), root->child({0}), 1);
}

TEST_F(Isl, Mupa) {
  isl::space space = isl::set(ctx, "{[a,b,c]:}").get_space();
  isl::space schedule_space = space.map_from_set();
  isl::pw_multi_aff identity_fun = isl::pw_multi_aff(schedule_space);
  isl::multi_union_pw_aff mupa =
      isl::multi_union_pw_aff(isl::multi_pw_aff(identity_fun));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
