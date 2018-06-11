/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "tc/core/check.h"

// gtest doesn't define a macro that inspects the contents of the exception
#define ASSERT_THROW_WHAT(x, y)      \
  try {                              \
    (x);                             \
    ASSERT_TRUE(false);              \
  } catch (std::runtime_error & e) { \
    ASSERT_EQ(y, e.what());          \
  }

TEST(CHECK, Plain) {
  ASSERT_NO_THROW(TC_CHECK(true));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 1 << ']';
    ASSERT_THROW_WHAT(TC_CHECK(false), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2 << ']'
             << ": 1+1=3";
    ASSERT_THROW_WHAT(TC_CHECK(false) << "1+1=3", expected.str());
  }
}

TEST(CHECK, Vector) {
  std::stringstream expected;
  auto v = std::vector<int>{1, 2, 3, 4};
  expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2 << ']'
           << ": 1,2,3,4";
  ASSERT_THROW_WHAT((TC_CHECK(false) << v), expected.str());
}

TEST(CHECK, EQ) {
  ASSERT_NO_THROW(TC_CHECK_EQ(1, 1));
  ASSERT_NO_THROW(TC_CHECK_EQ(std::string("aaa"), std::string("aaa")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2 << "] "
             << "1 not equal to 2";
    ASSERT_THROW_WHAT(TC_CHECK_EQ(1, 2), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 1 not equal to 2: 2+2=5";
    ASSERT_THROW_WHAT(TC_CHECK_EQ(1, 2) << "2+2=5", expected.str());
  }
}

TEST(CHECK, NE) {
  ASSERT_NO_THROW(TC_CHECK_NE(1, 2));
  ASSERT_NO_THROW(TC_CHECK_NE(std::string("aaa"), std::string("baa")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 1 equal to 1";
    ASSERT_THROW_WHAT(TC_CHECK_NE(1, 1), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 1 equal to 1: 2+2=5";
    ASSERT_THROW_WHAT(TC_CHECK_NE(1, 1) << "2+2=5", expected.str());
  }
}

TEST(CHECK, LT) {
  ASSERT_NO_THROW(TC_CHECK_LT(1, 2));
  ASSERT_NO_THROW(TC_CHECK_LT(std::string("aaa"), std::string("baa")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 1 not less than 1";
    ASSERT_THROW_WHAT(TC_CHECK_LT(1, 1), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 4 not less than " << 1 << ": 4+3=8";
    ASSERT_THROW_WHAT(TC_CHECK_LT(4, 1) << "4+3=8", expected.str());
  }
}

TEST(CHECK, GT) {
  ASSERT_NO_THROW(TC_CHECK_GT(2, 1));
  ASSERT_NO_THROW(TC_CHECK_GT(std::string("ca"), std::string("baa")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 1 not greater than " << 1;
    ASSERT_THROW_WHAT(TC_CHECK_GT(1, 1), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 2 not greater than 4: 3+3=7";
    ASSERT_THROW_WHAT(TC_CHECK_GT(2, 4) << "3+3=7", expected.str());
  }
}

TEST(CHECK, LE) {
  ASSERT_NO_THROW(TC_CHECK_LE(1, 2));
  ASSERT_NO_THROW(TC_CHECK_LE(1, 1));
  ASSERT_NO_THROW(TC_CHECK_LE(std::string("aaa"), std::string("baa")));
  ASSERT_NO_THROW(TC_CHECK_LE(std::string("aa"), std::string("aa")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 2 not less than or equal to 1";
    ASSERT_THROW_WHAT(TC_CHECK_LE(2, 1), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 4 not less than or equal to 1: 4+5=10";
    ASSERT_THROW_WHAT(TC_CHECK_LE(4, 1) << "4+5=10", expected.str());
  }
}

TEST(CHECK, GE) {
  ASSERT_NO_THROW(TC_CHECK_GE(2, 1));
  ASSERT_NO_THROW(TC_CHECK_GE(2, 2));
  ASSERT_NO_THROW(TC_CHECK_GE(std::string("ca"), std::string("baa")));
  ASSERT_NO_THROW(TC_CHECK_GE(std::string("ba"), std::string("ba")));
  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 7 not greater than or equal to 9";
    ASSERT_THROW_WHAT(TC_CHECK_GE(7, 9), expected.str());
  }

  {
    std::stringstream expected;
    expected << "Check failed [" << __FILE__ << ':' << __LINE__ + 2
             << "] 2 not greater than or equal to 6: 9+3=13";
    ASSERT_THROW_WHAT(TC_CHECK_GE(2, 6) << "9+3=13", expected.str());
  }
}

TEST(CHECK, CustomException) {
  ASSERT_THROW(TC_CHECK(false, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK(false, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK(true, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK(true, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_EQ(1, 2, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_EQ(1, 2, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_EQ(1, 1, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_EQ(1, 1, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_NE(1, 1, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_NE(1, 1, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_NE(1, 2, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_NE(1, 2, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_LT(1, 1, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_LT(1, 1, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_LT(1, 2, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_LT(1, 2, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_GT(1, 1, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_GT(1, 1, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_GT(2, 1, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_GT(2, 1, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_LE(2, 1, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_LE(2, 1, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_LE(1, 2, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_LE(1, 2, std::out_of_range) << "aa");

  ASSERT_THROW(TC_CHECK_GE(1, 2, std::out_of_range), std::out_of_range);
  ASSERT_THROW(TC_CHECK_GE(1, 2, std::out_of_range) << "aa", std::out_of_range);
  ASSERT_NO_THROW(TC_CHECK_GE(2, 1, std::out_of_range));
  ASSERT_NO_THROW(TC_CHECK_GE(2, 1, std::out_of_range) << "aa");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
