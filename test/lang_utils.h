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
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const std::string expected_file_path = "src/lang/test_expected/";

static inline void barf(const char* fmt, ...) {
  char msg[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);
  throw std::runtime_error(msg);
}

#define ASSERT(cond)                         \
  if (__builtin_expect(!(cond), 0)) {        \
    barf(                                    \
        "%s:%u: %s: Assertion `%s` failed.", \
        __FILE__,                            \
        __LINE__,                            \
        __func__,                            \
        #cond);                              \
  }

// note: msg must be a string literal
// node: In, ##__VA_ARGS '##' supresses the comma if __VA_ARGS__ is empty
#define ASSERTM(cond, msg, ...)                   \
  if (__builtin_expect(!(cond), 0)) {             \
    barf(                                         \
        "%s:%u: %s: Assertion `%s` failed: " msg, \
        __FILE__,                                 \
        __LINE__,                                 \
        __func__,                                 \
        #cond,                                    \
        ##__VA_ARGS__);                           \
  }

void writeFile(const std::string& filename, const std::string& value) {
  std::ofstream ofile(filename.c_str());
  ASSERT(ofile.good());
  ofile << value;
}

bool readFile(const std::string& filename, std::string& v) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.good())
    return false;
  std::stringstream input;
  input << ifile.rdbuf();
  v = input.str();
  return true;
}

bool acceptChanges = false;

void assertEqual(
    const std::string& expected_filename_,
    const std::string& the_value) {
  std::string expected_filename = expected_file_path + expected_filename_;
  std::string expected_value;
  if (acceptChanges) {
    writeFile(expected_filename, the_value);
    return;
  }
  if (!readFile(expected_filename, expected_value)) {
    throw std::runtime_error("expect file not found: " + expected_filename);
  }
  if (the_value != expected_value) {
    std::string output = expected_filename + "-actual";
    writeFile(output, the_value);
    std::stringstream ss;
    ss << expected_filename << " did not match. Run:\n diff -u "
       << expected_filename << " " << output
       << "\n to compare. Re-run with --accept to accept changes.";
    throw std::runtime_error(ss.str());
  }
}

void parseArgs(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    args.push_back(argv[i]);
  }
  for (auto& a : args) {
    if (a == "--accept") {
      acceptChanges = true;
    }
  }
}
