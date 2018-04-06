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

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "tc/core/cuda/cuda_compilation_cache.h"

DEFINE_string(output, "", "filename prefix for merged caches (prefix.options)");

bool fileExists(const std::string& filename) {
  struct stat buffer = {0};
  return stat(filename.c_str(), &buffer) == 0;
}

bool isOptionsFilename(const std::string& filename) {
  auto pos = filename.rfind(".options");
  if (pos == std::string::npos) {
    return false;
  }
  if (pos != filename.size() - 8) {
    return false;
  }
  return true;
}

std::vector<std::string> getFilenames(int argc, char** argv) {
  std::vector<std::string> filenames;
  for (int i = 1; i < argc; ++i) {
    filenames.emplace_back(argv[i]);
    const auto& f = filenames.back();
    if (not isOptionsFilename(f)) {
      std::stringstream ss;
      ss << f << "does not end with .options";
      throw std::invalid_argument(ss.str());
    }

    if (not fileExists(f)) {
      std::stringstream ss;
      ss << "File " << filenames.back() << " does not exist.";
      throw std::invalid_argument(ss.str());
    }
  }
  return filenames;
}

template <typename Cache>
std::vector<Cache> loadCaches(const std::vector<std::string>& filenames) {
  std::vector<Cache> caches;
  for (const auto& f : filenames) {
    typename Cache::ProtobufType buf;
    std::ifstream serialized(f, std::ios::binary);
    buf.ParseFromIstream(&serialized);
    caches.emplace_back(buf);
  }
  return caches;
}

template <typename Cache>
void saveCache(const Cache& c, const std::string& outputFilename) {
  std::fstream serialized(
      outputFilename, std::ios::binary | std::ios::trunc | std::ios::out);
  if (!serialized) {
    std::cout << "Failed to open the output stream for dumping protobuf: "
              << outputFilename;
  } else {
    c.toProtobuf().SerializePartialToOstream(&serialized);
  }
}

template <typename Cache>
void mergeCachesAndSave(
    const std::vector<Cache>& caches,
    const std::string& outputFilename) {
  Cache merged(caches.at(0).toProtobuf());
  for (auto it = caches.begin() + 1; it != caches.end(); ++it) {
    merged.mergeWith(*it);
  }
  saveCache(merged, outputFilename);
}

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_output == "") {
    std::cout << "No output filename prefix specified, use the --output flag\n";
    return 0;
  }

  auto optionsOutputName = FLAGS_output + ".options";
  if (fileExists(optionsOutputName)) {
    std::cout << "Output file already exist\n";
    return 0;
  }

  auto optionsFilenames = getFilenames(argc, argv);
  if (optionsFilenames.empty()) {
    std::cout << "No inputs: pass cache filenames as command line arguments\n";
    return 0;
  }

  mergeCachesAndSave(
      loadCaches<tc::OptionsCache>(optionsFilenames), optionsOutputName);
}
