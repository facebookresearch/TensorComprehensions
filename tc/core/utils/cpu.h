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

#include <unordered_map>

#include <cpuid.h>

#include "tc/core/flags.h"
#include "tc/core/utils/cpu.h"

namespace tc {
namespace utils {

#define INTEL_ebx 0x756e6547
#define INTEL_ecx 0x6c65746e
#define INTEL_edx 0x49656e69

/**
 * We start with a reasonable subset of the processors listed in the result
 * of running the command:
 *    llvm-as < /dev/null | llc -march=x86-64 -mcpu=help
 */
struct CPUID {
 public:
  CPUID() : eax(0), ebx(0), ecx(0), edx(0) {
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
  }

  static bool isIntel() {
    unsigned int a, b, c, d;
    __get_cpuid(0, &a, &b, &c, &d);
    return b == INTEL_ebx && c == INTEL_ecx && d == INTEL_edx;
  }

  using Stepping = unsigned char;
  using Model = unsigned char;
  using Family = unsigned char;
  using ProcessorType = unsigned char;
  using ExtendedModel = unsigned char;
  using ExtendedFamily = unsigned short;
  struct FullModel {
    FullModel(Model m, ExtendedModel em) {
      val = (em << 4) + m;
    }
    operator unsigned short() {
      return val;
    }
    operator const unsigned short&() const {
      return val;
    }
    unsigned short val;
  };

  static const std::unordered_map<unsigned short, std::string>&
  intelFamily6ExtendedFamily0() {
    static std::unordered_map<unsigned short, std::string> m{
        {FullModel(0xD, 0x3), "broadwell"}, // client
        {FullModel(0x7, 0x4), "broadwell"}, // client
        {FullModel(0xF, 0x4), "broadwell"}, // server
        {FullModel(0x6, 0x5), "broadwell"}, // server
        {FullModel(0x6, 0x6), "cannonlake"}, // client
        {FullModel(0x6, 0x4), "haswell"}, // client
        {FullModel(0x5, 0x4), "haswell"}, // client
        {FullModel(0xC, 0x3), "haswell"}, // client
        {FullModel(0xF, 0x3), "haswell"}, // server
        {FullModel(0xA, 0x3), "ivybridge"}, // client
        {FullModel(0xE, 0x3), "ivybridge"}, // server
        {FullModel(0xA, 0x2), "sandybridge"}, // client
        {FullModel(0xD, 0x2), "sandybridge"}, // server
        {FullModel(0xE, 0x4), "skylake"}, // client
        {FullModel(0xE, 0x5), "skylake"}, // client
        {FullModel(0x5, 0x5), "skylake-avx512"}, // server
        {FullModel(0x5, 0x2), "westmere"}, // client
        {FullModel(0xC, 0x2), "westmere"}, // server
        {FullModel(0xF, 0x2), "westmere"}, // server
    };
    return m;
  };

  static std::tuple<
      Stepping,
      Model,
      Family,
      ProcessorType,
      ExtendedModel,
      ExtendedFamily>
  parseCPU() {
    CPUID id;
    return std::make_tuple(
        static_cast<Stepping>(id.eax & 0x0000000F), // 3:0
        static_cast<Model>((id.eax >> 4) & 0x0000000F), // 7:4
        static_cast<Family>((id.eax >> 8) & 0x0000000F), // 11:8
        static_cast<ProcessorType>((id.eax >> 12) & 0x00000003), // 13:12
        static_cast<ExtendedModel>((id.eax >> 16) & 0x0000000F), // 19:16
        static_cast<ExtendedFamily>((id.eax >> 20) & 0x000000FF) // 27:20
    );
  }

#define INTEL_FAMILY_6 0x6
#define INTEL_EXTENDED_FAMILY_0 0x0
  static std::string mcpu() {
    if (FLAGS_mcpu.size() > 0) {
      return FLAGS_mcpu;
    }

    TC_CHECK(CPUID::isIntel());
    auto parsedCPU = CPUID::parseCPU();
    auto model = std::get<1>(parsedCPU);
    auto family = std::get<2>(parsedCPU);
    auto extendedModel = std::get<4>(parsedCPU);
    auto extendedFamily = std::get<5>(parsedCPU);
    if (family == INTEL_FAMILY_6 && extendedFamily == INTEL_EXTENDED_FAMILY_0) {
      if (intelFamily6ExtendedFamily0().count(FullModel(model, extendedModel)) >
          0) {
        return intelFamily6ExtendedFamily0().at(
            FullModel(model, extendedModel));
      }
      LOG(ERROR) << "FullModel: "
                 << (unsigned short)FullModel(model, extendedModel)
                 << " -> unspecified x86-64";
      return "x86-64";
    }
    TC_CHECK(false) << "Unsupported family/model/extendedmodel: " << family
                    << "/" << model << "/" << extendedModel;
    return "";
  }

  static std::string llcFlags() {
    return std::string("-march=x86-64 -mcpu=") + CPUID::mcpu();
  }

 public:
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
};
} // namespace utils
} // namespace tc
