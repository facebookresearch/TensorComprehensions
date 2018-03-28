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
#include <sstream>
#include <string>

#include <glog/logging.h>
#include "tc/external/isl.h"

#include "tc/core/polyhedral/schedule_tree.h"
#include "tc/core/polyhedral/schedule_tree_elem.h"
#include "tc/external/isl.h"

using namespace std;

namespace tc {
namespace polyhedral {
namespace detail {

namespace {

struct WS {
  static thread_local int n;
  WS() {
    n += 2;
  }
  ~WS() {
    n -= 2;
  }
  std::string tab() {
    std::stringstream ss;
    for (int i = 0; i < n; ++i) {
      ss << " ";
    }
    return ss.str();
  }
};
thread_local int WS::n = -2; // want 0 after first instantiation

/*
 * Very basic version of an ostream_joiner.
 * Can be replaced by standard version, once the switch to C++17 is made.
 */
class ostream_joiner
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
 public:
  ostream_joiner(std::ostream& os, const char* delimiter)
      : os_(os), delimiter_(delimiter) {}
  ostream_joiner& operator*() {
    return *this;
  }
  ostream_joiner& operator++() {
    return *this;
  }
  template <typename T>
  ostream_joiner& operator=(const T& value) {
    if (!first) {
      os_ << delimiter_;
    }
    first = false;
    os_ << value;
    return *this;
  }

 private:
  bool first = true;
  std::ostream& os_;
  const char* delimiter_;
};

ostream_joiner make_ostream_joiner(std::ostream& os, const char* delimiter) {
  return ostream_joiner(os, delimiter);
}

} // namespace

std::ostream& operator<<(std::ostream& os, isl::ast_loop_type lt) {
  WS w;
  os << "type(";
  if (lt == isl::ast_loop_type::error) {
    os << "error";
  } else if (lt == isl::ast_loop_type::_default) {
    os << "default";
  } else if (lt == isl::ast_loop_type::atomic) {
    os << "atomic";
  } else if (lt == isl::ast_loop_type::unroll) {
    os << "unroll";
  } else if (lt == isl::ast_loop_type::separate) {
    os << "separate";
  } else {
    LOG(FATAL) << "NYI: print type: " << static_cast<int>(lt);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, detail::ScheduleTreeType nt) {
  WS w;
  os << w.tab() << "type(";
  if (nt == detail::ScheduleTreeType::None) {
    os << "error";
  } else if (nt == detail::ScheduleTreeType::Band) {
    os << "band";
  } else if (nt == detail::ScheduleTreeType::Context) {
    os << "context";
  } else if (nt == detail::ScheduleTreeType::Domain) {
    os << "domain";
  } else if (nt == detail::ScheduleTreeType::Extension) {
    os << "extension";
  } else if (nt == detail::ScheduleTreeType::Filter) {
    os << "filter";
  } else if (nt == detail::ScheduleTreeType::MappingFilter) {
    os << "mapping_filter";
  } else if (nt == detail::ScheduleTreeType::Sequence) {
    os << "sequence";
  } else if (nt == detail::ScheduleTreeType::Set) {
    os << "seq";
  } else {
    LOG(FATAL) << "NYI: print type: " << static_cast<int>(nt);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScheduleTreeElemBase& st) {
  st.write(os);
  return os;
}

std::ostream& ScheduleTreeElemBand::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "band(n(" << coincident_.size() << ") permutable(";
  os << permutable_ << ") coincident(";
  std::copy(
      coincident_.begin(), coincident_.end(), make_ostream_joiner(os, ", "));
  os << ")";
  os << " unroll(";
  std::copy(unroll_.begin(), unroll_.end(), make_ostream_joiner(os, ", "));
  os << ")";
  for (const auto& upa : isl::MUPA(mupa_)) {
    os << std::endl
       << w.tab()
       << "-----------------------------------------------------------------------";
    for (const auto& pa : upa) {
      os << std::endl << w.tab() << "| " << pa.pa;
    }
  }
  os << std::endl
     << w.tab()
     << "-----------------------------------------------------------------------";
  return os;
}

std::ostream& ScheduleTreeElemContext::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "context(";
  for (const auto& u : isl::UNION_SET(context_)) {
    WS w2;
    os << std::endl << w2.tab() << u;
  }
  os << ")";
  return os;
}

std::ostream& ScheduleTreeElemDomain::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "domain(";
  for (const auto& u : isl::UNION_SET(domain_)) {
    WS w2;
    os << std::endl << w2.tab() << u;
  }
  os << ")";
  return os;
}

std::ostream& ScheduleTreeElemExtension::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "extension(";
  for (const auto& u : isl::UnionAsVector<isl::union_map>(extension_)) {
    WS w2;
    os << std::endl << w2.tab() << u;
  }
  os << ")";
  return os;
}

std::ostream& ScheduleTreeElemFilter::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "filter(";
  for (const auto& u : isl::UNION_SET(filter_)) {
    WS w2;
    os << std::endl << w2.tab() << u;
  }
  os << ")";
  return os;
}

std::ostream& ScheduleTreeElemMappingFilter::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "mapping_filter(ids(";
  for (const auto& id : mappingIds) {
    os << id << ", ";
  }
  os << ")";
  for (const auto& u : isl::UNION_SET(filter_)) {
    WS w2;
    os << std::endl << w2.tab() << u;
  }
  os << ")";
  return os;
}

std::ostream& ScheduleTreeElemSequence::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "sequence()";
  return os;
}

std::ostream& ScheduleTreeElemSet::write(std::ostream& os) const {
  WS w;
  os << w.tab() << "set()";
  return os;
}

std::ostream& operator<<(
    std::ostream& os,
    const std::vector<ScheduleTreeUPtr>& vst) {
  if (vst.size() == 0) {
    return os;
  }
  WS w;
  for (const auto& st : vst) {
    os << *(st.get());
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScheduleTree& st) {
  CHECK(st.elem_.get());
  os << *st.elem_ << "\n";
  os << st.children_;

  return os;
}

} // namespace detail
} // namespace polyhedral
} // namespace tc
