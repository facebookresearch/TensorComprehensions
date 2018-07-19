/**
 * Copyright (c) 2018, Facebook, Inc.
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

#include <stdlib.h>

#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

constexpr auto header = R"CPP(
struct Anonymous;

template <typename Name, typename First, typename Second>
struct NamedPair;

template <typename First, typename Second>
using Pair = NamedPair<Anonymous, First, Second>;
)CPP";

constexpr auto footer = R"CPP(
template <typename... Ts>
using AffOn = Aff<Ts..., Anonymous>;

template <typename... Ts>
using PwAffOn = PwAff<Ts..., Anonymous>;

template <typename... Ts>
using UnionPwAffOn = UnionPwAff<Ts..., Anonymous>;

template <typename... Ts>
using AffListOn = AffList<Ts..., Anonymous>;

template <typename... Ts>
using PwAffListOn = PwAffList<Ts..., Anonymous>;

template <typename... Ts>
using UnionPwAffListOn = UnionPwAffList<Ts..., Anonymous>;
)CPP";

static std::string dropIslNamespace(std::string type) {
  return std::regex_replace(type, std::regex("isl::"), "");
}

template <typename Type>
struct Signature {
  Type returnType;
  std::vector<Type> argTypes;
};

using Type = std::string;
struct BaseKind {
  BaseKind(const char* name) : name(name) {}
  BaseKind(std::initializer_list<BaseKind> c) : children(c) {
    if (c.size() == 2) {
      children.insert(children.begin(), "Anonymous");
    }
    if (children.size() != 3) {
      abort();
    }
  }
  std::string name;
  std::vector<BaseKind> children;
  bool operator<(const BaseKind& other) const {
    if (children.size() != other.children.size()) {
      return children.size() < other.children.size();
    }
    if (children.size() == 0) {
      return name < other.name;
    }
    for (size_t i = 0; i < children.size(); ++i) {
      if (children[i] != other.children[i]) {
        return children[i] < other.children[i];
      }
    }
    return false;
  }
  bool operator==(const BaseKind& other) const {
    return name == other.name && children == other.children;
  }
  bool operator!=(const BaseKind& other) const {
    return !(*this == other);
  }
};
using Kind = std::vector<BaseKind>;

struct Method {
  std::string name;
  Signature<Type> signature;
};

using Exported = std::unordered_map<std::string, std::vector<Method>>;

struct Class {
  std::string name;
  std::vector<Kind> kinds;
};

static Kind params_type() {
  return {};
}

static Kind set_type() {
  return {"Domain"};
}

static Kind map_type() {
  return {"Domain", "Range"};
}

static const std::unordered_map<std::string, Class> classes{
    {"space", {"Space", {params_type(), set_type(), map_type()}}},
    {"multi_id", {"MultiId", {set_type()}}},
    {"multi_val", {"MultiVal", {set_type()}}},
    {"set", {"Set", {params_type(), set_type()}}},
    {"map", {"Map", {map_type()}}},
    {"aff", {"Aff", {set_type(), map_type()}}},
    {"aff_list", {"AffList", {set_type(), map_type()}}},
    {"pw_aff", {"PwAff", {set_type(), map_type()}}},
    {"union_pw_aff", {"UnionPwAff", {set_type(), map_type()}}},
    {"multi_aff", {"MultiAff", {map_type()}}},
    {"pw_aff_list", {"PwAffList", {set_type(), map_type()}}},
    {"union_pw_aff_list", {"UnionPwAffList", {set_type(), map_type()}}},
    {"multi_union_pw_aff", {"MultiUnionPwAff", {map_type()}}},
    {"union_pw_multi_aff", {"UnionPwMultiAff", {map_type()}}},
    {"union_set", {"UnionSet", {params_type(), set_type()}}},
    {"union_map", {"UnionMap", {map_type()}}},
    {"map_list", {"MapList", {map_type()}}},
    {"union_access_info", {"UnionAccessInfo", {map_type()}}},
    {"union_flow", {"UnionFlow", {map_type()}}},
    {"stride_info", {"StrideInfo", {map_type()}}},
    {"fixed_box", {"FixedBox", {map_type()}}},
};

static Signature<Kind> create_params() {
  return {{}, {}};
}

static Signature<Kind> create_set() {
  return {{"Domain"}, {}};
}

static Signature<Kind> change_set() {
  return {{"ModifiedDomain"}, {{"Domain"}}};
}

static Signature<Kind> change_wrapped_set() {
  return {{{"ModifiedWrap", "WrappedDomain", "WrappedRange"}},
          {{{"Wrap", "WrappedDomain", "WrappedRange"}}}};
}

static Signature<Kind> change_range() {
  return {{"Domain", "ModifiedRange"}, {{"Domain", "Range"}}};
}

static Signature<Kind> create_map() {
  return {{"Domain", "Range"}, {}};
}

static Signature<Kind> apply_set() {
  return {{"Range"}, {{"Domain"}, {"Domain", "Range"}}};
}

static Signature<Kind> apply_domain() {
  return {{"Range3", "Range"}, {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

static Signature<Kind> preimage_domain() {
  return {{"Domain2", "Range"}, {{"Domain", "Range"}, {"Domain2", "Domain"}}};
}

static Signature<Kind> apply_range() {
  return {{"Domain", "Range2"}, {{"Domain", "Range"}, {"Range", "Range2"}}};
}

static Signature<Kind> modify_params_unary() {
  return {{}, {{}}};
}

static Signature<Kind> modify_params_binary() {
  return {{}, {{}, {}}};
}

static Signature<Kind> modify_set_params() {
  return {{"Domain"}, {{"Domain"}, {}}};
}

static Signature<Kind> modify_set_unary() {
  return {{"Domain"}, {{"Domain"}}};
}

static Signature<Kind> modify_set_binary() {
  return {{"Domain"}, {{"Domain"}, {"Domain"}}};
}

static Signature<Kind> modify_domain() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain"}}};
}

static Signature<Kind> modify_range() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Range"}}};
}

static Signature<Kind> modify_map_params() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {}}};
}

static Signature<Kind> modify_map() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain", "Range"}}};
}

static Signature<Kind> modify_map_unary() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}}};
}

static Signature<Kind> map_from_set() {
  return {{"Domain", "Domain"}, {{"Domain"}}};
}

static Signature<Kind> map_on_domain() {
  return {{"Domain", "Domain"}, {{"Domain", "Range"}}};
}

static Signature<Kind> map_from_domain() {
  return {{"Domain", "Range"}, {{"Domain"}}};
}

static Signature<Kind> set_from_params() {
  return {{"Domain"}, {{}}};
}

static Signature<Kind> map_from_params() {
  return {{"Domain", "Range"}, {{}}};
}

static Signature<Kind> set_params() {
  return {{}, {{"Domain"}}};
}

static Signature<Kind> map_params() {
  return {{}, {{"Domain", "Range"}}};
}

static Signature<Kind> set_binary_params() {
  return {{}, {{"Domain"}, {"Domain"}}};
}

static Signature<Kind> domain() {
  return {{"Domain"}, {{"Domain", "Range"}}};
}

static Signature<Kind> domain_map() {
  return {{{"Domain", "Range"}, "Domain"}, {{"Domain", "Range"}}};
}

static Signature<Kind> static_domain_map() {
  return {{{"Range", "Domain2"}, "Range"}, {{"Range", "Domain2"}}};
}

static Signature<Kind> static_range_map() {
  return {{{"Domain2", "Range"}, "Range"}, {{"Domain2", "Range"}}};
}

static Signature<Kind> static_wrapped_range_map() {
  return {{{"Wrap", "WrappedDomain", "WrappedRange"}, "WrappedRange"},
          {{{"Wrap", "WrappedDomain", "WrappedRange"}}}};
}

static Signature<Kind> domain_binary() {
  return {{"Domain"}, {{"Domain", "Range"}, {"Domain", "Range"}}};
}

static Signature<Kind> domain_binary_map() {
  return {{"Domain", "Domain2"}, {{"Domain", "Range"}, {"Domain2", "Range"}}};
}

static Signature<Kind> range() {
  return {{"Range"}, {{"Domain", "Range"}}};
}

static Signature<Kind> from_domain_and_range() {
  return {{"Domain", "Range"}, {{"Domain"}, {"Range"}}};
}

static Signature<Kind> from_range_and_domain() {
  return {{"Domain2", "Domain"}, {{"Domain"}, {"Domain2"}}};
}

static Signature<Kind> reverse() {
  return {{"Range", "Domain"}, {{"Domain", "Range"}}};
}

static Signature<Kind> test_map() {
  return {{"Domain", "Domain"}, {{"Domain", "Domain"}, {"Domain", "Range2"}}};
}

static Signature<Kind> range_product() {
  return {{"Domain", {"Range", "Range3"}},
          {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

static Signature<Kind> flat_range_product() {
  return {{"Domain", "Range2"}, {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

static Signature<Kind> curry() {
  return {{"Domain2", {"Range2", "Range"}}, {{{"Domain2", "Range2"}, "Range"}}};
}

static Signature<Kind> uncurry() {
  return {{{"Domain", "Domain2"}, "Range2"},
          {{"Domain", {"Domain2", "Range2"}}}};
}

static Signature<Kind> domain_factor_domain() {
  return {{"Domain2", "Range"}, {{{"Domain2", "Range2"}, "Range"}}};
}

static Signature<Kind> wrap() {
  return {{{"Domain", "Range"}}, {{"Domain", "Range"}}};
}

static Signature<Kind> wrap_binary() {
  return {{{"Domain", "Range"}}, {{"Domain"}, {"Range"}}};
}

static Signature<Kind> unwrap() {
  return {{"MapDomain", "MapRange"}, {{{"MapDomain", "MapRange"}}}};
}

static Signature<Kind> zip() {
  return {{{"WrappedDomain1", "WrappedDomain2"},
           {"WrappedRange1", "WrappedRange2"}},
          {{{"WrappedDomain1", "WrappedRange1"},
            {"WrappedDomain2", "WrappedRange2"}}}};
}

static Signature<Kind> get_map_anonymous() {
  return {{"Domain", "Anonymous"}, {{"Domain", "Range"}}};
}

static Signature<Kind> add_map_anonymous() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain", "Anonymous"}}};
}

static Signature<Kind> add_range_anonymous() {
  return {{"Domain", "Range"}, {{"Range"}, {"Domain", "Anonymous"}}};
}

static const std::unordered_map<std::string, std::vector<Signature<Kind>>>
    signatures{
        {"add_param", {modify_params_unary(), modify_set_unary()}},
        {"align_params", {modify_set_params(), modify_map_params()}},
        {"apply", {apply_set(), apply_range()}},
        {"apply_domain", {apply_domain()}},
        {"preimage_domain", {preimage_domain()}},
        {"pullback", {preimage_domain()}},
        {"apply_range", {apply_range()}},
        {"coalesce",
         {modify_params_unary(), modify_set_unary(), modify_map_unary()}},
        {"eq_at", {test_map()}},
        {"ge_set", {set_binary_params()}},
        {"get_space",
         {modify_params_unary(), modify_set_unary(), modify_map_unary()}},
        {"gist", {modify_set_binary(), modify_map()}},
        {"intersect",
         {modify_params_binary(), modify_set_binary(), modify_map()}},
        {"intersect_domain", {modify_domain()}},
        {"intersect_range", {modify_range()}},
        {"intersect_params", {modify_set_params(), modify_map_params()}},
        {"lt_set", {domain_binary()}},
        {"le_set", {domain_binary()}},
        {"eq_set", {domain_binary()}},
        {"lt_map", {domain_binary_map()}},
        {"gt_map", {domain_binary_map()}},
        {"params", {set_params(), map_params()}},
        {"from_params", {set_from_params()}},
        {"map_from_set", {map_from_set()}},
        {"add_named_tuple_id_ui", {set_from_params()}},
        {"add_unnamed_tuple_ui", {set_from_params(), map_from_domain()}},
        {"product", {wrap_binary()}},
        {"map_from_domain_and_range", {from_domain_and_range()}},
        {"domain", {domain()}},
        {"domain_map", {domain_map()}},
        {"range", {range()}},
        {"reverse", {reverse()}},
        {"subtract", {modify_set_binary(), modify_map()}},
        {"unbind_params_insert_domain", {from_range_and_domain()}},
        {"sum", {modify_map()}},
        {"unite", {modify_set_binary(), modify_map()}},
        {"union_add", {modify_map()}},
        {"range_product", {range_product()}},
        {"flat_range_product", {flat_range_product()}},
        {"curry", {curry()}},
        {"uncurry", {uncurry()}},
        {"domain_factor_domain", {domain_factor_domain()}},
        {"wrap", {wrap()}},
        {"unwrap", {unwrap()}},
        {"zip", {zip()}},
        {"add", {modify_set_binary(), modify_map()}},
        {"sub", {modify_set_binary(), modify_map()}},
        {"mul", {modify_set_binary()}},
        {"div", {modify_set_binary()}},
        {"mod",
         {modify_set_binary(),
          modify_map(),
          modify_set_unary(),
          modify_map_unary()}},
        {"get_at", {modify_set_unary(), modify_map_unary()}},
        {"get_map_list", {modify_map_unary()}},
        {"get_aff", {get_map_anonymous()}},
        {"set_aff", {add_map_anonymous()}},
        {"get_aff_list", {get_map_anonymous()}},
        {"get_union_pw_aff", {get_map_anonymous()}},
        {"set_union_pw_aff", {add_map_anonymous()}},
        {"get_union_pw_aff_list", {get_map_anonymous()}},
        {"pos_set", {set_params(), domain()}},
        {"nonneg_set", {set_params(), domain()}},
        {"zero_set", {set_params(), domain()}},
        {"zero_union_set", {set_params(), domain()}},
        {"add_constant", {modify_set_unary(), modify_map_unary()}},
        {"add_constant_si", {modify_set_unary(), modify_map_unary()}},
        {"set_val", {modify_set_unary()}},
        {"floor", {modify_map_unary()}},
        {"neg", {modify_set_unary(), modify_map_unary()}},
        {"drop", {modify_map_unary()}},
        {"scale", {modify_map_unary(), modify_range()}},
        {"scale_down", {modify_map_unary(), modify_range()}},
        {"set_set_tuple_id", {change_wrapped_set(), change_set()}},
        {"set_tuple_id", {change_wrapped_set(), change_set()}},
        {"set_range_tuple_id", {change_set(), change_range()}},
        {"get_range_stride_info", {get_map_anonymous()}},
        {"get_offset", {modify_map_unary()}},
        {"get_range_simple_fixed_box_hull", {modify_map_unary()}},
        {"set_may_source", {modify_map()}},
        {"set_schedule", {modify_map_unary()}},
        {"compute_flow", {modify_map_unary()}},
        {"get_may_dependence", {map_on_domain()}},
    };

static const std::
    map<std::pair<std::string, std::string>, std::vector<Signature<Kind>>>
        specificSignatures{
            {{"set", "identity"}, {map_from_set()}},
            {{"union_set", "get_space"}, {set_params()}},
            {{"union_map", "get_space"}, {set_params()}},
            {{"union_pw_aff", "get_space"}, {set_params()}},
            {{"union_pw_multi_aff", "get_space"}, {set_params()}},
            {{"union_set", "universe"}, {modify_set_unary()}},
            {{"union_map", "universe"}, {modify_map_unary()}},
            // should be called "gist_domain"
            {{"multi_union_pw_aff", "gist"}, {modify_domain()}},
            {{"multi_union_pw_aff", "get_space"}, {range()}},
            {{"aff_list", "reverse"}, {modify_map_unary()}},
            {{"pw_aff_list", "reverse"}, {modify_map_unary()}},
            {{"union_pw_aff_list", "reverse"}, {modify_map_unary()}},
            {{"map_list", "reverse"}, {modify_map_unary()}},
        };

static const std::unordered_map<std::string, std::vector<Signature<Kind>>>
    staticSignatures{
        {"from", {modify_map_unary()}},
        {"identity", {modify_map_unary()}},
        {"param_on_domain_space", {set_from_params()}},
        {"param_on_domain", {map_from_domain()}},
        {"empty",
         {modify_params_unary(), modify_set_unary(), modify_map_unary()}},
        {"universe",
         {modify_params_unary(), modify_set_unary(), modify_map_unary()}},
        {"zero", {modify_set_unary(), modify_map_unary()}},
        {"zero_on_domain", {map_from_domain()}},
        {"from_domain", {map_from_domain()}},
    };

static const std::
    map<std::pair<std::string, std::string>, std::vector<Signature<Kind>>>
        specificStaticSignatures{
            {{"multi_aff", "domain_map"}, {static_domain_map()}},
            {{"multi_aff", "range_map"}, {static_range_map()}},
            {{"multi_aff", "wrapped_range_map"}, {static_wrapped_range_map()}},
            {{"union_set", "empty"}, {set_from_params()}},
            {{"union_map", "empty"}, {map_from_params()}},
        };

struct Constructor {
  std::vector<Type> argTypes;
  std::vector<Signature<Kind>> signatures;
};

static const std::unordered_map<std::string, std::vector<Constructor>>
    constructors{
        {"multi_id", {{{"space", "id_list"}, {modify_set_unary()}}}},
        {"multi_val", {{{"space", "val_list"}, {modify_set_unary()}}}},
        {"multi_aff", {{{"space", "aff_list"}, {add_map_anonymous()}}}},
        {"union_pw_aff", {{{"union_set", "val"}, {map_from_domain()}}}},
        {"multi_union_pw_aff",
         {{{"space", "union_pw_aff_list"}, {add_range_anonymous()}},
          {{"union_set", "multi_val"}, {from_domain_and_range()}}}},
        {"map", {{{"multi_aff"}, {modify_map_unary()}}}},
        {"union_map", {{{"map"}, {modify_map_unary()}}}},
        {"union_set", {{{"set"}, {modify_set_unary()}}}},
        {"pw_aff", {{{"aff"}, {modify_set_unary(), modify_map_unary()}}}},
        {"aff_list",
         {{{"aff"}, {modify_set_unary(), modify_map_unary()}},
          {{"ctx", "int"}, {create_set(), create_map()}}}},
        // should be replaced by constructor without int argument
        {"space", {{{"ctx", "int"}, {create_params()}}}},
        {"union_pw_aff_list", {{{"ctx", "int"}, {create_set(), create_map()}}}},
        {"union_access_info", {{{"union_map"}, {modify_map_unary()}}}},
    };

static bool isForeach(const std::string& name) {
  return name.find("foreach_") != std::string::npos;
}

using Subs = std::map<std::string, BaseKind>;

static std::set<std::string>
collect(const Kind& kind, const Subs& subs, std::set<std::string> set = {});

static std::set<std::string> collect(
    const BaseKind& base,
    const Subs& subs,
    std::set<std::string> set = {}) {
  if (base.children.size() == 0) {
    if (subs.count(base.name) != 0) {
      set = collect(subs.at(base.name), {}, set);
    } else if (base.name != "Anonymous") {
      set.insert(base.name);
    }
  } else {
    for (const auto& el : base.children) {
      set = collect(el, subs, set);
    }
  }
  return set;
}

static std::set<std::string>
collect(const Kind& kind, const Subs& subs, std::set<std::string> set) {
  for (auto base : kind) {
    set = collect(base, subs, set);
  }
  return set;
}

static std::set<std::string> collect(
    const Signature<Kind>& signature,
    const Subs& subs) {
  auto set = collect(signature.returnType, subs);
  for (auto arg : signature.argTypes) {
    set = collect(arg, subs, set);
  }
  return set;
}

static void printTemplateList(
    const std::set<std::string> set,
    const std::string& qualifier) {
  std::cout << "<";
  bool first = true;
  for (auto s : set) {
    if (!first) {
      std::cout << ", ";
    }
    std::cout << qualifier << s;
    first = false;
  }
  std::cout << ">";
}

static void
print(std::ostream& os, const BaseKind& base, const Subs& subs = {});

static void
print(std::ostream& os, const std::string& s, const Subs& subs = {}) {
  if (subs.count(s) != 0) {
    print(os, subs.at(s));
  } else {
    os << s;
  }
}

static void print(std::ostream& os, const BaseKind& base, const Subs& subs) {
  if (base.children.size() == 3) {
    if (base.children[0] == "Anonymous") {
      os << "Pair<";
    } else {
      os << "NamedPair<";
      print(os, base.children[0], subs);
      os << ",";
    }
    print(os, base.children[1], subs);
    os << ",";
    print(os, base.children[2], subs);
    os << ">";
  } else {
    print(os, base.name, subs);
  }
}

template <typename T>
static void printTemplateList(
    const std::vector<T> list,
    const std::string& qualifier,
    const Subs& subs = {}) {
  std::cout << "<";
  for (unsigned i = 0; i < list.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << qualifier;
    print(std::cout, list[i], subs);
  }
  std::cout << ">";
}

template <typename T>
static void printTemplate(const T& t) {
  std::cout << "template ";
  printTemplateList(t, "typename ");
  std::cout << "\n";
}

static void printClassDeclaration(const std::string& name, const Kind& kind) {
  printTemplate(collect(kind, {}));
  std::cout << "struct " << name;
  printTemplateList(kind, "");
}

static void printForwardDeclarations() {
  for (auto kvp : classes) {
    std::cout << "\n";
    std::cout << "template <typename...>\n";
    std::cout << "struct " << kvp.second.name;
    std::cout << ";\n";
  }
}

static BaseKind specialize(const BaseKind& base, const Subs& subs) {
  if (base.children.size() == 0) {
    if (subs.count(base.name) != 0) {
      return subs.at(base.name);
    } else {
      return base;
    }
  } else {
    return BaseKind{specialize(base.children[0], subs),
                    specialize(base.children[1], subs),
                    specialize(base.children[2], subs)};
  }
}

static Kind specialize(const Kind& kind, const Subs& subs) {
  if (subs.size() == 0) {
    return kind;
  }
  Kind specialized;
  for (auto base : kind) {
    specialized.emplace_back(specialize(base, subs));
  }
  return specialized;
}

static std::vector<Kind> specialize(
    const std::vector<Kind>& vector,
    const Subs& subs) {
  std::vector<Kind> specialized;
  for (auto kind : vector) {
    specialized.emplace_back(specialize(kind, subs));
  }
  return specialized;
}

static Signature<Kind> specialize(
    const Signature<Kind>& signature,
    const Subs& subs) {
  return {specialize(signature.returnType, subs),
          specialize(signature.argTypes, subs)};
}

static void printExtraTemplate(
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Subs& subs,
    bool isStatic) {
  auto classBase = collect(classKind, {});
  classBase.insert("Anonymous");
  auto signatureBase = collect(signature, subs);
  std::vector<std::string> extra;
  for (auto base : signatureBase) {
    if (classBase.count(base) == 0) {
      extra.emplace_back(base);
    }
  }
  if (extra.size() != 0) {
    printTemplate(extra);
  }
}

static void
printType(const Type& type, const Kind& kind, const Subs& subs = {}) {
  if (classes.count(type) == 0) {
    std::cout << type;
  } else {
    const auto& returnType = classes.at(type);
    std::cout << returnType.name;
    printTemplateList(kind, "", subs);
  }
}

static void printReturnType(
    const Signature<Kind>& signature,
    const Method& method,
    const Subs& subs = {}) {
  printType(method.signature.returnType, signature.returnType, subs);
}

static Subs specializer(
    const std::vector<BaseKind>& dst,
    const std::vector<BaseKind>& src,
    Subs subs = {}) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i].children.size() == 0) {
      subs.emplace(src[i].name, dst[i]);
    } else if (src[i].children.size() == dst[i].children.size()) {
      subs = specializer(dst[i].children, src[i].children, subs);
    }
  }
  return subs;
}

static Signature<Kind> specialize(
    const Signature<Kind>& signature,
    const Kind& classKind) {
  Subs subs = specializer(classKind, signature.argTypes[0]);
  return specialize(signature, subs);
}

static bool printMethod(
    const std::string& base,
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Method& method,
    const Subs& subs,
    bool isStatic = false) {
  auto specializedSignature =
      isStatic ? signature : specialize(signature, classKind);
  const auto& match = isStatic ? specializedSignature.returnType
                               : specializedSignature.argTypes[0];
  auto specializedMatch = specialize(match, subs);
  if (specializedMatch != classKind) {
    return false;
  }
  printExtraTemplate(classKind, specializedSignature, subs, isStatic);
  if (isStatic) {
    std::cout << "static ";
  }
  std::cout << "inline ";
  printReturnType(specializedSignature, method, subs);
  std::cout << " ";
  std::cout << method.name;
  std::cout << "(";
  size_t j = isStatic ? 0 : 1;
  for (size_t i = 0; i < method.signature.argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "const ";
    const Type& type = method.signature.argTypes[i];
    if (classes.count(type) == 0) {
      std::cout << type;
    } else {
      printType(type, specializedSignature.argTypes[j++], subs);
    }
    std::cout << "& arg" << i;
  }
  std::cout << ")";
  if (!isStatic) {
    std::cout << " const";
  }
  std::cout << " {\n";
  std::cout << "auto res = ";
  if (!isStatic) {
    std::cout << "this->";
  }
  std::cout << base << "::" << method.name << "(";
  for (size_t i = 0; i < method.signature.argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "arg" << i;
  }
  std::cout << ");\n";
  std::cout << "return ";
  printReturnType(specializedSignature, method, subs);
  std::cout << "(res);\n";
  std::cout << "}\n";
  return true;
}

static void printTo(
    const Signature<Kind>& signature,
    const Method& method,
    const Subs& subs) {
  std::cout << "inline ";
  printReturnType(signature, method, subs);
  std::cout << " to" << classes.at(method.signature.returnType).name
            << "() const {\n";
  std::cout << "return ";
  printReturnType(signature, method, subs);
  std::cout << "::from(*this);\n";
  std::cout << "}\n";
}

static void printAs(
    const Signature<Kind>& signature,
    const Method& method,
    const Subs& subs) {
  std::cout << "inline ";
  printReturnType(signature, method, subs);
  std::cout << " as" << classes.at(method.signature.returnType).name
            << "() const {\n";
  std::cout << "return ";
  printReturnType(signature, method, subs);
  std::cout << "(*this);\n";
  std::cout << "}\n";
}

static void printForeach(
    const std::string& base,
    const Kind& classKind,
    const Method& method) {
  const auto& fn = method.signature.argTypes[0];
  auto open = fn.find("(");
  auto close = fn.find(")", open + 1);
  if (close == std::string::npos) {
    return;
  }
  auto argType = fn.substr(open + 1, close - (open + 1));
  if (classes.count(argType) == 0) {
    return;
  }
  std::cout << "inline void " << method.name << "(";
  std::cout << fn.substr(0, open + 1);
  printType(argType, classKind);
  std::cout << fn.substr(close);
  std::cout << "& fn) const {\n";
  std::cout << "auto lambda = [fn](" << argType << " arg) -> void {\n";
  std::cout << "fn(";
  printType(argType, classKind);
  std::cout << "(arg));";
  std::cout << "};\n";
  std::cout << "this->" << base << "::" << method.name << "(lambda);\n";
  std::cout << "}\n";
}

static bool matches(
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Method& method) {
  if (signature.argTypes[0].size() != classKind.size()) {
    return false;
  }
  size_t count = 0;
  for (const auto& type : method.signature.argTypes) {
    if (classes.count(type) != 0) {
      ++count;
    }
  }
  return signature.argTypes.size() == 1 + count;
}

static void printMethods(
    const std::string& base,
    const Kind& classKind,
    const std::vector<Method>& methods,
    const Subs& subs) {
  for (auto method : methods) {
    if (specificSignatures.count({base, method.name}) != 0) {
      for (const auto& signature : specificSignatures.at({base, method.name})) {
        if (matches(classKind, signature, method)) {
          printMethod(base, classKind, signature, method, subs);
        }
      }
    } else if (specificStaticSignatures.count({base, method.name}) != 0) {
      for (const auto& signature :
           specificStaticSignatures.at({base, method.name})) {
        if (signature.returnType.size() == classKind.size()) {
          printMethod(base, classKind, signature, method, subs, true);
        }
      }
    } else if (signatures.count(method.name) != 0) {
      for (const auto& signature : signatures.at(method.name)) {
        if (matches(classKind, signature, method)) {
          if (printMethod(base, classKind, signature, method, subs)) {
            break;
          }
        }
      }
    } else if (staticSignatures.count(method.name) != 0) {
      for (const auto& signature : staticSignatures.at(method.name)) {
        if (signature.returnType.size() == classKind.size()) {
          printMethod(base, classKind, signature, method, subs, true);
        }
      }
    } else if (
        method.name == "#to" &&
        classes.count(method.signature.returnType) == 1) {
      for (auto returnKind : classes.at(method.signature.returnType).kinds) {
        if (classKind.size() == 2 && returnKind.size() == 2) {
          printTo(modify_map_unary(), method, subs);
        }
      }
    } else if (
        method.name == "#as" &&
        classes.count(method.signature.returnType) == 1) {
      for (auto returnKind : classes.at(method.signature.returnType).kinds) {
        if (classKind.size() == returnKind.size()) {
          for (const auto& constructor :
               constructors.at(method.signature.returnType)) {
            for (const auto& signature : constructor.signatures) {
              if (constructor.argTypes[0] == base &&
                  signature.returnType.size() == classKind.size()) {
                printAs(signature, method, subs);
              }
            }
          }
        }
      }
    } else if (isForeach(method.name)) {
      printForeach(base, classKind, method);
    }
  }
}

static void printConstructor(
    const std::string& base,
    const std::string& className,
    const Kind& classKind,
    const std::vector<Type>& argTypes,
    const Signature<Kind>& signature,
    const Subs& subs) {
  std::cout << className << "(";
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "const ";
    printType(argTypes[i], signature.argTypes[i], subs);
    std::cout << "& arg" << i;
  }
  std::cout << ") : " << base << "(";
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "arg" << i;
  }
  std::cout << ") {}\n";
}

static void printOneDefinition(
    const std::string& base,
    const std::string& className,
    const Kind& classKind,
    const Exported& exported,
    const Subs& subs) {
  std::cout << "\n";
  printClassDeclaration(className, classKind);
  std::cout << " : public " << base << " {\n";
  std::cout << className << "() = default;\n";
  std::cout << "explicit " << className << "(const " << base
            << "& obj) : " << base << "(obj) {}\n";
  if (constructors.count(base) != 0) {
    for (const auto& constructor : constructors.at(base)) {
      for (const auto& signature : constructor.signatures) {
        if (classKind.size() == signature.returnType.size()) {
          printConstructor(
              base,
              className,
              classKind,
              constructor.argTypes,
              signature,
              subs);
        }
      }
    }
  }
  if (exported.count(base) != 0) {
    printMethods(base, classKind, exported.at(base), subs);
  }
  std::cout << "};\n";
}

static void printDefinition(
    const std::string& base,
    const std::string& className,
    const Kind& classKind,
    const Exported& exported) {
  std::set<Kind> kinds{classKind};
  if (exported.count(base) != 0) {
    for (auto method : exported.at(base)) {
      if (specificStaticSignatures.count({base, method.name}) != 0) {
        for (const auto& signature :
             specificStaticSignatures.at({base, method.name})) {
          if (signature.returnType.size() == classKind.size() &&
              signature.returnType != classKind) {
            kinds.emplace(signature.returnType);
          }
        }
      } else if (signatures.count(method.name) != 0) {
        for (const auto& signature : signatures.at(method.name)) {
          if (matches(classKind, signature, method) &&
              signature.argTypes[0] != classKind) {
            kinds.emplace(signature.argTypes[0]);
          }
        }
      }
    }
  }
  for (auto kind : kinds) {
    Subs subs;
    for (size_t i = 0; i < classKind.size(); ++i) {
      if (classKind[i] != kind[i]) {
        subs.emplace(classKind[i].name, kind[i]);
      }
    }
    printOneDefinition(base, className, kind, exported, subs);
  }
}

static void printDefinitions(const Exported& exported) {
  printDefinition("space", "Space", params_type(), exported);
  printDefinition("set", "Set", params_type(), exported);
  for (auto kvp : classes) {
    for (auto kind : kvp.second.kinds) {
      if (kind.size() == 0 && kvp.first != "space" && kvp.first != "set") {
        printDefinition(kvp.first, kvp.second.name, kind, exported);
      }
    }
  }
  for (auto kvp : classes) {
    for (auto kind : kvp.second.kinds) {
      if (kind.size() != 0) {
        printDefinition(kvp.first, kvp.second.name, kind, exported);
      }
    }
  }
}

static std::string extractArg(std::string arg) {
  size_t start = 0;
  constexpr auto constStr = "const ";
  if (arg.find(constStr) != std::string::npos) {
    start += strlen(constStr);
  }
  return dropIslNamespace(arg.substr(0, arg.find(" ", start)));
}

static std::vector<std::string> splitArgs(const std::string& args) {
  std::vector<std::string> list;
  size_t pos, old = 0;

  while ((pos = args.find(", ", old)) != std::string::npos) {
    list.emplace_back(extractArg(args.substr(old, pos)));
    old = pos + 2;
  }
  if (args.length() > 0) {
    list.emplace_back(extractArg(args.substr(old)));
  }
  return list;
}

int main(int argc, char** argv) {
  Exported exported;
  for (std::string line; std::getline(std::cin, line);) {
    std::regex declaration("^([a-z_:]+) (.*)::([a-z_]+)\\((.*)\\)(.*const)?$");
    std::smatch match;
    if (!std::regex_match(line, match, declaration)) {
      continue;
    }

    auto retType = dropIslNamespace(match[1].str());
    auto className = dropIslNamespace(match[2].str());
    auto name = match[3].str();
    auto args = splitArgs(match[4].str());

    if (name == "from" && args.size() == 1) {
      exported[args[0]].emplace_back(Method{"#to", {retType, args}});
    }
    if (signatures.count(name) == 0 &&
        specificSignatures.count({className, name}) == 0 &&
        staticSignatures.count(name) == 0 &&
        specificStaticSignatures.count({className, name}) == 0 &&
        !isForeach(name)) {
      continue;
    }

    exported[className].emplace_back(Method{name, {retType, args}});
  }
  for (auto kvp : constructors) {
    for (const auto& constructor : kvp.second) {
      const auto& args = constructor.argTypes;
      if (args.size() == 1) {
        exported[args[0]].emplace_back(Method{"#as", {kvp.first, args}});
      }
    }
  }

  std::cout << header;
  printForwardDeclarations();
  printDefinitions(exported);
  std::cout << footer;

  return EXIT_SUCCESS;
}
