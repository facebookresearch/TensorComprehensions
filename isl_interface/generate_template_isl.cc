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

/*
 * Fixed initial part of the output.
 */
constexpr auto header = R"CPP(
// Template kind argument corresponding to a (collection of) anonymous space(s)
// or to a missing name.
struct Anonymous;

// Template kind argument wrapping a pair of kinds and a given name.
template <typename Name, typename First, typename Second>
struct NamedPair;

// Template kind argument wrapping a pair of kinds without name.
template <typename First, typename Second>
using Pair = NamedPair<Anonymous, First, Second>;
)CPP";

/*
 * Fixed final part of the output.
 */
constexpr auto footer = R"CPP(
// The *AffOn type represent affine functions on a domain.
// The kind of such an affine function is that of the domain combined
// with an anonymous range.
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

/*
 * Drop the "isl::" namespace specifier, if any.
 */
static std::string dropIslNamespace(std::string type) {
  return std::regex_replace(type, std::regex("isl::"), "");
}

/*
 * The type of an isl object in the standard C++ interface.
 */
using IslType = std::string;

/*
 * A Kind represents a collection of spaces with a fixed number of tuples.
 * Each tuple represents a collections of space tuples and is identified
 * by a name and, optionally, two children.
 * A missing name is represented by the name "Anonymous".
 */
struct Tuple {
  Tuple(const char* name) : name(name) {}
  Tuple(const Tuple& t1, const Tuple& t2)
      : name("Anonymous"), children({t1, t2}) {}
  Tuple(const char* name, const Tuple& t1, const Tuple& t2)
      : name(name), children({t1, t2}) {}
  std::string name;
  std::vector<Tuple> children;
  bool operator<(const Tuple& other) const {
    if (name != other.name) {
      return name < other.name;
    }
    if (children.size() != other.children.size()) {
      return children.size() < other.children.size();
    }
    for (size_t i = 0; i < children.size(); ++i) {
      if (children[i] != other.children[i]) {
        return children[i] < other.children[i];
      }
    }
    return false;
  }
  bool operator==(const Tuple& other) const {
    return name == other.name && children == other.children;
  }
  bool operator!=(const Tuple& other) const {
    return !(*this == other);
  }
};
using Kind = std::vector<Tuple>;

/*
 * A generic signature, either in terms of the isl type or
 * in terms of the space kinds.
 */
template <typename Type>
struct Signature {
  Type returnType;
  std::vector<Type> argTypes;
};

/*
 * A method exported by the standard C++ interface.
 */
struct Method {
  std::string name;
  Signature<IslType> signature;
};

/*
 * The collection of exported C++ methods, per class name,
 * along with special "#to" and "#as" methods.
 * A "#to" method is added for each "from" method (with arguments reversed),
 * while an "#as" method is added for each conversion constructor.
 */
using Exported = std::unordered_map<std::string, std::vector<Method>>;

/*
 * A class is the templated isl interface.
 * The "kinds" array specifies all the different kinds
 * that this class may have.
 */
struct Class {
  std::string name;
  std::vector<Kind> kinds;
};

/*
 * A Kind with zero tuples.
 */
static Kind paramsKind() {
  return {};
}

/*
 * A Kind with a single tuple.
 */
static Kind setKind() {
  return {"Domain"};
}

/*
 * A Kind with two tuples.
 */
static Kind mapKind() {
  return {"Domain", "Range"};
}

/*
 * For each class in the regular C++ interface,
 * the corresponding class in the templated interface,
 * including a list of possible basic kinds.
 */
static const std::unordered_map<std::string, Class> classes{
    {"space", {"Space", {paramsKind(), setKind(), mapKind()}}},
    {"multi_id", {"MultiId", {setKind()}}},
    {"multi_val", {"MultiVal", {setKind()}}},
    {"set", {"Set", {paramsKind(), setKind()}}},
    {"map", {"Map", {mapKind()}}},
    {"aff", {"Aff", {setKind(), mapKind()}}},
    {"aff_list", {"AffList", {setKind(), mapKind()}}},
    {"pw_aff", {"PwAff", {setKind(), mapKind()}}},
    {"union_pw_aff", {"UnionPwAff", {setKind(), mapKind()}}},
    {"multi_aff", {"MultiAff", {setKind(), mapKind()}}},
    {"pw_aff_list", {"PwAffList", {setKind(), mapKind()}}},
    {"union_pw_aff_list", {"UnionPwAffList", {setKind(), mapKind()}}},
    {"multi_union_pw_aff", {"MultiUnionPwAff", {setKind(), mapKind()}}},
    {"union_pw_multi_aff", {"UnionPwMultiAff", {setKind(), mapKind()}}},
    {"union_set", {"UnionSet", {paramsKind(), setKind()}}},
    {"union_map", {"UnionMap", {mapKind()}}},
    {"map_list", {"MapList", {mapKind()}}},
    {"union_access_info", {"UnionAccessInfo", {mapKind()}}},
    {"union_flow", {"UnionFlow", {mapKind()}}},
    {"stride_info", {"StrideInfo", {mapKind()}}},
    {"fixed_box", {"FixedBox", {mapKind()}}},
};

/*
 * Create an object living in the parameter space.
 */
static Signature<Kind> createParams() {
  return {{}, {}};
}

/*
 * Create an object living in a collection of set spaces.
 */
static Signature<Kind> createSet() {
  return {{"Domain"}, {}};
}

static Signature<Kind> change_wrapped_set() {
  return {{{"ModifiedWrap", "WrappedDomain", "WrappedRange"}},
          {{{"Wrap", "WrappedDomain", "WrappedRange"}}}};
}

/*
 * Create an object living in a collection of map spaces.
 */
static Signature<Kind> createMap() {
  return {{"Domain", "Range"}, {}};
}

/*
 * Take an object living in a collection of set spaces and
 * a map with this collection as domain and return
 * an object in the corresponding range.
 */
static Signature<Kind> applySet() {
  return {{"Range"}, {{"Domain"}, {"Domain", "Range"}}};
}

/*
 * Take an object living in a collection of map spaces and
 * another map with with the same domain and return
 * an object in the original collection with the domain
 * replaced by the range of the second map.
 */
static Signature<Kind> applyDomain() {
  return {{"Range3", "Range"}, {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

/*
 * Take an object living in a collection of map spaces and
 * another map with with the range of the first as domain and return
 * an object in the original collection with the range
 * replaced by the range of the second map.
 */
static Signature<Kind> applyRange() {
  return {{"Domain", "Range2"}, {{"Domain", "Range"}, {"Range", "Range2"}}};
}

/*
 * Perform the same operation as applyRange, but with the arguments
 * interchanged.
 */
static Signature<Kind> preimageDomain() {
  return {{"Domain2", "Range"}, {{"Domain", "Range"}, {"Domain2", "Domain"}}};
}

/*
 * Return an object that, like the input, lives in a parameter space.
 */
static Signature<Kind> updateParams() {
  return {{}, {{}}};
}

/*
 * Return an object that, like the two inputs, lives in a parameter space.
 */
static Signature<Kind> updateParamsBinary() {
  return {{}, {{}, {}}};
}

/*
 * Return an object that lives in the same collection of set spaces
 * as the first input and that takes an object living in a parameter space
 * as second input.
 */
static Signature<Kind> updateSetParams() {
  return {{"Domain"}, {{"Domain"}, {}}};
}

/*
 * Return an object that lives in the same collection of set spaces
 * as the input.
 */
static Signature<Kind> updateSet() {
  return {{"Domain"}, {{"Domain"}}};
}

/*
 * Return an object that lives in a (potentially) different
 * collection of set spaces compared to the input.
 */
static Signature<Kind> changeSet() {
  return {{"ModifiedDomain"}, {{"Domain"}}};
}

/*
 * Return an object that lives in the same collection of set spaces
 * as the two inputs.
 */
static Signature<Kind> updateSetBinary() {
  return {{"Domain"}, {{"Domain"}, {"Domain"}}};
}

/*
 * Return an object that lives in the same collection of map spaces
 * as the input.
 */
static Signature<Kind> updateMap() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}}};
}

/*
 * Return an object that lives in the same collection of map spaces
 * as the two inputs.
 */
static Signature<Kind> updateMapBinary() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain", "Range"}}};
}

/*
 * Return an object that lives in the same collection of map spaces
 * as the first input and that takes an object living in a parameter space
 * as second input.
 */
static Signature<Kind> updateMapParams() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {}}};
}

/*
 * Return an object that lives in the same collection of map spaces
 * as the first input and that takes an object living in the domain
 * of the first input as second input.
 */
static Signature<Kind> updateMapDomain() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain"}}};
}

/*
 * Return an object that lives in the same collection of map spaces
 * as the first input and that takes an object living in the range
 * of the first input as second input.
 */
static Signature<Kind> updateMapRange() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Range"}}};
}

/*
 * Return an object that lives in a collection of map spaces
 * where the range is potentially different from that of the input.
 */
static Signature<Kind> changeRange() {
  return {{"Domain", "ModifiedRange"}, {{"Domain", "Range"}}};
}

/*
 * Return a map with the input collection of spaces as both domain and range.
 */
static Signature<Kind> mapOnSet() {
  return {{"Domain", "Domain"}, {{"Domain"}}};
}

/*
 * Return a map with the domain of the input collection of spaces
 * as both domain and range.
 */
static Signature<Kind> mapOnDomain() {
  return {{"Domain", "Domain"}, {{"Domain", "Range"}}};
}

/*
 * Return a set given an object living in a parameter space.
 */
static Signature<Kind> setFromParams() {
  return {{"Domain"}, {{}}};
}

/*
 * Return a map given an object living in a parameter space.
 */
static Signature<Kind> mapFromParams() {
  return {{"Domain", "Range"}, {{}}};
}

/*
 * Return a map with the input collection of spaces as domain.
 */
static Signature<Kind> mapFromDomain() {
  return {{"Domain", "Range"}, {{"Domain"}}};
}

/*
 * Given an object living in a collection of set spaces,
 * return an object living in a parameter space.
 */
static Signature<Kind> setParams() {
  return {{}, {{"Domain"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object living in a parameter space.
 */
static Signature<Kind> mapParams() {
  return {{}, {{"Domain", "Range"}}};
}

/*
 * Given two objects living in the collection of set spaces,
 * return an object living in a parameter space.
 */
static Signature<Kind> setParamsBinary() {
  return {{}, {{"Domain"}, {"Domain"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object living in the domain.
 */
static Signature<Kind> mapDomain() {
  return {{"Domain"}, {{"Domain", "Range"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object that maps such objects to their domains.
 */
static Signature<Kind> mapDomainMap() {
  return {{{"Domain", "Range"}, "Domain"}, {{"Domain", "Range"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object that maps such objects to their domains.
 * For use on static methods.
 */
static Signature<Kind> mapDomainMapStatic() {
  return {{{"Range", "Domain2"}, "Range"}, {{"Range", "Domain2"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object that maps such objects to their ranges.
 * For use on static methods.
 */
static Signature<Kind> mapRangeMapStatic() {
  return {{{"Domain2", "Range"}, "Range"}, {{"Domain2", "Range"}}};
}

/*
 * Given an object living in a collection of set spaces wrapping map spaces,
 * return an object that maps such objects to their wrapped ranges.
 * For use on static methods.
 */
static Signature<Kind> setWrappedRangeMapStatic() {
  return {{{"Wrap", "WrappedDomain", "WrappedRange"}, "WrappedRange"},
          {{{"Wrap", "WrappedDomain", "WrappedRange"}}}};
}

/*
 * Given a pair of objects living in the same collection of map spaces,
 * return an object living in their domain.
 */
static Signature<Kind> mapDomainBinary() {
  return {{"Domain"}, {{"Domain", "Range"}, {"Domain", "Range"}}};
}

/*
 * Given a pair of objects living in the collections of map spaces
 * with the same range,
 * return an object living in a map between their domains.
 */
static Signature<Kind> mapDomainBinaryMap() {
  return {{"Domain", "Domain2"}, {{"Domain", "Range"}, {"Domain2", "Range"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object living in the range.
 */
static Signature<Kind> mapRange() {
  return {{"Range"}, {{"Domain", "Range"}}};
}

/*
 * Given an object living in a collection of domain set spaces and
 * an object living in a collection of range set spaces,
 * return the combined collection of map spaces.
 */
static Signature<Kind> mapFromDomainAndRange() {
  return {{"Domain", "Range"}, {{"Domain"}, {"Range"}}};
}

/*
 * Performs the same operation as mapFromDomainAndRange,
 * but with the arguments reversed.
 */
static Signature<Kind> mapFromRangeAndDomain() {
  return {{"Domain2", "Domain"}, {{"Domain"}, {"Domain2"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return one where domain and range have been interchanged.
 */
static Signature<Kind> mapReverse() {
  return {{"Range", "Domain"}, {{"Domain", "Range"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * relating the same collection of set spaces as well
 * another map on the same domain,
 * return an object living in the same collection as the first input.
 */
static Signature<Kind> mapTest() {
  return {{"Domain", "Domain"}, {{"Domain", "Domain"}, {"Domain", "Range2"}}};
}

/*
 * Given a pair of objects living in collections of map spaces
 * with the same domain, return an object in the range product.
 */
static Signature<Kind> mapRangeProduct() {
  return {{"Domain", {"Range", "Range3"}},
          {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

/*
 * Given a pair of objects living in collections of map spaces
 * with the same domain, return an object in the flat range product.
 */
static Signature<Kind> mapFlatRangeProduct() {
  return {{"Domain", "Range2"}, {{"Domain", "Range"}, {"Domain", "Range3"}}};
}

/*
 * Given an object living in a collections of map spaces
 * with nested domains, return the result of currying.
 */
static Signature<Kind> mapCurry() {
  return {{"Domain2", {"Range2", "Range"}}, {{{"Domain2", "Range2"}, "Range"}}};
}

/*
 * Given an object living in a collections of map spaces
 * with nested range, return the result of uncurrying.
 */
static Signature<Kind> mapUncurry() {
  return {{{"Domain", "Domain2"}, "Range2"},
          {{"Domain", {"Domain2", "Range2"}}}};
}

/*
 * Given an object living in a collections of map spaces
 * with nested domains, return an object in a collection of spaces
 * where the domain has been replaced by its nested domain.
 */
static Signature<Kind> mapDomainFactorDomain() {
  return {{"Domain2", "Range"}, {{{"Domain2", "Range2"}, "Range"}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an object in the collection of wrapped set spaces.
 */
static Signature<Kind> mapWrap() {
  return {{{"Domain", "Range"}}, {{"Domain", "Range"}}};
}

/*
 * Given a pair of objects living in collections of set spaces,
 * return an object living in the product.
 */
static Signature<Kind> setProduct() {
  return {{{"Domain", "Range"}}, {{"Domain"}, {"Range"}}};
}

/*
 * Given an object living in a collection of wrapped set spaces,
 * return an object in the corresponding collection of map spaces.
 */
static Signature<Kind> setUnwrap() {
  return {{"MapDomain", "MapRange"}, {{{"MapDomain", "MapRange"}}}};
}

/*
 * Given a pair of objects living in collections of map spaces
 * with the same domain, return an object in the result of zipping.
 */
static Signature<Kind> mapZip() {
  return {{{"WrappedDomain1", "WrappedDomain2"},
           {"WrappedRange1", "WrappedRange2"}},
          {{{"WrappedDomain1", "WrappedRange1"},
            {"WrappedDomain2", "WrappedRange2"}}}};
}

/*
 * Given an object living in a collection of map spaces,
 * return an anonymous object defined over the same domain.
 */
static Signature<Kind> mapAnonymous() {
  return {{"Domain", "Anonymous"}, {{"Domain", "Range"}}};
}

/*
 * Update an object living in a collection of map spaces
 * with an anonymous object defined over the same domain.
 */
static Signature<Kind> updateMapAnonymous() {
  return {{"Domain", "Range"}, {{"Domain", "Range"}, {"Domain", "Anonymous"}}};
}

/*
 * Update an object living in a collection of range set spaces
 * with an anonymous object defined over some domain
 * to form a map from the domain to the range.
 */
static Signature<Kind> updateRangeAnonymous() {
  return {{"Domain", "Range"}, {{"Range"}, {"Domain", "Anonymous"}}};
}

/*
 * Potential kind signatures for methods with the given name.
 */
static const std::unordered_map<std::string, std::vector<Signature<Kind>>>
    signatures{
        {"add_param", {updateParams(), updateSet()}},
        {"align_params", {updateSetParams(), updateMapParams()}},
        {"apply", {applySet(), applyRange()}},
        {"apply_domain", {applyDomain()}},
        {"preimage_domain", {preimageDomain()}},
        {"pullback", {preimageDomain()}},
        {"apply_range", {applyRange()}},
        {"coalesce", {updateParams(), updateSet(), updateMap()}},
        {"eq_at", {mapTest()}},
        {"ge_set", {setParamsBinary()}},
        {"get_space", {updateParams(), updateSet(), updateMap()}},
        {"gist", {updateSetBinary(), updateMapBinary()}},
        {"intersect",
         {updateParamsBinary(), updateSetBinary(), updateMapBinary()}},
        {"intersect_domain", {updateMapDomain()}},
        {"intersect_range", {updateMapRange()}},
        {"intersect_params", {updateSetParams(), updateMapParams()}},
        {"lt_set", {mapDomainBinary()}},
        {"le_set", {mapDomainBinary()}},
        {"eq_set", {mapDomainBinary()}},
        {"lt_map", {mapDomainBinaryMap()}},
        {"gt_map", {mapDomainBinaryMap()}},
        {"params", {setParams(), mapParams()}},
        {"from_params", {setFromParams()}},
        {"map_from_set", {mapOnSet()}},
        {"add_named_tuple_id_ui", {setFromParams()}},
        {"add_unnamed_tuple_ui", {setFromParams(), mapFromDomain()}},
        {"product", {setProduct()}},
        {"map_from_domain_and_range", {mapFromDomainAndRange()}},
        {"domain", {mapDomain()}},
        {"domain_map", {mapDomainMap()}},
        {"range", {mapRange()}},
        {"reverse", {mapReverse()}},
        {"subtract", {updateSetBinary(), updateMapBinary()}},
        {"unbind_params_insert_domain", {mapFromRangeAndDomain()}},
        {"sum", {updateMapBinary()}},
        {"unite", {updateSetBinary(), updateMapBinary()}},
        {"union_add", {updateMapBinary()}},
        {"range_product", {mapRangeProduct()}},
        {"flat_range_product", {mapFlatRangeProduct()}},
        {"curry", {mapCurry()}},
        {"uncurry", {mapUncurry()}},
        {"domain_factor_domain", {mapDomainFactorDomain()}},
        {"wrap", {mapWrap()}},
        {"unwrap", {setUnwrap()}},
        {"zip", {mapZip()}},
        {"add", {updateSetBinary(), updateMapBinary()}},
        {"sub", {updateSetBinary(), updateMapBinary()}},
        {"mul", {updateSetBinary()}},
        {"div", {updateSetBinary()}},
        {"mod",
         {updateSetBinary(), updateMapBinary(), updateSet(), updateMap()}},
        {"get_at", {updateSet(), updateMap()}},
        {"get_map_list", {updateMap()}},
        {"get_aff", {mapAnonymous()}},
        {"set_aff", {updateMapAnonymous()}},
        {"get_aff_list", {mapAnonymous()}},
        {"get_union_pw_aff", {mapAnonymous()}},
        {"set_union_pw_aff", {updateMapAnonymous()}},
        {"get_union_pw_aff_list", {mapAnonymous()}},
        {"pos_set", {setParams(), mapDomain()}},
        {"nonneg_set", {setParams(), mapDomain()}},
        {"zero_set", {setParams(), mapDomain()}},
        {"zero_union_set", {setParams(), mapDomain()}},
        {"add_constant", {updateSet(), updateMap()}},
        {"add_constant_si", {updateSet(), updateMap()}},
        {"set_val", {updateSet()}},
        {"floor", {updateMap()}},
        {"neg", {updateSet(), updateMap()}},
        {"drop", {updateMap()}},
        {"scale", {updateMap(), updateMapRange()}},
        {"scale_down", {updateMap(), updateMapRange()}},
        {"set_set_tuple_id", {change_wrapped_set(), changeSet()}},
        {"set_tuple_id", {change_wrapped_set(), changeSet()}},
        {"set_range_tuple_id", {changeSet(), changeRange()}},
        {"get_range_stride_info", {mapAnonymous()}},
        {"get_offset", {updateMap()}},
        {"get_range_simple_fixed_box_hull", {updateMap()}},
        {"set_may_source", {updateMapBinary()}},
        {"set_schedule", {updateMap()}},
        {"compute_flow", {updateMap()}},
        {"get_may_dependence", {mapOnDomain()}},
    };

/*
 * Potential kind signatures for classes and methods with the given names.
 * These override the generic signatures above.
 */
static const std::
    map<std::pair<std::string, std::string>, std::vector<Signature<Kind>>>
        specificSignatures{
            {{"set", "identity"}, {mapOnSet()}},
            {{"union_set", "get_space"}, {setParams()}},
            {{"union_map", "get_space"}, {setParams()}},
            {{"union_pw_aff", "get_space"}, {setParams()}},
            {{"union_pw_multi_aff", "get_space"}, {setParams()}},
            {{"union_set", "universe"}, {updateSet()}},
            {{"union_map", "universe"}, {updateMap()}},
            // should be called "gist_domain"
            {{"multi_union_pw_aff", "gist"}, {updateMapDomain()}},
            {{"multi_union_pw_aff", "get_space"}, {mapRange()}},
            {{"aff_list", "reverse"}, {updateMap()}},
            {{"pw_aff_list", "reverse"}, {updateMap()}},
            {{"union_pw_aff_list", "reverse"}, {updateMap()}},
            {{"map_list", "reverse"}, {updateMap()}},
        };

/*
 * Potential kind signatures for static methods with the given name.
 */
static const std::unordered_map<std::string, std::vector<Signature<Kind>>>
    staticSignatures{
        {"from", {updateMap()}},
        {"identity", {updateMap()}},
        {"param_on_domain_space", {setFromParams()}},
        {"param_on_domain", {mapFromDomain()}},
        {"empty", {updateParams(), updateSet(), updateMap()}},
        {"universe", {updateParams(), updateSet(), updateMap()}},
        {"zero", {updateSet(), updateMap()}},
        {"zero_on_domain", {mapFromDomain()}},
        {"from_domain", {mapFromDomain()}},
    };

/*
 * Potential kind signatures for classes and
 * static methods with the given names.
 * These override the generic static signatures above.
 */
static const std::
    map<std::pair<std::string, std::string>, std::vector<Signature<Kind>>>
        specificStaticSignatures{
            {{"multi_aff", "domain_map"}, {mapDomainMapStatic()}},
            {{"multi_aff", "range_map"}, {mapRangeMapStatic()}},
            {{"multi_aff", "wrapped_range_map"}, {setWrappedRangeMapStatic()}},
            {{"union_set", "empty"}, {setFromParams()}},
            {{"union_map", "empty"}, {mapFromParams()}},
        };

/*
 * A constructor, described by the argument types and
 * one or more corresponding kind signatures.
 */
struct Constructor {
  std::vector<IslType> argTypes;
  std::vector<Signature<Kind>> signatures;
};

/*
 * For each class in the regular C++ interface,
 * a hard-coded list of constructors that need to be exported.
 */
static const std::unordered_map<std::string, std::vector<Constructor>>
    constructors{
        {"multi_id", {{{"space", "id_list"}, {updateSet()}}}},
        {"multi_val", {{{"space", "val_list"}, {updateSet()}}}},
        {"multi_aff", {{{"space", "aff_list"}, {updateMapAnonymous()}}}},
        {"union_pw_aff", {{{"union_set", "val"}, {mapFromDomain()}}}},
        {"multi_union_pw_aff",
         {{{"space", "union_pw_aff_list"}, {updateRangeAnonymous()}},
          {{"union_set", "multi_val"}, {mapFromDomainAndRange()}}}},
        {"map", {{{"multi_aff"}, {updateMap()}}}},
        {"union_map", {{{"map"}, {updateMap()}}}},
        {"union_set", {{{"set"}, {updateSet()}}}},
        {"pw_aff", {{{"aff"}, {updateSet(), updateMap()}}}},
        {"aff_list",
         {{{"aff"}, {updateSet(), updateMap()}},
          {{"ctx", "int"}, {createSet(), createMap()}}}},
        // should be replaced by a constructor without int argument
        {"space", {{{"ctx", "int"}, {createParams()}}}},
        {"union_pw_aff_list", {{{"ctx", "int"}, {createSet(), createMap()}}}},
        {"union_access_info", {{{"union_map"}, {updateMap()}}}},
    };

/*
 * Is "name" the name of a "foreach" method?
 */
static bool isForeach(const std::string& name) {
  return name.find("foreach_") != std::string::npos;
}

/*
 * A collection of substitutions of names by tuples.
 * There are two kinds of names, those that represent the leave tuples
 * in a nesting structure and those that represent the names
 * of a particular nested sequence of tuples.
 * The substitution of the second kind is always
 * (a tuple constructed from) a name.
 */
using Subs = std::map<std::string, Tuple>;

/*
 * Collect all non-anonymous template arguments from the tuple and
 * add them to the given set.
 */
static std::set<std::string> collect(
    const Tuple& tuple,
    std::set<std::string> set = {}) {
  if (tuple.name != "Anonymous") {
    set.insert(tuple.name);
  }
  for (const auto& el : tuple.children) {
    set = collect(el, set);
  }
  return set;
}

/*
 * Collect all non-anonymous template arguments from the kind and
 * add them to the given set.
 */
static std::set<std::string> collect(
    const Kind& kind,
    std::set<std::string> set = {}) {
  for (auto base : kind) {
    set = collect(base, set);
  }
  return set;
}

/*
 * Collect all non-anonymous template arguments from the kind signature and
 * add them to the given set.
 */
static std::set<std::string> collect(const Signature<Kind>& signature) {
  auto set = collect(signature.returnType);
  for (auto arg : signature.argTypes) {
    set = collect(arg, set);
  }
  return set;
}

/*
 * Print "s" to "os", for use in template functions.
 */
static void print(std::ostream& os, const std::string& s) {
  os << s;
}

/*
 * Print a tuple to "os".
 * If the tuple is represented by a name, that name gets printed.
 * Otherwise, it represents a nested pairs of tuples and is
 * printed as a "Pair" or as a "NamedPair" depending
 * on whether it's name is "Anonymous".
 */
static void print(std::ostream& os, const Tuple& tuple) {
  if (tuple.children.size() == 2) {
    if (tuple.name == "Anonymous") {
      os << "Pair<";
    } else {
      os << "NamedPair<";
      print(os, tuple.name);
      os << ",";
    }
    print(os, tuple.children[0]);
    os << ",";
    print(os, tuple.children[1]);
    os << ">";
  } else {
    print(os, tuple.name);
  }
}

/*
 * Print a collection of template arguments, where each argument
 * is preceded by "qualifier" (possibly an empty string).
 */
template <typename C>
static void printTemplateArguments(
    const C collection,
    const std::string& qualifier) {
  std::cout << "<";
  bool first = true;
  for (auto s : collection) {
    if (!first) {
      std::cout << ", ";
    }
    std::cout << qualifier;
    print(std::cout, s);
    first = false;
  }
  std::cout << ">";
}

/*
 * Print a template declaring all the elements in "t"
 * as template arguments.
 */
template <typename T>
static void printTemplate(const T& t) {
  std::cout << "template ";
  printTemplateArguments(t, "typename ");
  std::cout << "\n";
}

/*
 * Print a class declaration for "name" with partial
 * template specialization specified by "kind".
 */
static void printClassDeclaration(const std::string& name, const Kind& kind) {
  printTemplate(collect(kind, {}));
  std::cout << "struct " << name;
  printTemplateArguments(kind, "");
}

/*
 * Print forward declarations for all exported classes
 * with template parameter packs.
 */
static void printForwardDeclarations() {
  for (auto kvp : classes) {
    std::cout << "\n";
    std::cout << "template <typename...>\n";
    std::cout << "struct " << kvp.second.name;
    std::cout << ";\n";
  }
}

/*
 * Specialize "name" based on the given substitutions.
 * If "name" appears among the substitutions, then return the result
 * of the substitution.  Otherwise, return "name" itself (as a Tuple).
 */
static Tuple specialize(const std::string& name, const Subs& subs) {
  if (subs.count(name) != 0) {
    return subs.at(name);
  } else {
    return name.c_str();
  }
}

/*
 * Recursively specialize "tuple" based on the given substitutions.
 */
static Tuple specialize(const Tuple& tuple, const Subs& subs) {
  if (tuple.children.size() == 0) {
    return specialize(tuple.name, subs);
  } else {
    return Tuple{specialize(tuple.name, subs).name.c_str(),
                 specialize(tuple.children[0], subs),
                 specialize(tuple.children[1], subs)};
  }
}

/*
 * Specialize "kind" based on the given substitutions.
 */
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

/*
 * Specialize the given vector of kinds based on the given substitutions.
 */
static std::vector<Kind> specialize(
    const std::vector<Kind>& vector,
    const Subs& subs) {
  std::vector<Kind> specialized;
  for (auto kind : vector) {
    specialized.emplace_back(specialize(kind, subs));
  }
  return specialized;
}

/*
 * Specialize the given signature based on the given substitutions.
 */
static Signature<Kind> specialize(
    const Signature<Kind>& signature,
    const Subs& subs) {
  return {specialize(signature.returnType, subs),
          specialize(signature.argTypes, subs)};
}

/*
 * Print a template for all the template arguments in "signature"
 * that do not already appear in "classKind", if there are any.
 */
static void printExtraTemplate(
    const Kind& classKind,
    const Signature<Kind>& signature) {
  auto classBase = collect(classKind, {});
  classBase.insert("Anonymous");
  auto signatureBase = collect(signature);
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

/*
 * Print the templated type corresponding to the given isl type and
 * the given kind.
 */
static void printTemplatedType(const IslType& type, const Kind& kind) {
  const auto& templateClass = classes.at(type);
  std::cout << templateClass.name;
  printTemplateArguments(kind, "");
}

/*
 * Print the templated return type of "method" with kind information
 * taken from "signature".
 */
static void printTemplatedReturnType(
    const Signature<Kind>& signature,
    const Method& method) {
  printTemplatedType(method.signature.returnType, signature.returnType);
}

/*
 * Print the templated argument types specified by "argTypes" and
 * "signature".  For non-static methods, the first element of
 * the argument types of "signature" corresponds to the object
 * on which the method is called and is therefore skipped.
 * Only elements in "argTypes" that have a corresponding
 * templated class type consume an element in the argument
 * types of "signature".
 */
static void printTemplatedArgumentTypes(
    const std::vector<IslType>& argTypes,
    const Signature<Kind>& signature,
    bool isStatic) {
  std::cout << "(";
  size_t j = isStatic ? 0 : 1;
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "const ";
    const IslType& type = argTypes[i];
    if (classes.count(type) == 0) {
      std::cout << type;
    } else {
      printTemplatedType(type, signature.argTypes[j++]);
    }
    std::cout << "& arg" << i;
  }
  std::cout << ")";
}

/*
 * Return a substitution that extends "subs" and turns "src" into "dst",
 * assuming that "dst" is a special case of "src".
 */
static Subs specializer(
    const std::vector<Tuple>& dst,
    const std::vector<Tuple>& src,
    Subs subs = {}) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i].children.size() == 0) {
      subs.emplace(src[i].name, dst[i]);
    } else if (src[i].children.size() == dst[i].children.size()) {
      subs.emplace(src[i].name, dst[i].name.c_str());
      subs = specializer(dst[i].children, src[i].children, subs);
    }
  }
  return subs;
}

/*
 * Specialize "signature" to "classKind" based on a specialization
 * of the object type to "classKind".
 */
static Signature<Kind> specialize(
    const Signature<Kind>& signature,
    const Kind& classKind) {
  Subs subs = specializer(classKind, signature.argTypes[0]);
  return specialize(signature, subs);
}

/*
 * Print a templated version of the method "method"
 * with the given space kind signature
 * as part of a class definition of a templated class
 * derived from "base" with the given specialized kind.
 *
 * The templated method simply calls the corresponding
 * method in the base class and converts the result
 * to a templated type.
 * The return type is always templated because
 * methods that do not return a type that can/should
 * be templated can simply be called directly in the base class.
 */
static bool printMethod(
    const std::string& base,
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Method& method,
    bool isStatic = false) {
  printExtraTemplate(classKind, signature);
  if (isStatic) {
    std::cout << "static ";
  }
  std::cout << "inline ";
  printTemplatedReturnType(signature, method);
  std::cout << " ";
  std::cout << method.name;
  printTemplatedArgumentTypes(method.signature.argTypes, signature, isStatic);
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
  printTemplatedReturnType(signature, method);
  std::cout << "(res);\n";
  std::cout << "}\n";
  return true;
}

/*
 * Print a method called to*, derived from a "from" method
 * with return value and argument interchanged.
 * The return type is included in the name of the method.
 */
static void printTo(const Signature<Kind>& signature, const Method& method) {
  std::cout << "inline ";
  printTemplatedReturnType(signature, method);
  std::cout << " to" << classes.at(method.signature.returnType).name
            << "() const {\n";
  std::cout << "return ";
  printTemplatedReturnType(signature, method);
  std::cout << "::from(*this);\n";
  std::cout << "}\n";
}

/*
 * Print a method called as*, derived from a conversion constructor
 * with return value and argument interchanged.
 * The return type is included in the name of the method.
 */
static void printAs(const Signature<Kind>& signature, const Method& method) {
  std::cout << "inline ";
  printTemplatedReturnType(signature, method);
  std::cout << " as" << classes.at(method.signature.returnType).name
            << "() const {\n";
  std::cout << "return ";
  printTemplatedReturnType(signature, method);
  std::cout << "(*this);\n";
  std::cout << "}\n";
}

/*
 * Print a "foreach" method corresponding to "method"
 * in the class derived from "base" with the given class kind.
 * The "foreach" method calls "method" with a lambda that
 * converts the regular isl type to the corresponding
 * templated type.
 */
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
  printTemplatedType(argType, classKind);
  std::cout << fn.substr(close);
  std::cout << "& fn) const {\n";
  std::cout << "auto lambda = [fn](" << argType << " arg) -> void {\n";
  std::cout << "fn(";
  printTemplatedType(argType, classKind);
  std::cout << "(arg));";
  std::cout << "};\n";
  std::cout << "this->" << base << "::" << method.name << "(lambda);\n";
  std::cout << "}\n";
}

/*
 * Return the number of elements in "argTypes" that have a corresponding
 * templated type.
 */
static size_t countTemplated(const std::vector<IslType>& argTypes) {
  size_t count = 0;
  for (const auto& type : argTypes) {
    if (classes.count(type) != 0) {
      ++count;
    }
  }
  return count;
}

/*
 * Does "signature" match the static method with the given arguments and
 * class kind?
 * The number of tuple template arguments
 * of the return value needs to match that of the class kind and
 * the number of arguments in the signature needs to be equal to
 * the number of arguments of the method that have a corresponding
 * templated type.
 */
static bool matchesStatic(
    const Kind& classKind,
    const Signature<Kind>& signature,
    const std::vector<IslType>& argTypes) {
  return signature.returnType.size() == classKind.size() &&
      countTemplated(argTypes) == signature.argTypes.size();
}

/*
 * Does "signature" match the given method and class kind?
 * Delegate to matchesStatic for static methods.
 * For other methods, the number of tuple template arguments
 * of the first argument needs to match that of the class kind and
 * the number of arguments in the signature needs to be one more
 * than the number of arguments of the method that have a corresponding
 * templated type, the extra element corresponding to the object
 * on which the method is called.
 */
static bool matches(
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Method& method,
    bool isStatic) {
  if (isStatic) {
    return matchesStatic(classKind, signature, method.signature.argTypes);
  }
  if (signature.argTypes[0].size() != classKind.size()) {
    return false;
  }
  auto count = countTemplated(method.signature.argTypes);
  return signature.argTypes.size() == 1 + count;
}

/*
 * Call "f" on the signature specialized by "subs"
 * if it matches the method and class kind.
 * For non-static methods, the object kind may be a partial specialization.
 * The signature is therefore further specialized to "classKind".
 */
template <typename Functor>
static void callIfMatching(
    const Kind& classKind,
    const Signature<Kind>& signature,
    const Method& method,
    bool isStatic,
    const Subs& subs,
    const Functor& f) {
  auto specializedSignature = specialize(signature, subs);
  if (matches(classKind, specializedSignature, method, isStatic)) {
    if (!isStatic) {
      specializedSignature = specialize(specializedSignature, classKind);
    }
    f(specializedSignature, isStatic);
  }
}

/*
 * Call "f" on each of the matching signatures, specialized by "subs",
 * for the given method, regular isl class ("base") and
 * the given number of template arguments of the matching kind
 * (first argument for regular methods and return value for static methods).
 *
 * Specific signatures override the generic signatures, so they are
 * examined first.
 */
template <typename Functor>
static void foreachStructureMatchingSignature(
    const std::string& base,
    const Kind& classKind,
    const Method& method,
    const Subs& subs,
    const Functor& f) {
  if (specificSignatures.count({base, method.name}) != 0) {
    for (const auto& signature : specificSignatures.at({base, method.name})) {
      callIfMatching(classKind, signature, method, false, subs, f);
    }
  } else if (specificStaticSignatures.count({base, method.name}) != 0) {
    for (const auto& signature :
         specificStaticSignatures.at({base, method.name})) {
      callIfMatching(classKind, signature, method, true, subs, f);
    }
  } else if (signatures.count(method.name) != 0) {
    for (const auto& signature : signatures.at(method.name)) {
      callIfMatching(classKind, signature, method, false, subs, f);
    }
  } else if (staticSignatures.count(method.name) != 0) {
    for (const auto& signature : staticSignatures.at(method.name)) {
      callIfMatching(classKind, signature, method, true, subs, f);
    }
  }
}

/*
 * Call "f" on the first matching signature, specialized by "subs",
 * for the given method, regular isl class ("base") and the given matching kind
 * (first argument for regular methods and return value for static methods).
 */
template <typename Functor>
static void onFirstMatchingSignature(
    const std::string& base,
    const Kind& classKind,
    const Method& method,
    const Subs& subs,
    const Functor& f) {
  bool first = true;
  foreachStructureMatchingSignature(
      base,
      classKind,
      method,
      subs,
      [classKind, f, &first](const Signature<Kind>& signature, bool isStatic) {
        const auto& match =
            isStatic ? signature.returnType : signature.argTypes[0];
        if (first && match == classKind) {
          f(signature, isStatic);
          first = false;
        }
      });
}

/*
 * Print templated versions of the methods in "methods"
 * as part of a class definition of a templated class
 * derived from "base" with the given specialized kind.
 * "subs" specializes the corresponding generic class kind
 * to the specialized class kind.
 * The #to, #as and foreach* methods are handled separately.
 */
static void printMethods(
    const std::string& base,
    const Kind& specializedClassKind,
    const std::vector<Method>& methods,
    const Subs& subs) {
  for (auto method : methods) {
    onFirstMatchingSignature(
        base,
        specializedClassKind,
        method,
        subs,
        [base, specializedClassKind, method, subs](
            const Signature<Kind>& signature, bool isStatic) {
          printMethod(base, specializedClassKind, signature, method, isStatic);
        });
    if (method.name == "#to" &&
        classes.count(method.signature.returnType) == 1) {
      for (auto returnKind : classes.at(method.signature.returnType).kinds) {
        if (specializedClassKind.size() == 2 && returnKind.size() == 2) {
          printTo(specialize(updateMap(), subs), method);
        }
        if (specializedClassKind.size() == 1 && returnKind.size() == 1) {
          printTo(specialize(updateSet(), subs), method);
        }
      }
    } else if (
        method.name == "#as" &&
        classes.count(method.signature.returnType) == 1) {
      for (auto returnKind : classes.at(method.signature.returnType).kinds) {
        if (specializedClassKind.size() == returnKind.size()) {
          for (const auto& constructor :
               constructors.at(method.signature.returnType)) {
            for (const auto& signature : constructor.signatures) {
              if (constructor.argTypes[0] == base &&
                  signature.returnType.size() == specializedClassKind.size()) {
                auto specializedSignature = specialize(signature, subs);
                printAs(specializedSignature, method);
              }
            }
          }
        }
      }
    } else if (isForeach(method.name)) {
      printForeach(base, specializedClassKind, method);
    }
  }
}

/*
 * Print a constructor for the templated class "className"
 * derived from the regular class "base", with the given
 * argument types and kind signature.
 *
 * Each constructor simply calls the constructor of the base class.
 */
static void printConstructor(
    const std::string& base,
    const std::string& className,
    const std::vector<IslType>& argTypes,
    const Signature<Kind>& signature) {
  std::cout << className;
  printTemplatedArgumentTypes(argTypes, signature, true);
  std::cout << " : " << base << "(";
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "arg" << i;
  }
  std::cout << ") {}\n";
}

/*
 * Print a templated version of class "base" called "className"
 * for the given specialized class kind.
 * "subs" specializes the corresponding generic class kind
 * to the specialized class kind.
 */
static void printOneDefinition(
    const std::string& base,
    const std::string& className,
    const Kind& specializedClassKind,
    const Exported& exported,
    const Subs& subs) {
  std::cout << "\n";
  printClassDeclaration(className, specializedClassKind);
  std::cout << " : public " << base << " {\n";
  std::cout << className << "() = default;\n";
  std::cout << "explicit " << className << "(const " << base
            << "& obj) : " << base << "(obj) {}\n";
  if (constructors.count(base) != 0) {
    for (const auto& constructor : constructors.at(base)) {
      for (const auto& signature : constructor.signatures) {
        const auto& argTypes = constructor.argTypes;
        if (matchesStatic(specializedClassKind, signature, argTypes)) {
          auto specializedSignature = specialize(signature, subs);
          printConstructor(
              base, className, constructor.argTypes, specializedSignature);
        }
      }
    }
  }
  if (exported.count(base) != 0) {
    printMethods(base, specializedClassKind, exported.at(base), subs);
  }
  std::cout << "};\n";
}

/*
 * Add the specialized kind of each signature for any exported method
 * of class "base" with number of template arguments specified
 * by "classKind" to "kinds".
 * For a non-static signature, add the object kind.
 * For a static signature, add the return kind.
 */
static void collectSpecializedKinds(
    const std::string& base,
    const Kind& classKind,
    const Exported& exported,
    std::set<Kind>& kinds) {
  for (auto method : exported.at(base)) {
    foreachStructureMatchingSignature(
        base,
        classKind,
        method,
        {},
        [&kinds](const Signature<Kind>& signature, bool isStatic) {
          if (isStatic) {
            kinds.emplace(signature.returnType);
          } else {
            kinds.emplace(signature.argTypes[0]);
          }
        });
  }
}

/*
 * Print templated versions of the exported methods for the class
 * with name "base" in the regular interface and name "className"
 * in the templated interface and for a number of template arguments
 * as specified by "classKind".
 *
 * Besides the generic version described by "classKind", also
 * print definitions for specialized versions where one or more
 * of the template arguments have been specialized to
 * a nested tuple structure.
 * In particular, print a specialized version for each specialized
 * kind that appears in one of the signatures as either the object
 * kind (for non-static signature) or the return type (for static
 * signatures).
 * Note that if an instance occurs where the domain is specialized and
 * one where the range is specialized, then an instance where both
 * are specialized should also be generated.
 * This is not currently guaranteed by the code.
 * Instead, the required kind is assumed to be among
 * those derived from the signatures of the exported methods.
 */
static void printDefinition(
    const std::string& base,
    const std::string& className,
    const Kind& genericClassKind,
    const Exported& exported) {
  if (exported.count(base) == 0) {
    return;
  }

  std::set<Kind> kinds{genericClassKind};
  collectSpecializedKinds(base, genericClassKind, exported, kinds);
  for (auto kind : kinds) {
    Subs subs = specializer(kind, genericClassKind);
    printOneDefinition(base, className, kind, exported, subs);
  }
}

/*
 * Print templated versions of the exported methods for each class and
 * for each number of template arguments.
 *
 * For the most part, the order in which the different specializations
 * appear is immaterial.  However, a specialization with zero template
 * arguments needs to appear before any use, as such a use is otherwise
 * considered to refer to an incomplete type.
 * Specializations with zero template arguments are therefore printed first.
 * Since Space<> is used by Set<>, while Set<> is used by other *<>,
 * these two are printed first among those with zero template arguments.
 */
static void printDefinitions(const Exported& exported) {
  printDefinition("space", "Space", paramsKind(), exported);
  printDefinition("set", "Set", paramsKind(), exported);
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

/*
 * Extract the type from a method argument.
 * Some of these types are const, while others are not.
 */
static std::string extractType(std::string arg) {
  size_t start = 0;
  constexpr auto constStr = "const ";
  if (arg.find(constStr) == 0) {
    start += strlen(constStr);
  }
  return dropIslNamespace(arg.substr(0, arg.find(" ", start)));
}

/*
 * Rudimentary extraction of argument types from whatever appears
 * between the parentheses in a method declaration.
 * In particular, split "args" into pieces separated by ", ".
 * This does not work properly if any of the arguments
 * also contains this sequence, e.g., a function pointer
 * to a function with multiple arguments.
 */
static std::vector<std::string> extractArgumentTypes(const std::string& args) {
  std::vector<std::string> list;
  size_t pos, old = 0;

  while ((pos = args.find(", ", old)) != std::string::npos) {
    list.emplace_back(extractType(args.substr(old, pos)));
    old = pos + 2;
  }
  if (args.length() > 0) {
    list.emplace_back(extractType(args.substr(old)));
  }
  return list;
}

/*
 * Collect all C++ methods exported by the regular C++ interface
 * from standard input for which a kind signature is available.
 * For each method called "from" with a single argument,
 * a special "#to" method is also added to the class
 * corresponding to this single argument.
 * Methods that have a names starting with "foreach" are also added.
 */
static void collectExportedFromInput(Exported& exported) {
  for (std::string line; std::getline(std::cin, line);) {
    // This matches both static and regular methods.
    // The static methods do not have a "const" specifier.
    std::regex declaration("^([a-z_:]+) (.*)::([a-z_]+)\\((.*)\\)(.*const)?$");
    std::smatch match;
    if (!std::regex_match(line, match, declaration)) {
      continue;
    }

    auto retType = dropIslNamespace(match[1].str());
    auto className = dropIslNamespace(match[2].str());
    auto name = match[3].str();
    auto args = extractArgumentTypes(match[4].str());

    if (name == "from" && args.size() == 1) {
      exported[args[0]].emplace_back(Method{"#to", {retType}});
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
}

/*
 * Collect all C++ methods exported by the regular C++ interface,
 * for which a kind signature is available.
 * For each method called "from" with a single argument,
 * a special "#to" method is also added to the class
 * corresponding to this single argument.
 *
 * Additionally, for each hard-coded constructor in "constructors"
 * with a single argument, an "#as" method is added to the class
 * corresponding to this single argument.
 */
static Exported collectExported() {
  Exported exported;

  collectExportedFromInput(exported);

  for (auto kvp : constructors) {
    for (const auto& constructor : kvp.second) {
      const auto& args = constructor.argTypes;
      if (args.size() == 1) {
        exported[args[0]].emplace_back(Method{"#as", {kvp.first, args}});
      }
    }
  }

  return exported;
}

/*
 * Generate a templated isl C++ interface on standard output
 * from the regular C++ interface read from standard input.
 */
int main(int argc, char** argv) {
  auto exported = collectExported();

  std::cout << header;
  printForwardDeclarations();
  printDefinitions(exported);
  std::cout << footer;

  return EXIT_SUCCESS;
}
