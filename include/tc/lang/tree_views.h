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
#include "error_report.h"
#include "tree.h"

namespace lang {

/// TreeView provides a statically-typed way to access the members of a TreeRef
/// instead of using TK_MATCH
struct TreeView {
  explicit TreeView(const TreeRef& tree_) : tree_(tree_) {}
  TreeRef tree() const {
    return tree_;
  }
  const SourceRange& range() const {
    return tree_->range();
  }
  operator TreeRef() const {
    return tree_;
  }

 protected:
  TreeRef tree_;
};

template <typename T>
struct ListViewIterator {
  ListViewIterator(TreeList::const_iterator it) : it(it) {}
  bool operator!=(const ListViewIterator& rhs) const {
    return it != rhs.it;
  }
  T operator*() const {
    return T(*it);
  }
  void operator++() {
    ++it;
  }
  void operator--() {
    --it;
  }

 private:
  TreeList::const_iterator it;
};

template <typename T>
struct ListView : public TreeView {
  ListView(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_LIST);
  }
  typedef ListViewIterator<T> iterator;
  typedef ListViewIterator<T> const_iterator;
  iterator begin() const {
    return iterator(tree_->trees().begin());
  }
  iterator end() const {
    return iterator(tree_->trees().end());
  }
  T operator[](size_t i) const {
    return T(tree_->trees().at(i));
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  size_t size() const {
    return tree_->trees().size();
  }
  static TreeRef create(const SourceRange& range, TreeList elements) {
    return Compound::create(TK_LIST, range, std::move(elements));
  }
};

using List = ListView<TreeRef>;

template <typename T>
struct OptionView : public TreeView {
  explicit OptionView(const TreeRef& tree) : TreeView(tree) {
    TC_ASSERT(tree, tree->kind() == TK_OPTION);
  }
  bool present() const {
    return tree_->trees().size() > 0;
  }
  T get() const {
    TC_ASSERT(tree_, present());
    return T(tree_->trees()[0]);
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
};

struct Ident : public TreeView {
  // each subclass of TreeView provides:
  // 1. a constructor that takes a TreeRef, and matches it to the right type.
  explicit Ident(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_IDENT, name_);
  }
  // 2. accessors that get underlying information out of the object
  // in this case, we return the name of the identifier, and handle the
  // converstion to a string in the method
  const std::string& name() const {
    return name_->stringValue();
  }

  // 3. a static method 'create' that creates the underlying TreeRef object
  // for every TreeRef kind that has a TreeView, the parser always uses
  // (e.g.) Ident::create rather than Compound::Create, this means that
  // changes to the structure of Ident are always made right here rather
  // than both in the parser and in this code
  static TreeRef create(const SourceRange& range, const std::string& name) {
    return Compound::create(TK_IDENT, range, {String::create(name)});
  }

 private:
  TreeRef name_;
};

template <int kind>
struct ApplyLike : public TreeView {
  explicit ApplyLike(const TreeRef& tree) : TreeView(tree) {
    tree_->match(kind, name_, arguments_);
  }

  Ident name() const {
    return Ident(name_);
  }
  ListView<TreeRef> arguments() const {
    return ListView<TreeRef>(arguments_);
  }

  static TreeRef
  create(const SourceRange& range, TreeRef name, TreeRef arguments) {
    return Compound::create(kind, range, {name, arguments});
  }

 private:
  TreeRef name_;
  TreeRef arguments_;
};
using Apply = ApplyLike<TK_APPLY>;
using Access = ApplyLike<TK_ACCESS>;

struct BuiltIn : public TreeView {
  explicit BuiltIn(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_BUILT_IN, name_, arguments_, type_);
  }
  const std::string& name() const {
    return name_->stringValue();
  }
  ListView<TreeRef> arguments() const {
    return ListView<TreeRef>(arguments_);
  }

  TreeRef type() const {
    return type_;
  }

  static TreeRef create(
      const SourceRange& range,
      const std::string& name,
      TreeRef arguments,
      TreeRef type) {
    return Compound::create(
        TK_BUILT_IN, range, {String::create(name), arguments, type});
  }

 private:
  TreeRef name_;
  TreeRef arguments_;
  TreeRef type_; // because Halide needs to know the output type
};

struct TensorType : public TreeView {
  explicit TensorType(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_TENSOR_TYPE, scalar_type_, dims_);
  }
  static TreeRef
  create(const SourceRange& range, TreeRef scalar_type_, TreeRef dims_) {
    return Compound::create(TK_TENSOR_TYPE, range, {scalar_type_, dims_});
  }
  TreeRef scalarTypeTree() const {
    if (scalar_type_->kind() == TK_IDENT)
      throw ErrorReport(tree_)
          << " TensorType has a symbolic ident " << Ident(scalar_type_).name()
          << " rather than a concrete type";
    return scalar_type_;
  }
  int scalarType() const {
    return scalarTypeTree()->kind();
  }
  // either an Ident or a constant
  ListView<TreeRef> dims() const {
    return ListView<TreeRef>(dims_);
  }

 private:
  TreeRef scalar_type_;
  TreeRef dims_;
};

struct Param : public TreeView {
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_PARAM, ident_, type_);
  }
  static TreeRef create(const SourceRange& range, TreeRef ident, TreeRef type) {
    return Compound::create(TK_PARAM, range, {ident, type});
  }
  // when the type of a field is statically know the accessors return
  // the wrapped type. for instance here we know ident_ is an identifier
  // so the accessor returns an Ident
  // this means that clients can do p.ident().name() to get the name of the
  // parameter.
  Ident ident() const {
    return Ident(ident_);
  }
  // may be TensorType or TK_INFERRED
  TreeRef type() const {
    return type_;
  }
  bool typeIsInferred() const {
    return type_->kind() == TK_INFERRED;
  }
  // helper for when you know the type is not inferred.
  TensorType tensorType() const {
    return TensorType(type_);
  }

 private:
  TreeRef ident_;
  TreeRef type_;
};

struct Equivalent : public TreeView {
  explicit Equivalent(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_EQUIVALENT, name_, accesses_);
  }
  static TreeRef
  create(const SourceRange& range, const std::string& name, TreeRef accesses) {
    return Compound::create(
        TK_EQUIVALENT, range, {String::create(name), accesses});
  }
  const std::string& name() const {
    return name_->stringValue();
  }
  ListView<TreeRef> accesses() const {
    return ListView<TreeRef>(accesses_);
  }

 private:
  TreeRef name_;
  TreeRef accesses_;
};

struct RangeConstraint : public TreeView {
  explicit RangeConstraint(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_RANGE_CONSTRAINT, ident_, start_, end_);
  }
  static TreeRef
  create(const SourceRange& range, TreeRef ident, TreeRef start, TreeRef end) {
    return Compound::create(TK_RANGE_CONSTRAINT, range, {ident, start, end});
  }
  Ident ident() const {
    return Ident(ident_);
  }
  TreeRef start() const {
    return start_;
  }
  TreeRef end() const {
    return end_;
  }

 private:
  TreeRef ident_;
  TreeRef start_;
  TreeRef end_;
};

struct Comprehension : public TreeView {
  explicit Comprehension(const TreeRef& tree) : TreeView(tree) {
    tree_->match(
        TK_COMPREHENSION,
        ident_,
        indices_,
        assignment_,
        rhs_,
        range_constraints_,
        equivalent_,
        reduction_variables_);
  }
  static TreeRef create(
      const SourceRange& range,
      TreeRef ident,
      TreeRef indices,
      TreeRef assignment,
      TreeRef rhs,
      TreeRef range_constraints,
      TreeRef equivalent,
      TreeRef reduction_variables) {
    return Compound::create(
        TK_COMPREHENSION,
        range,
        {ident,
         indices,
         assignment,
         rhs,
         range_constraints,
         equivalent,
         reduction_variables});
  }
  // when the type of a field is statically know the accessors return
  // the wrapped type. for instance here we know ident_ is an identifier
  // so the accessor returns an Ident
  // this means that clients can do p.ident().name() to get the name of the
  // parameter.
  Ident ident() const {
    return Ident(ident_);
  }
  ListView<Ident> indices() const {
    return ListView<Ident>(indices_);
  }
  // kind == '=', TK_PLUS_EQ, TK_PLUS_EQ_B, etc.
  TreeRef assignment() const {
    return assignment_;
  }
  TreeRef rhs() const {
    return rhs_;
  }
  // we don't use these yet, so there is now TreeView class for them.
  ListView<RangeConstraint> rangeConstraints() const {
    return ListView<RangeConstraint>(range_constraints_);
  }
  OptionView<Equivalent> equivalent() const {
    return OptionView<Equivalent>(equivalent_);
  }
  ListView<Ident> reductionVariables() const {
    return ListView<Ident>(reduction_variables_);
  }

 private:
  TreeRef ident_;
  TreeRef indices_;
  TreeRef assignment_;
  TreeRef rhs_;
  TreeRef range_constraints_;
  TreeRef equivalent_;
  TreeRef reduction_variables_;
};

struct Def : public TreeView {
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DEF, name_, paramlist, retlist, stmts_list);
  }
  Ident name() {
    return Ident(name_);
  }
  // ListView helps turn TK_LISTs into vectors of TreeViews
  // so that we can, e.g., return lists of parameters
  ListView<Param> params() const {
    return ListView<Param>(paramlist);
  }
  ListView<Param> returns() const {
    return ListView<Param>(retlist);
  }
  ListView<Comprehension> statements() const {
    return ListView<Comprehension>(stmts_list);
  }
  static TreeRef create(
      const SourceRange& range,
      TreeRef name,
      TreeRef paramlist,
      TreeRef retlist,
      TreeRef stmts_list) {
    return Compound::create(
        TK_DEF, range, {name, paramlist, retlist, stmts_list});
  }

 private:
  TreeRef name_;
  TreeRef paramlist;
  TreeRef retlist;
  TreeRef stmts_list;
};

struct Select : public TreeView {
  explicit Select(const TreeRef& tree) : TreeView(tree) {
    tree_->match('.', name_, index_);
  }
  Ident name() const {
    return Ident(name_);
  }
  int index() const {
    return index_->doubleValue();
  }
  static TreeRef create(const SourceRange& range, TreeRef name, TreeRef index) {
    return Compound::create('.', range, {name, index});
  }

 private:
  TreeRef name_;
  TreeRef index_;
};

struct Const : public TreeView {
  explicit Const(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_CONST, value_, type_);
  }
  double value() const {
    return value_->doubleValue();
  }
  TreeRef type() const {
    return type_;
  }
  static TreeRef create(const SourceRange& range, TreeRef value, TreeRef type) {
    return Compound::create(TK_CONST, range, {value, type});
  }

 private:
  TreeRef value_;
  TreeRef type_;
};

struct Cast : public TreeView {
  explicit Cast(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_CAST, value_, type_);
  }
  TreeRef value() const {
    return value_;
  }
  TreeRef type() const {
    return type_;
  }
  static TreeRef create(const SourceRange& range, TreeRef value, TreeRef type) {
    return Compound::create(TK_CAST, range, {value, type});
  }

 private:
  TreeRef value_;
  TreeRef type_;
};

} // namespace lang
