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
#include "tc/lang/error_report.h"
#include "tc/lang/tree.h"

namespace lang {

/// TreeView provides a statically-typed way to access the members of a TreeRef
/// instead of using TK_MATCH
//
// A few notes on types and their aliases:
// - List<T> is really a Tree with kind TK_LIST and elements as subtrees
// - Maybe<T> is really a Tree with kind TK_OPTION that has 0 or 1 subtree of type T
// - Builtin types are: Ident (TK_IDENT), String (TK_STRING)
//
// -- NB: dim_list can only contain Const and Ident trees
// -- NB: dim_list is optional (can be empty)
// Type  = TensorType(ScalarType scalar_type, List<Expr> dim_list)      TK_TENSOR_TYPE
// Param = Param(Ident name, Type type)                                 TK_PARAM
//
// Def   = Def(Ident name, List<Param> params, List<Param> returns, List<Stmt> body) TK_DEF
//
// -- NB: reduction_variables are only filled during semantic analysis
// Stmt  = Comprehension(Ident lhs_ident, List<Ident> lhs_indices,      TK_COMPREHENSION
//                       AssignKind assignment, Expr rhs,
//                       List<WhereClause> range_constraints,
//                       Option<Equivalent> eqiuvalent_stmt,
//                       List<Ident> reduction_variables)
//
// WhereClause = Let(Ident name, Expr expr)                             TK_LET
//             | RangeConstraint(Ident name, Expr l, Expr r)            TK_RANGE_CONSTRAINT
//             | Exists(Expr expr)                                      TK_EXISTS
//
// Equivalent = Equivalent(String name, List<Expr> accesses)            TK_EQUIVALENT
//
// Expr  = TernaryIf(Expr cond, Expr true_expr, Expr false_expr)        TK_IF_EXPR
//       | BinOp(Expr lhs, Expr rhs)
//       |     And                                                      TK_AND
//       |     Or                                                       TK_OR
//       |     Lt                                                       '<'
//       |     Gt                                                       '>'
//       |     Eq                                                       TK_EQ
//       |     Le                                                       TK_LE
//       |     Ge                                                       TK_GE
//       |     Ne                                                       TK_NE
//       |     Add                                                      '+'
//       |     Sub                                                      '-'
//       |     Mul                                                      '*'
//       |     Div                                                      '/'
//       | UnaryOp(Expr expr)
//       |     Not                                                      '!'
//       |     USub                                                     '-'
//       | Const(Number value, ScalarType type)                         TK_CONST
//       | Cast(Expr expr, ScalarType type)                             TK_CAST
//       | Select(Expr base, Number dim)                                '.'
//       -- XXX: Apply is emitted by the parser, and gets desugared into
//       -- Access and BuiltIn as part of the Sema pass.
//       | Apply(Ident name, List<Expr> args)                           TK_APPLY
//       | Access(Ident name, List<Expr> args)                          TK_ACCESS
//       | BuiltIn(Ident name, List<Expr> args, Type type)              TK_BUILT_IN
//       -- XXX: yes, Ident is a valid Expr too
//       | Ident name                                                   TK_IDENT
//
// ScalarType = Int8()                                                  TK_INT8
//            | Int16()                                                 TK_INT16
//            | Int32()                                                 TK_INT32
//            | Int64()                                                 TK_INT64
//            | UInt8()                                                 TK_UINT8
//            | UInt16()                                                TK_UINT16
//            | UInt32()                                                TK_UINT32
//            | UInt64()                                                TK_UINT64
//            | Bool()                                                  TK_BOOL
//            | Float()                                                 TK_FLOAT
//            | Double()                                                TK_DOUBLE
//
// AssignKind = PlusEq()                                                TK_PLUS_EQ
//            | TimesEq()                                               TK_TIMES_EQ
//            | MinEq()                                                 TK_MIN_EQ
//            | MaxEq()                                                 TK_MAX_EQ
//            | PlusEqB()                                               TK_PLUS_EQ_B
//            | TimesEqB()                                              TK_TIMES_EQ_B
//            | MinEqB()                                                TK_MIN_EQ_B
//            | MaxEqB()                                                TK_MAX_EQ_B

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
  TreeRef subtree(size_t i) const {
    return tree_->tree(i);
  }
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
    tree->expect(TK_LIST);
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
    tree->expect(TK_OPTION);
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
    tree_->expect(TK_IDENT, 1);
  }
  // 2. accessors that get underlying information out of the object
  // in this case, we return the name of the identifier, and handle the
  // converstion to a string in the method
  const std::string& name() const {
    return subtree(0)->stringValue();
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
    tree_->expect(kind, 2);
  }

  Ident name() const {
    return Ident(subtree(0));
  }
  ListView<TreeRef> arguments() const {
    return ListView<TreeRef>(subtree(1));
  }

  static TreeRef
  create(const SourceRange& range, TreeRef name, TreeRef arguments) {
    return Compound::create(kind, range, {name, arguments});
  }
};
using Apply = ApplyLike<TK_APPLY>;
using Access = ApplyLike<TK_ACCESS>;

struct BuiltIn : public TreeView {
  explicit BuiltIn(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_BUILT_IN, 3);
  }
  const std::string& name() const {
    return subtree(0)->stringValue();
  }
  ListView<TreeRef> arguments() const {
    return ListView<TreeRef>(subtree(1));
  }

  TreeRef type() const {
    return subtree(2);
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
    tree_->expect(TK_TENSOR_TYPE, 2);
  }
  static TreeRef
  create(const SourceRange& range, TreeRef scalar_type_, TreeRef dims_) {
    return Compound::create(TK_TENSOR_TYPE, range, {scalar_type_, dims_});
  }
  TreeRef scalarTypeTree() const {
    auto scalar_type_ = subtree(0);
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
    return ListView<TreeRef>(subtree(1));
  }
};

struct Param : public TreeView {
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_PARAM, 2);
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
    return Ident(subtree(0));
  }
  // may be TensorType or TK_INFERRED
  TreeRef type() const {
    return subtree(1);
  }
  bool typeIsInferred() const {
    return type()->kind() == TK_INFERRED;
  }
  // helper for when you know the type is not inferred.
  TensorType tensorType() const {
    return TensorType(type());
  }
};

struct Equivalent : public TreeView {
  explicit Equivalent(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_EQUIVALENT, 2);
  }
  static TreeRef
  create(const SourceRange& range, const std::string& name, TreeRef accesses) {
    return Compound::create(
        TK_EQUIVALENT, range, {String::create(name), accesses});
  }
  const std::string& name() const {
    return subtree(0)->stringValue();
  }
  ListView<TreeRef> accesses() const {
    return ListView<TreeRef>(subtree(1));
  }
};

struct RangeConstraint : public TreeView {
  explicit RangeConstraint(const TreeRef& tree) : TreeView(tree) {
    tree->expect(TK_RANGE_CONSTRAINT, 3);
  }
  static TreeRef
  create(const SourceRange& range, TreeRef ident, TreeRef start, TreeRef end) {
    return Compound::create(TK_RANGE_CONSTRAINT, range, {ident, start, end});
  }
  Ident ident() const {
    return Ident(subtree(0));
  }
  TreeRef start() const {
    return subtree(1);
  }
  TreeRef end() const {
    return subtree(2);
  }
};

struct Comprehension : public TreeView {
  explicit Comprehension(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_COMPREHENSION, 7);
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
    return Ident(subtree(0));
  }
  ListView<Ident> indices() const {
    return ListView<Ident>(subtree(1));
  }
  // kind == '=', TK_PLUS_EQ, TK_PLUS_EQ_B, etc.
  TreeRef assignment() const {
    return subtree(2);
  }
  TreeRef rhs() const {
    return subtree(3);
  }

  // where clauses are either RangeConstraints or Let bindings
  ListView<TreeRef> whereClauses() const {
    return ListView<TreeRef>(subtree(4));
  }
  OptionView<Equivalent> equivalent() const {
    return OptionView<Equivalent>(subtree(5));
  }
  ListView<Ident> reductionVariables() const {
    return ListView<Ident>(subtree(6));
  }
};

struct Def : public TreeView {
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->expect(TK_DEF, 4);
  }
  Ident name() {
    return Ident(subtree(0));
  }
  // ListView helps turn TK_LISTs into vectors of TreeViews
  // so that we can, e.g., return lists of parameters
  ListView<Param> params() const {
    return ListView<Param>(subtree(1));
  }
  ListView<Param> returns() const {
    return ListView<Param>(subtree(2));
  }
  ListView<Comprehension> statements() const {
    return ListView<Comprehension>(subtree(3));
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
};

struct Select : public TreeView {
  explicit Select(const TreeRef& tree) : TreeView(tree) {
    tree_->expect('.', 2);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  int index() const {
    return subtree(1)->doubleValue();
  }
  static TreeRef create(const SourceRange& range, TreeRef name, TreeRef index) {
    return Compound::create('.', range, {name, index});
  }
};

struct Const : public TreeView {
  explicit Const(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_CONST, 2);
  }
  double value() const {
    return subtree(0)->doubleValue();
  }
  TreeRef type() const {
    return subtree(1);
  }
  static TreeRef create(const SourceRange& range, TreeRef value, TreeRef type) {
    return Compound::create(TK_CONST, range, {value, type});
  }
};

struct Cast : public TreeView {
  explicit Cast(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_CAST, 2);
  }
  TreeRef value() const {
    return subtree(0);
  }
  TreeRef type() const {
    return subtree(1);
  }
  static TreeRef create(const SourceRange& range, TreeRef value, TreeRef type) {
    return Compound::create(TK_CAST, range, {value, type});
  }
};

struct Let : public TreeView {
  explicit Let(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_LET, 2);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  TreeRef rhs() const {
    return subtree(1);
  }
  static TreeRef create(const SourceRange& range, TreeRef name, TreeRef rhs) {
    return Compound::create(TK_LET, range, {name, rhs});
  }
};

struct Exists : public TreeView {
  explicit Exists(const TreeRef& tree) : TreeView(tree) {
    tree_->expect(TK_EXISTS, 1);
  }
  TreeRef exp() const {
    return subtree(0);
  }
  static TreeRef create(const SourceRange& range, TreeRef exp) {
    return Compound::create(TK_EXISTS, range, {exp});
  }
};

} // namespace lang
