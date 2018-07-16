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
#include "tc/lang/tc_format.h"
#include "tc/lang/tree_views.h"

namespace lang {

namespace {

void showExpr(std::ostream& s, const TreeRef& expr);
void showStmt(std::ostream& s, const TreeRef& stmt);

template <typename T>
void show(std::ostream& s, T x) {
  s << x;
}

template <typename T, typename F>
void showList(std::ostream& s, const ListView<T>& list, F elem_cb) {
  bool first = true;
  for (const auto& elem : list) {
    if (!first) {
      s << ", ";
    }
    elem_cb(s, elem);
    first = false;
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const ListView<T>& list) {
  showList(s, list, show<T>);
  return s;
}

std::ostream& operator<<(std::ostream& s, const Ident& id) {
  return s << id.name();
}

std::ostream& operator<<(std::ostream& s, const Param& p) {
  if (!p.typeIsInferred()) {
    TensorType type{p.type()};
    s << kindToString(type.scalarType()) << "(";
    showList(s, type.dims(), showExpr);
    s << ") ";
  }
  return s << p.ident();
}

std::ostream& operator<<(std::ostream& s, const For& f) {
  s << "for " << f.index() << " in " << f.range().start() << ":"
    << f.range().end() << " {";
  for (const TreeRef& stmt : f.statements()) {
    showStmt(s, stmt);
  }
  s << "}";
  return s;
}

std::ostream& operator<<(std::ostream& s, const Comprehension& comp) {
  s << comp.ident() << "(" << comp.indices() << ") "
    << kindToToken(comp.assignment()->kind()) << " ";
  showExpr(s, comp.rhs());
  if (!comp.whereClauses().empty())
    throw std::runtime_error("Printing of where clauses is not supported yet");
  if (comp.equivalent().present())
    throw std::runtime_error(
        "Printing of equivalent comprehensions is not supported yet");
  return s;
}

void showStmt(std::ostream& s, const TreeRef& stmt) {
  switch (stmt->kind()) {
    case TK_FOR:
      s << "  " << For(stmt) << "\n";
      break;
    case TK_COMPREHENSION:
      s << "  " << Comprehension(stmt) << "\n";
      break;
    default:
      std::stringstream ss;
      ss << "Incorrect statement kind: " << stmt->kind();
      throw std::runtime_error(ss.str());
  }
}

void showExpr(std::ostream& s, const TreeRef& expr) {
  switch (expr->kind()) {
    case TK_IDENT: {
      s << Ident(expr);
      break;
    }
    case TK_AND:
    case TK_OR:
    case '<':
    case '>':
    case TK_EQ:
    case TK_LE:
    case TK_GE:
    case TK_NE:
    case '+':
    case '*':
    case '/': {
      s << "(";
      showExpr(s, expr->tree(0));
      s << " " << kindToToken(expr->kind()) << " ";
      showExpr(s, expr->tree(1));
      s << ")";
      break;
      // '-' is annoying because it can be both unary and binary
    }
    case '-': {
      if (expr->trees().size() == 1) {
        s << "-";
        showExpr(s, expr->tree(0));
      } else {
        s << "(";
        showExpr(s, expr->tree(0));
        s << " - ";
        showExpr(s, expr->tree(1));
        s << ")";
      }
      break;
    }
    case '!': {
      s << "!";
      showExpr(s, expr->tree(0));
      break;
    }
    case TK_CONST: {
      Const con{expr};
      int scalarType = con.type()->kind();
      switch (scalarType) {
        case TK_FLOAT:
        case TK_DOUBLE:
          s << con.value();
          break;
        case TK_UINT8:
        case TK_UINT16:
        case TK_UINT32:
        case TK_UINT64:
          s << static_cast<uint64_t>(con.value());
          break;
        case TK_INT8:
        case TK_INT16:
        case TK_INT32:
        case TK_INT64:
          s << static_cast<int64_t>(con.value());
          break;
        default:
          throw std::runtime_error(
              "Unknown scalar type in const: " +
              kindToString(con.type()->kind()));
      }
      break;
    }
    case TK_CAST: {
      Cast cast{expr};
      s << kindToToken(cast.type()->kind()) << "(";
      showExpr(s, cast.value());
      s << ")";
      break;
    }
    case '.': {
      Select sel{expr};
      s << sel.name() << "." << sel.index();
      break;
    }
    case TK_APPLY:
    case TK_ACCESS:
    case TK_BUILT_IN: {
      s << Ident(expr->tree(0)) << "(";
      showList(s, ListView<TreeRef>(expr->tree(1)), showExpr);
      s << ")";
      break;
    }
    default: {
      throw std::runtime_error(
          "Unexpected kind in showExpr: " + kindToString(expr->kind()));
    }
  }
}

} // anonymous namespace

void tcFormat(std::ostream& s, TreeRef _def) {
  Def def{_def};
  s << "def " << def.name() << "(" << def.params() << ")"
    << " -> (" << def.returns() << ") {\n";
  for (const TreeRef& stmt : def.statements()) {
    showStmt(s, stmt);
  }
  s << "}";
}

} // namespace lang
