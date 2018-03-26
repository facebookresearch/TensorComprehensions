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
#include <assert.h>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <iterator>

#include <sstream>
#include <string>

#include "tc/lang/canonicalize.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

using namespace lang;

#ifdef ROOT_PATH
  const std::string expected_file_path = std::string(ROOT_PATH) + "src/lang/test_expected/";
#else
  const std::string expected_file_path = "src/lang/test_expected/";
#endif

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

void assertParseEqual(
    const std::string& test_name,
    const std::string& text,
    std::function<TreeRef(Parser&)> fn) {
  Parser parser(text);
  auto r = fn(parser);
  std::stringstream ss;
  ss << r;
  std::string s = ss.str();
  assertEqual(test_name, s);
}
void assertSemaEqual(const std::string& test_name, const std::string& text) {
  Parser parser(text);
  auto p = parser.parseFunction();
  Sema sem;
  std::stringstream ss;
  auto r = sem.checkFunction(p);
  ss << r << sem.dumpEnv();
  std::string s = ss.str();
  assertEqual(test_name, s);
}
void assertSemaThrows(const std::string& errcontents, const std::string& text) {
  Parser parser(text);
  auto p = parser.parseFunction();
  Sema sem;
  std::stringstream ss;
  bool threw = false;
  try {
    auto r = sem.checkFunction(p);
  } catch (const ErrorReport& e) {
    std::string report = e.what();
    ASSERT(report.find(errcontents) != std::string::npos);
    threw = true;
  }
  ASSERT(threw);
}
TreeRef loadText(const std::string& text) {
  return Sema().checkFunction(Parser(text).parseFunction());
}

std::string canonicalText(const std::string& text) {
  std::stringstream ss;
  ss << canonicalize(loadText(text));
  return ss.str();
}

int main(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    args.push_back(argv[i]);
  }
  for (auto& a : args) {
    if (a == "--accept") {
      acceptChanges = true;
    }
  }

  {
    std::string s = "min min+max 1.4 .3 3 3.\n3e-3 .5e-7 3E-5 foobar";
    std::vector<int> expected({TK_MIN,
                               TK_MIN,
                               '+',
                               TK_MAX,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_NUMBER,
                               TK_IDENT});
    Lexer lex(s);
    std::stringstream out;
    for (int i = 0; lex.cur().kind != TK_EOF; i++) {
      out << "FOUND '" << kindToString(lex.cur().kind) << "'\n";
      lex.cur().range.highlight(out);
      out << "---------\n";
      ASSERT(lex.cur().kind == expected[i]);
      lex.next();
    }
    assertEqual("lexer.expected", out.str());
  }
  {
    std::string somethings = "min\n4.63";
    Lexer lex(somethings);
    auto s = Compound::create(
        TK_CONST, lex.cur().range, {String::create(lex.cur().text())});
    lex.next();
    auto n = Compound::create(
        TK_CONST, lex.cur().range, {Number::create(lex.cur().doubleValue())});
    TreeList foo = {s, n};
    auto bar = Compound::create('-', s->range(), std::move(foo));
    ASSERT(foo.size() == 0);
    ASSERT(bar->range().start() == 0 && bar->range().end() == n->range().end());
    ASSERT(n->trees()[0]->doubleValue() == 4.63);
    ASSERT(s->trees()[0]->stringValue() == "min");
    std::stringstream ss;
    bar->range().highlight(ss);
    assertEqual("lexer2.expected", ss.str());
    s->expect(TK_CONST, 1);
    ASSERT(s->tree(0)->stringValue() == "min");
  }
  {
    std::string stuff = "-3+4*5+7-a";
    Parser p(stuff);
    auto r = p.parseExp();
    std::stringstream ss;
    ss << r;
    assertEqual("math.expected", ss.str());
    std::string stuff2 = "foo(3,4,5+6,a)";
    Parser p2(stuff2);
    std::stringstream ss2;
    ss2 << p2.parseExp();
    assertEqual("function.expected", ss2.str());
  }
  assertParseEqual("trinary.expected", "a ? 3 : b ? 3 : 4", [&](Parser& p) {
    return p.parseExp();
  });
  assertParseEqual(
      "comprehension.expected", "R(i,j) +=! A(i,j) * B(j,k)", [&](Parser& p) {
        return p.parseStmt();
      });
  assertParseEqual(
      "statement.expected",
      "output(b, op, h, w) +=! input(b, ip, h + kh, w + kw) * weight(op, ip, kh, kw)",
      [&](Parser& p) { return p.parseStmt(); });

  auto conv = R"(
    def conv(float(B,IP,H,W) input, float(OP,IP,KH,KW) weight) -> (output) {
      output(b, op, h, w) +=! input(b, ip, h + kh, w + kw) * weight(op, ip, kh, kw)
    }
  )";

  assertParseEqual("convolution.expected", conv, [&](Parser& p) {
    return p.parseFunction();
  });

  auto scalar = R"(
    def fun(float(M) I) -> (O) {
      O +=! I(i)
    }
  )";

  assertParseEqual(
      "scalar.expected", scalar, [&](Parser& p) { return p.parseFunction(); });

  auto r = Parser("a + b + c").parseExp();
  ASSERT(r->kind() == '+');
  auto ab = r->tree(0);
  auto c = r->tree(1);
  ASSERT(ab->kind() == '+');
  auto a = ab->tree(0);
  auto b = ab->tree(1);
  ASSERT(a->range().text() == "a" && b->range().text() == "b");
  ASSERT(c->range().text() == "c");
  ASSERT(c->kind() == TK_IDENT);

  {
    Parser p(conv);
    auto t = p.parseFunction();
    Sema s;
    TreeRef r = s.checkFunction(t);
    std::stringstream ss;
    ss << r;
    assertEqual("sema.expected", ss.str());
    assertEqual("sema-env.expected", s.dumpEnv());
  }

  assertSemaEqual(
      "builtins.expected",
      R"(
    def fun(float(M) I) -> (O) {
      O +=! log(tanh(I(i)))
    }
  )");
  assertSemaEqual(
      "indirect.expected",
      R"(
    def fun(float(M) A, float(N) B) -> (O) {
      O +=! A(int32(B(i)))
    }
  )");
  assertSemaEqual(
      "maxeq.expected",
      R"(
    def fun(float(M) A, float(N) B) -> (O) {
      O max=! A(int32(B(i)))
    }
  )");
  assertSemaEqual(
      "annotate.expected",
      R"(
    def fun(float(X,Y) A, float(Y,Z) B) -> (O) {
      O(i,j) +=! A(i,k) * B(k,j) <=> matmul(A(i,k), B(k,j))
    }
  )");
  assertSemaThrows(
      "but does not specify a reduction",
      R"(
    def fun(float(M,N) A) -> (O) {
      O(i) = A(i,j)
    }
  )");

  auto option_one = R"(
    def fun(float(B, N, M) X, float(B, M, K) Y) -> (Z)
    {
       Z(b, i, j) += X(b, i, k) * Y(b, k, j)
    }
  )";

  auto option_two = R"(
    def fun2(float(B, N, M) X, float(B, M, K) Y) -> (Q) {
            Q(b, ii, j) += X(b, ii, k) * Y(b, k, j)
    }
  )";
  ASSERT(canonicalText(option_one) == canonicalText(option_two));

  // assertSemaEqual(
  //     "comments.expected",
  //     R"(#beginning comment
  //   def fun(float(M) A, float(N) B) -> (O) {
  //     # two lines of comments
  //     # in a row, also a keyword def ->
  //     O max= A(B(i))
  //   }
  // #terminal comment)");
}
