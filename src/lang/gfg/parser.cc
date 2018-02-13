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
#include <tc/lang/parser.h>

// #include <glog/logging.h>

#include <sstream>
#include <unordered_set>

using namespace std;

namespace tc {
namespace parser {

#define CHECK(x) ;

int uid() {
  static int id = 0;
  return ++id;
}

Node& GFG::addNode(const string& name) {
  nodes.emplace_back(new Node(name));
  CHECK(nodes.size() == nodes.back().id);
  name2NodeId.insert(
      make_pair<string, int>(string(name), int(nodes.back()->id)));
  return *nodes.back();
}

const Edge& GFG::addEdge(Node& s, const string& transition, const Node& t) {
  CHECK(name2Node.at(s.name) != name2Node.end());
  CHECK(name2Node.at(t.name) != name2Node.end());
  s.outEdges.emplace_back(s, transition, t);
  return s.outEdges.back();
}

ostream& operator<<(ostream& os, const Node& n) {
  os << "Node (" << n.id << "): " << n.name;
  return os;
}

ostream& operator<<(ostream& os, const Edge& e) {
  os << "Edge: " << e.source << " -> " << e.transition << " -> " << e.target;
  return os;
}

ostream& operator<<(ostream& os, const GFG& g) {
  os << "GFG:" << endl;
  for (const auto& n : g.nodes) {
    os << "  " << *n << endl;
    for (const auto& e : n->outEdges) {
      os << "    " << e << endl;
    }
  }
  os << "Name2Node:" << endl;
  for (const auto& kvp : g.name2NodeId) {
    os << "  " << kvp.first << " -> " << kvp.second << endl;
  }
  return os;
}

string productionDef = "::=";
string productionChoice = "|";
string epsilon = "epsilon";

unordered_set<string> terminals;
unordered_map<string, vector<vector<string>>> productions;

void expectNewProduction(const std::string& name) {
  if (productions.find(name) != productions.end()) {
    std::string err("Parsing Error: ");
    err = err + "Duplicate production for: ";
    err = err + name;
    throw runtime_error(err.c_str());
  }
}

void expectValid(istringstream& in) {
  if (in.eof()) {
    throw runtime_error("Parsing error: EOF");
  }
  if (!in.good()) {
    throw runtime_error("Parsing error: Not good");
  }
}

void expectEqual(const std::string& expected, const std::string& actual) {
  if (expected != actual) {
    std::string err("Parsing Error:");
    err = err + " EXPECTEDEQ " + expected;
    err = err + " VS actual " + actual;
    throw runtime_error(err.c_str());
  }
}

void expectNEqual(const std::string& expected, const std::string& actual) {
  if (expected == actual) {
    std::string err("Parsing Error:");
    err = err + " EXPECTEDNEQ " + expected;
    err = err + " VS actual " + actual;
    throw runtime_error(err.c_str());
  }
}

struct SG {
  static int& depth() {
    static int val = 0;
    return val;
  }
  static string open() {
    return string(2 * ++depth(), ' ');
  }
  static string close() {
    return string(2 * depth()--, ' ');
  }
  SG(string s, ostream& stream = cout) : name(s), os(stream) {
    os << SG::open() << "Start " << name << endl;
  }
  ~SG() {
    os << SG::close() << "End " << name << endl;
  }
  string name;
  ostream& os;
};

string readToken(istringstream& in) {
  SG sg("readToken");
  string tok;
  in >> tok;
  cout << SG::open() << "tok:***" << tok << "***" << SG::close() << endl;
  return tok;
}

bool parseProductionChoice(istringstream& in, vector<string>& choices) {
  SG sg("parseProductionChoice");
  do {
    string tok = readToken(in);
    choices.push_back(tok);
  } while (!in.eof() && in.good());
  return true;
}

string parseProduction(istringstream& in) {
  SG sg("parseProduction");
  string production;
  std::getline(in, production, '\\');
  if (in.eof()) {
    return "";
  }

  istringstream in2(production);
  // name
  std::string name = readToken(in2);
  expectNewProduction(name);
  // ::=
  std::string def = readToken(in2);
  expectEqual(productionDef, def);
  expectValid(in2);

  string productionChoice;
  vector<vector<string>> choices;
  while (std::getline(in2, productionChoice, '|')) {
    istringstream in3(productionChoice);
    vector<string> choice;
    parseProductionChoice(in3, choice);
    choices.push_back(choice);
  }
  productions[name] = choices;
  return name;
}

GFG GFG::makeGFG(const string& grammar) {
  bool disable_printing = true;
  if (disable_printing) {
    cout.setstate(std::ios_base::badbit);
  }

  SG sg("parseGFG");
  GFG res;
  istringstream in(grammar);
  while (!in.eof()) {
    auto n = parseProduction(in);
  }

  cout.clear();

  for (auto& kvp : productions) {
    LOG(INFO) << "Prod: " << kvp.first << endl;
    for (auto& v : kvp.second) {
      LOG(INFO) << "  Choices: " << endl;
      for (auto& vv : v) {
        LOG(INFO) << "    " << vv;
      }
      LOG(INFO) << endl;
    }
  }

  // TODO: Build GFG and Pingali's parser
  return res;
}

#undef CHECK
}
} // ns tc::parser
