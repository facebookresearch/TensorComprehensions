# Existing projects
TC uses multiple third-party libraries.
Modifications and extensions should follow the style of each library.

# TC
We derive the following set of rules from [HHVM](https://github.com/facebook/hhvm/edit/master/hphp/doc/coding-conventions.md).

TC Coding Conventions
=======================

This document is meant to serve as a guide to writing C++ in the TC codebase,
covering when and how to use various language features as well as how code
should be formatted. Our goal is to ensure a consistently high-quality codebase
that is easy to read and contribute to, especially for newcomers.

There's no well-defined cutoff here - just try to minimize effort for your
reviewers. A good rule of thumb is that if your cosmetic changes require adding
significant new sections to the diff (such as a function rename that touches
all call sites), it should probably be pulled out into its own diff.

## Headers ##

Every .cc file in the TC repository should have a corresponding .h file with
the same name, and which declares its public interfaces. We tend to value API
documentation more heavily than inline implementation comments, so *all*
declarations in headers (classes, enums, functions, constants, etc.)  should be
documented. See Comments and Documentation for more details.

Build times are a frequent source of pain in many large C++ projects. Try not
to make large header files that mostly serve to include groups of other large
header files. This can discourage "include what you use," discussed in the
"What to include section".

### Include guards ###

To prevent multiple inclusion, all headers should have the following directive
after their license header comment:

```cpp
/*
 * ... TBD: see the 'File copyright' section for details on what goes here...
 */

#pragma once

// File contents
```

### What to include ###

The golden rule for what to include is "include what you use" (IWYU). In brief,
this means you should not rely on any headers you include to transitively
include other headers which have definitions you require. You should also
prefer to forward declare structs and classes when the definition is not needed
(so, "don't include what you don't use"), which helps reduce TC's nontrivial
build time.

To make it easier to achieve IWYU, we have the following guidelines for
includes:

- Always include the corresponding .h for a .cpp first, before even system
  headers.
- Separate includes into groups: C++ standard library headers, external projects, and finally headers within TC.
  Each group should be separated by a newline, for readability. (Whether to separate TC
  includes by subsystem (e.g., `jit`) is left up to the author.)
- Keep headers alphabetized within each group. This makes it easier to ensure
  that all necessary includes are made, and no extraneous ones are left behind.
- Use double quotes for TC headers and angle brackets for all
  others.

As an example, here is what the include section might look like for a file
named `bytecode.cpp` (example borrowed from the HHVM guidelines):

```cpp
#include "hphp/runtime/vm/bytecode.h"

#include <cstdio>
#include <string>

#include <boost/program_options/options_description.hpp>

#include "hphp/runtime/vm/class.h"
#include "hphp/runtime/vm/func.h"
#include "hphp/runtime/vm/hhbc.h"
#include "hphp/util/string.h"
```

### Inline functions ###

Defining functions inline is encouraged for very short functions.

When defining inline member functions on structs or classes which have tight,
compact interfaces (e.g., a smart pointer class, or any wrapper class), prefer
to define the functions in the class definition, for concision.

However, for classes with more complex, malleable APIs where inline helpers
proliferate, restrict the class
definition to member function prototypes *only*. This makes the API much
cleaner. For these classes, define all inline functions in a corresponding
`-inl.h` file.

```cpp
// At the bottom of func.h.

#include "hphp/runtime/vm/func-inl.h"
```

```cpp
// After the copyright in func-inl.h.

namespace HPHP {

// Definitions go here.

}
```

For API's large enough to warrant -inl.h files, move *all* definitions into the
-inl.h, even one-line accessors. This serves both to keep the API cleaner and
to avoid splitting implementations among three files (the header, the inline,
and the source).

Some files, with or without a corresponding -inl.h file, may need a -defs.h
file. This file also contains definitions of inline functions, but it is *not*
included by the main header. It is intended to be used when only a few callers
need access to the definitions, or when the definitions can't be in the main
header because it would create circular dependencies. It should be included
directly by the callers that do need access to the definitions it contains.


## Structs and Classes ##

Classes are used extensively throughout the TC codebase, with a number of
coding conventions. See also Naming for conventions around class naming.

### Using struct vs. class ###

In C++, `struct` and `class` have nearly identical meanings; the only
difference lies in the default visibility (`struct` defaults to public, and
`class` defaults to private).  Do not rely on the default access specifier, always
explicitly specify visibility of the members independently of `class` or `struct`
being used.

```cpp
struct Foo {
 public:    // even though struct has public visibility by default
  int bar;
};
```

### Implicit and explicit constructors ###

By default, always use `explicit` for single-argument, non-initializer list
constructors.

```cpp
struct MyStruct {
  // We don't want to implicitly convert ints to MyStructs
  explicit MyStruct(int foo);

  // Two-argument constructor; no need for explicit
  MyStruct(const std::string& name, int age);
};
```

### Public data members vs. getters/setters ###

Prefer declaring public member variables to using getters and setters. Getters
and setters that don't manage object state in a nontrivial way serve to bloat
the API and introduce unnecessary boilerplate.

Getters are, of course, encouraged for private members. Avoid prefixing getters
with `get`:

```cpp
struct Func {
  const SVInfoVec& staticVars() const;
  void setStaticVars(const SVInfoVec&);

  BuiltinFunction builtinFuncPtr() const;

  static constexpr ptrdiff_t sharedBaseOff();
};
```

### Declaration order ###

Adhere to the following order for declarations in a struct or class definition:

1. Friend classes.
2. Nested classes, enums, typedefs. (If possible, just declare the nested
   class and define it following the enclosing class definition.)
3. Constructors, destructor.
4. Member functions, including static functions, documented and grouped
   coherently.
5. Constants and static data members.
6. *All* instance data members, regardless of accessibility.

Private member functions can be interspersed with public functions, or
relegated to a single section before the data members. However, all instance
properties *must* occur contiguously at the end of the class definition.


## Other C++ Language Features ##
Very few language features are unconditionally banned. However, if you want to
use one of the more controversial constructs such as `goto` or `operator,()`,
you'd better have a convincing argument as to why it's better than the
alternatives. C++ is a very large and complex language and we don't want to
artificially limit what developers can do, but that puts a lot of
responsibility on your shoulders.

To maximize productivity but still reduce dependences barrier to entry,
we use a ```gcc-4.8```-compatible subset of C++11. In particular we do not use
```std::regex``` which is not properly supported before ```gcc-4.9```.
The CircleCI contbuild for Ubuntu-14.04 uses ```gcc-4.8``` and using ```std::regex```
or C++14 and newer features will break continuous integration.

Avoiding restrictions on useful language features (e.g., exceptions, templates,
C++11 lambdas) is a major motivating factor for maintaining our own style guide
rather than adopting an existing one.

### Namespaces ###

All TC code should be scoped in `namespace tc { /* everything */ }`. Large
submodules such as `tc::polyhedral` and `tc::tc` may be contained in their own
namespace within `tc`. We often use anonymous namespaces instead of the
`static` keyword to keep symbols internal to their translation unit. This is
mostly left up to the author; just keep in mind that classes and structs,
unlike functions and variables, *must* be in an anonymous namespace in order to
be properly hidden.

Avoid `using namespace`in headers. It is acceptable in `.cpp` files if
it will significantly aid in readability of the code that follows.
It is encouraged in local scopes such as blocks and functions.

Additionally, the different projects we use have their own redefinitions of
standard terms like ```isl::list```. We also
define our own C++ extensions to ISL ```tc::polyhedral::detail::Schedule``` and functional-style functions
```tc::functional::Map/Reduce/Filter/MapReduce```.
Such cases should be fully qualified, especially for code interfacing STL and ISL or Halide and ISL.

### Enums ###

Only use `enum class`. Old-style enums should not be used.

### Memory ownership ###
We follow these basic simple rules:
  1. ownership must not be carried or exchanged via naked pointers
  2. shared and unique ptrs should be used when a change of ownership is involved.
  3. functions that don't require ownership changes, should take const pointer, pointer or if really there is a good reason, shared_ptr (usually related to multithreading so not a concern here).
  4. factory functions create an object for you and give you ownership, they must return a unique_ptr, you can do with it as you please: release and move content into an object, make a shared_ptr out of it, keep a unique_ptr
  5. shared_ptr should be used when there is good expectation that members will outlive their class

## Naming ##

TC code adheres to some broad naming conventions.

When the convention is left open, or when working in the space of
one of the third-party dependences, prefer the local conventions used in the
file you are working on.

### Variables ###

Use `lowerCamelCase` or `lower_case_with_underscores` for all local variables,
adhering to whichever is the discernible local convention if possible.
For instance, in the isl related files, it is a good idea to use `lowerCamelCase`
for TC islpp extensions as opposed to `lower_case_with_underscores` for C or
the isl C++ bindings. This help better differentiate between the different conventions.

Static variables (whether declared in an anonymous namespace or with the `static`
keyword) should additionally be prefixed by `s` (e.g., `s_funcVec`).

Global variables should be avoided.
If you really need a global variable, implement it by returning a
reference from a function where the global local static variable. This uses local
static variable initialization, which is threadsafe in C++11.

### Constants ###

All constants should be prefixed with `k` and use `CamelCase`, e.g.,
`kInvalidHandle`. Prefer `constexpr` to `const` whenever possible.

### Class data members ###

As with variables, use `lowerCamelCase` or `lower_case_with_underscores`.
Private variables should be suffixed with `_`.
Prefer to leave public members unprefixed and unsuffixed.

### Functions ###

We generally prefer `lowerCamelCase` for header-exposed functions but this is
not strictly enforced when working at the interface between TC and other
projects (Halide, Caffe2, ISL).
As usual, follow the local naming conventions of the file you are working in.

If you are modeling a class after an existing pattern, such as an STL
container, prefer to follow the appropriate conventions (e.g.,
`my_list::push_back` is preferred over `my_list::pushBack`).

### Classes ###

Classes use `UpperCamelCase`, except when modeling existing patterns like STL
containers or smart pointers.

### Namespaces ###

New namespaces should use `lowercase`---and single-word namespaces are greatly
preferred for common usage.  For longer namespaces, use
`lower_case_with_underscores`.

### Other conventions ###

Prefer UpperCamelCase acronyms in new code (e.g., prefer `TcCompiler`
to `TCCompiler`). In this vein, prefer `Id` (e.g., `IslId`) to `ID`
(e.g., `ISLID`) in new code.


## Formatting ##

While consistent code formatting doesn't directly affect correctness, it makes
it easier to read and maintain. For this reason, we've added a push blocker lint
checker using [clang-format](https://github.com/facebookresearch/TensorComprehensions/#clang-format).

### General rules ###

- The [clang-format](https://github.com/facebookresearch/TensorComprehensions/#clang-format) checker is
  the source of truth. All guidelines below are ideally enforced by the linter but this
  is not fully stable yet.
- All indentation is to be done using spaces.
- Each indentation level is 2 spaces wide.
- Lines may be no longer than 80 characters, unless absolutely required for
  some syntactic reason.
- Lines should not have any trailing whitespace. This includes blank lines at
  non-zero indentation levels; the only character on those lines should be a
  newline.

### Types and variables ###

- When declaring a variable or typedef, the `*` and `&` characters for pointer
  and reference types should be adjacent to the type, not the name (e.g.,
  `const Func*& func`).
- Limit variable declarations to one per line.

### Function signatures ###

The following function signature is formatted properly:

```cpp
// If arguments would fit on 1 line:
inline void Func::appendParam(bool ref, const Func::ParamInfo& info) {
}

If the arguments need to wrap, then let `clang-format` do the wrapping.

Wrapped arguments should always be aligned with the argument on the previous
line. The opening curly brace should be on the same line as the last argument,
with the exception of class constructors (see the Constructor initializer list
section). When writing function declarations in headers, include argument names
unless they add no value:

```cpp
struct Person {
  // The single string argument here is obviously the name.
  void setName(const std::string&);

  // Two string arguments, so it's not obvious what each one is without names.
  void setFavorites(const std::string& color, const std::string& animal);
};
```

### Statements ###

Conditional and loop statements should be formatted like so:

```cpp
if (vmpc() == nullptr) {
  fprintf(stderr, "whoops!\n");
  std::abort();
}
```

Note that there is a single space after the `if` keyword, no spaces between
`condition` and the surrounding parentheses, and a single space between the `)`
and the `{`. As with all blocks, the body should be one indentation level
deeper than the `if`. If the *entire* statement (condition and body) fits on
one line, you may leave it on one line, *never* omitting the curly braces.
In all cases, braces are required.

```cpp
if (obj->_count == 0) { deleteObject(obj); }

for (auto block : blocks) { block->setParent(nullptr); }

if (veryLongVariableName.hasVeryLongFieldName() &&
    (rand() % 5) == 0) {
  launchRocket();
}
```

Avoid assignments in conditional expressions, unless the variable is declared
within the condition, e.g.,

```cpp
if (auto const unit = getMyUnit(from, these, args)) {
  // Do stuff with unit.
}
```

Prefer C++11 foreach syntax to explicit iterators:

```cpp
for (auto const& thing : thingVec) {
  // Do stuff with thing.
}
```

### Expressions ###

- All binary operators should have one space on each side, except for `.`,
  `->`, `.*`, and `->*` which should have zero.
- Do not include redundant parentheses unless you think the expression would be
  confusing to read otherwise. A good rule of thumb is that if you and/or your
  reviewers have to look at a chart of operator precedence to decide if the
  expression parses as expected, you probably need some extra parentheses. GCC
  or clang may suggest extra parens in certain situations; we compile with
  `-Werror` so you must always follow those guidelines.
- If an expression does not fit on one line, attempt to wrap it after an
  operator (rather than an identifier or keyword) and indent subsequent lines
  with the beginning of the current parenthesis/brace nesting level. For
  example, here are some long expressions, formatted appropriately:
```cpp
if (RuntimeOption::EvalJitRegionSelector != "" &&
    (RuntimeOption::EvalHHIRRefcountOpts ||
     RuntimeOption::EvalHHITExtraOptPass) &&
    Func::numLoadedFuncs() < 600) {
  // ...
}

longFunctionName(argumentTheFirst,
                 argumentTheSecond,
                 argumentTheThird,
                 argumentTheFourth);
```

- Function calls should be formatted primarily using the previous rule. If one
  or more of the arguments to the function is very wide, it may be necessary to
  shift all the arguments down one line and align them one level deeper than
  the current scope. This is always acceptable, but is especially common when
  passing lambdas:
```cpp
m_irb->ifThen(
  [&](Block* taken) {
    gen(CheckType, Type::Int, taken, src);
  },
  [&] {
    doSomeStuff();
    lotsOfNonTrivialCode();
    // etc...
  }
);
```

### Member initializer lists in constructor ###

If an initializer list can be kept on a single line, it is fine to do so:

```cpp
MyClass::MyClass(uint64_t idx) : m_idx(idx) {}

MyClass::MyClass(const Func* func) : m_idx(-1) {
  // Do stuff.
}
```

Otherwise, it is always correct to format lists this way:

```cpp
MyClass::MyClass(const Class* cls, const Func* func, const Class* ctx)
  : m_cls(cls)
  , m_func(func)
  , m_ctx(ctx)
  , m_isMyConditionMet(false)
{}

MyClass::MyClass(const Class* cls, const Func* func)
  : m_cls(cls)
  , m_func(func)
  , m_ctx(nullptr)
  , m_isMyConditionMet(false)
{
  // Do stuff.
}
```

### Namespaces ###

We don't nest namespaces very deeply, so prefer to keep the scoping to a single
line:

```cpp
namespace tc { namespace polyhedral { namespace detail {
///////////////////////////////////////////////////////////////////////////////

/*
 * Some nice documentation.
 */
struct SomeNiceThing {
  // some nice properties
};

///////////////////////////////////////////////////////////////////////////////
} // namespace detail
} // namespace polyhedral
} // namespace tc
```

Do not increase the indentation level when entering namespace scope. Instead,
consider adding a line of forward slashes as a separator, to more clearly
delineate the namespace (this is especially useful for anonymous namespaces in
source files). This form of delineation is encouraged, but we have no strict
convention for its formatting (you'll see 70- or 79- or 80-character
separators, with or without an extra newline between it and the braces, etc.).


## Comments ##

All public and private APIs in headers should be documented in detail. Names
and notions which are not obvious (e.g., "persistent" or "simple") should be
explained. Preconditions and postconditions should be noted.

Inline code comments are encouraged for complex logic, but their density is
left up to the author. Rather than summarizing/paraphrasing what your code is
doing, focus on explaining what overarching goal the code is achieving and/or
why that goal is necessary or desirable.

### Comment style ###

Here are some comment styles we use or avoid:

```cpp
// This style of comment is the most common for relatively short inline
// comments. It's fine if it's multi-line.
//
// It's also fine if it has line breaks. The extra newline aids readability in
// this case.

/*
 * This style of comment is the right one to use for struct/function
 * documentation. Prefer one star on the opening line, as opposed to the
 * doxygen standard of two.
 *
 * This is also sometimes used for inline code comments, although the // style
 * makes it easier to comment out blocks of code.
 */

struct ClassLikeThing {
  std::vector<const Func*> methods; // This is fine for short annotations.

  /* This is also ok, though try not to mix and match too much in a file. */
  std::vector<const ClassLikeThing*> parents;
};

/* Don't write multiline comments where some lines are missing their prefix.
   This is pretty weird. */
```

Try to use complete sentences in all but the shortest of comments. All comments
should be flowed to 80 characters in width.

### Separators ###

Delineate sections of code with a line of forward slashes. There is no strict
convention, but prefer lines of slashes to other delineators (e.g., `/*****/`,
five newlines, ASCII-art cartoon characters).

## Commits ##

Strive towards one commit per logical unit of change in your pull requests.
This means that a change should not be spread over multiple commits and
that a single commit should not contain multiple changes.
In particular, if you are working on a feature and you notice a mistake
in an earlier commit on your working branch (i.e., a commit that has
not been merged in yet), then squash in the fix in that commit.
This holds especially for compilation problems.  In general, each commit
should compile to ease both reviewing and bisecting.
Note that it is perfectly fine to commit more frequently (i.e., partial
changes) while working on a feature and even in "WIP" pull requests,
as long as the pieces are recombined (e.g., through an interactive rebase)
into logical units when the feature is ready for merging.
Force-pushing in PR branches is fine.

Each commit should have a proper commit message.
A commit message consists of a one-line summary, followed by an empty line and
the main body with more details focusing on the motivation of the change.
Both the one-line summary and the lines in the main body
should not exceed 80 characters.
The commit message should contain all the information that is needed
to understand the change within the git repository.
The main body can only be omitted if the motivation is completely obvious
from the one-line summary for an independent observer.

Coding Conventions for writing Tensor Comprehensions
====================================================

Please see the following documentation
[entry](https://facebookresearch.github.io/TensorComprehensions/coding_conventions.html)
on how to write Tensor Comprehensions in a standard legible fashion.
