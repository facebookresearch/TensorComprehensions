// Copyright (c) 2017-present, Facebook, Inc.
// #
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// #
//     http://www.apache.org/licenses/LICENSE-2.0
// #
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ##############################################################################
/// These are automatically generated C++ bindings for isl.
///
/// isl is a library for computing with integer sets and maps described by
/// Presburger formulas. On top of this, isl provides various tools for
/// polyhedral compilation, ranging from dependence analysis over scheduling
/// to AST generation.

#ifndef ISL_CPP
#define ISL_CPP

// clang-format off
#include <isl/space.h>
#include <isl/val.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/ilp.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/ast_build.h>
#include <isl/id.h>
#include <isl/fixed_box.h>

#include <isl/ctx.h>
#include <isl/options.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

/* ISL_USE_EXCEPTIONS should be defined to 1 if exceptions are available.
 * gcc and clang define __cpp_exceptions; MSVC and xlC define _CPPUNWIND.
 * Older versions of gcc (e.g., 4.9) only define __EXCEPTIONS.
 * If exceptions are not available, any error condition will result
 * in an abort.
 */
#ifndef ISL_USE_EXCEPTIONS
#if defined(__cpp_exceptions) || defined(_CPPUNWIND) || defined(__EXCEPTIONS)
#define ISL_USE_EXCEPTIONS	1
#else
#define ISL_USE_EXCEPTIONS	0
#endif
#endif

namespace isl {

class ctx {
	isl_ctx *ptr;
public:
	/* implicit */ ctx(isl_ctx *ctx) : ptr(ctx) {}
	isl_ctx *release() {
		auto tmp = ptr;
		ptr = nullptr;
		return tmp;
	}
	isl_ctx *get() {
		return ptr;
	}
};

/* Macros hiding try/catch.
 * If exceptions are not available, then no exceptions will be thrown and
 * there is nothing to catch.
 */
#if ISL_USE_EXCEPTIONS
#define ISL_CPP_TRY		try
#define ISL_CPP_CATCH_ALL	catch (...)
#else
#define ISL_CPP_TRY		if (1)
#define ISL_CPP_CATCH_ALL	if (0)
#endif

#if ISL_USE_EXCEPTIONS

/* Class capturing isl errors.
 *
 * The what() return value is stored in a reference counted string
 * to ensure that the copy constructor and the assignment operator
 * do not throw any exceptions.
 */
class exception : public std::exception {
	std::shared_ptr<std::string> what_str;

protected:
	inline exception(const char *what_arg, const char *msg,
		const char *file, int line);
public:
	exception() {}
	exception(const char *what_arg) {
		what_str = std::make_shared<std::string>(what_arg);
	}
	static inline exception create(enum isl_error error, const char *msg,
		const char *file, int line);
	static inline exception create_from_last_error(ctx ctx);
	virtual const char *what() const noexcept {
		return what_str->c_str();
	}

	/* Default behavior on error conditions that occur inside isl calls
	 * performed from inside the bindings.
	 * In the case exceptions are available, isl should continue
	 * without printing a warning since the warning message
	 * will be included in the exception thrown from inside the bindings.
	 */
	static constexpr auto on_error = ISL_ON_ERROR_CONTINUE;
	/* Wrapper for throwing an exception with the given message.
	 */
	static void throw_invalid(const char *msg, const char *file, int line) {
		throw create(isl_error_invalid, msg, file, line);
	}
	/* Wrapper for throwing an exception corresponding to the last
	 * error on "ctx".
	 */
	static void throw_last_error(ctx ctx) {
		throw create_from_last_error(ctx);
	}
};

/* Create an exception of a type described by "what_arg", with
 * error message "msg" in line "line" of file "file".
 *
 * Create a string holding the what() return value that
 * corresponds to what isl would have printed.
 * If no error message or no error file was set, then use "what_arg" instead.
 */
exception::exception(const char *what_arg, const char *msg, const char *file,
	int line)
{
	if (!msg || !file)
		what_str = std::make_shared<std::string>(what_arg);
	else
		what_str = std::make_shared<std::string>(std::string(file) +
				    ":" + std::to_string(line) + ": " + msg);
}

class exception_abort : public exception {
	friend exception;
	exception_abort(const char *msg, const char *file, int line) :
		exception("execution aborted", msg, file, line) {}
};

class exception_alloc : public exception {
	friend exception;
	exception_alloc(const char *msg, const char *file, int line) :
		exception("memory allocation failure", msg, file, line) {}
};

class exception_unknown : public exception {
	friend exception;
	exception_unknown(const char *msg, const char *file, int line) :
		exception("unknown failure", msg, file, line) {}
};

class exception_internal : public exception {
	friend exception;
	exception_internal(const char *msg, const char *file, int line) :
		exception("internal error", msg, file, line) {}
};

class exception_invalid : public exception {
	friend exception;
	exception_invalid(const char *msg, const char *file, int line) :
		exception("invalid argument", msg, file, line) {}
};

class exception_quota : public exception {
	friend exception;
	exception_quota(const char *msg, const char *file, int line) :
		exception("quota exceeded", msg, file, line) {}
};

class exception_unsupported : public exception {
	friend exception;
	exception_unsupported(const char *msg, const char *file, int line) :
		exception("unsupported operation", msg, file, line) {}
};

/* Create an exception of the class that corresponds to "error", with
 * error message "msg" in line "line" of file "file".
 *
 * isl_error_none is treated as an invalid error type.
 */
exception exception::create(enum isl_error error, const char *msg,
	const char *file, int line)
{
	switch (error) {
	case isl_error_none:
		break;
	case isl_error_abort: return exception_abort(msg, file, line);
	case isl_error_alloc: return exception_alloc(msg, file, line);
	case isl_error_unknown: return exception_unknown(msg, file, line);
	case isl_error_internal: return exception_internal(msg, file, line);
	case isl_error_invalid: return exception_invalid(msg, file, line);
	case isl_error_quota: return exception_quota(msg, file, line);
	case isl_error_unsupported:
				return exception_unsupported(msg, file, line);
	}

	throw exception_invalid("invalid error type", file, line);
}

/* Create an exception from the last error that occurred on "ctx" and
 * reset the error.
 *
 * If "ctx" is NULL or if it is not in an error state at the start,
 * then an invalid argument exception is thrown.
 */
exception exception::create_from_last_error(ctx ctx)
{
	enum isl_error error;
	const char *msg, *file;
	int line;

	error = isl_ctx_last_error(ctx.get());
	msg = isl_ctx_last_error_msg(ctx.get());
	file = isl_ctx_last_error_file(ctx.get());
	line = isl_ctx_last_error_line(ctx.get());
	isl_ctx_reset_error(ctx.get());

	return create(error, msg, file, line);
}

#else

#include <stdio.h>
#include <stdlib.h>

class exception {
public:
	/* Default behavior on error conditions that occur inside isl calls
	 * performed from inside the bindings.
	 * In the case exceptions are not available, isl should abort.
	 */
	static constexpr auto on_error = ISL_ON_ERROR_ABORT;
	/* Wrapper for throwing an exception with the given message.
	 * In the case exceptions are not available, print an error and abort.
	 */
	static void throw_invalid(const char *msg, const char *file, int line) {
		fprintf(stderr, "%s:%d: %s\n", file, line, msg);
		abort();
	}
	/* Wrapper for throwing an exception corresponding to the last
	 * error on "ctx".
	 * isl should already abort when an error condition occurs,
	 * so this function should never be called.
	 */
	static void throw_last_error(ctx ctx) {
		abort();
	}
};

#endif

/* Helper class for setting the on_error and resetting the option
 * to the original value when leaving the scope.
 */
class options_scoped_set_on_error {
	isl_ctx *ctx;
	int saved_on_error;
public:
	options_scoped_set_on_error(class ctx ctx, int on_error) {
		this->ctx = ctx.get();
		saved_on_error = isl_options_get_on_error(this->ctx);
		isl_options_set_on_error(this->ctx, on_error);
	}
	~options_scoped_set_on_error() {
		isl_options_set_on_error(ctx, saved_on_error);
	}
};

} // namespace isl

namespace isl {

// forward declarations
class aff;
class aff_list;
class ast_build;
class ast_expr;
class ast_expr_id;
class ast_expr_int;
class ast_expr_op;
class ast_node;
class ast_node_block;
class ast_node_for;
class ast_node_if;
class ast_node_list;
class ast_node_mark;
class ast_node_user;
class ast_op_access;
class ast_op_add;
class ast_op_address_of;
class ast_op_and;
class ast_op_and_then;
class ast_op_call;
class ast_op_cond;
class ast_op_div;
class ast_op_eq;
class ast_op_fdiv_q;
class ast_op_ge;
class ast_op_gt;
class ast_op_le;
class ast_op_lt;
class ast_op_max;
class ast_op_member;
class ast_op_min;
class ast_op_minus;
class ast_op_mul;
class ast_op_or;
class ast_op_or_else;
class ast_op_pdiv_q;
class ast_op_pdiv_r;
class ast_op_select;
class ast_op_sub;
class ast_op_zdiv_r;
class basic_map;
class basic_map_list;
class basic_set;
class basic_set_list;
class fixed_box;
class id;
class id_list;
class local_space;
class map;
class map_list;
class multi_aff;
class multi_id;
class multi_pw_aff;
class multi_union_pw_aff;
class multi_val;
class point;
class pw_aff;
class pw_aff_list;
class pw_multi_aff;
class schedule;
class schedule_constraints;
class schedule_node;
class schedule_node_band;
class schedule_node_context;
class schedule_node_domain;
class schedule_node_expansion;
class schedule_node_extension;
class schedule_node_filter;
class schedule_node_guard;
class schedule_node_leaf;
class schedule_node_mark;
class schedule_node_sequence;
class schedule_node_set;
class set;
class set_list;
class space;
class stride_info;
class union_access_info;
class union_flow;
class union_map;
class union_pw_aff;
class union_pw_aff_list;
class union_pw_multi_aff;
class union_set;
class union_set_list;
class val;
class val_list;

// declarations for isl::aff
inline aff manage(__isl_take isl_aff *ptr);
inline aff manage_copy(__isl_keep isl_aff *ptr);

class aff {
  friend inline aff manage(__isl_take isl_aff *ptr);
  friend inline aff manage_copy(__isl_keep isl_aff *ptr);

protected:
  isl_aff *ptr = nullptr;

  inline explicit aff(__isl_take isl_aff *ptr);

public:
  inline /* implicit */ aff();
  inline /* implicit */ aff(const aff &obj);
  inline explicit aff(local_space ls);
  inline explicit aff(local_space ls, val val);
  inline explicit aff(ctx ctx, const std::string &str);
  inline aff &operator=(aff obj);
  inline ~aff();
  inline __isl_give isl_aff *copy() const &;
  inline __isl_give isl_aff *copy() && = delete;
  inline __isl_keep isl_aff *get() const;
  inline __isl_give isl_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline aff add(aff aff2) const;
  inline aff add_constant(val v) const;
  inline aff add_constant_si(int v) const;
  inline aff ceil() const;
  inline aff div(aff aff2) const;
  inline set eq_set(aff aff2) const;
  inline val eval(point pnt) const;
  inline aff floor() const;
  inline set ge_set(aff aff2) const;
  inline val get_constant_val() const;
  inline val get_denominator_val() const;
  inline aff get_div(int pos) const;
  inline local_space get_local_space() const;
  inline space get_space() const;
  inline set gt_set(aff aff2) const;
  inline set le_set(aff aff2) const;
  inline set lt_set(aff aff2) const;
  inline aff mod(val mod) const;
  inline aff mul(aff aff2) const;
  inline set ne_set(aff aff2) const;
  inline aff neg() const;
  static inline aff param_on_domain_space(space space, id id);
  inline bool plain_is_equal(const aff &aff2) const;
  inline aff project_domain_on_params() const;
  inline aff pullback(multi_aff ma) const;
  inline aff scale(val v) const;
  inline aff scale_down(val v) const;
  inline aff scale_down_ui(unsigned int f) const;
  inline aff set_constant_si(int v) const;
  inline aff set_constant_val(val v) const;
  inline aff sub(aff aff2) const;
  inline aff unbind_params_insert_domain(multi_id domain) const;
  static inline aff zero_on_domain(space space);
  typedef isl_aff* isl_ptr_t;
};

// declarations for isl::aff_list
inline aff_list manage(__isl_take isl_aff_list *ptr);
inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

class aff_list {
  friend inline aff_list manage(__isl_take isl_aff_list *ptr);
  friend inline aff_list manage_copy(__isl_keep isl_aff_list *ptr);

protected:
  isl_aff_list *ptr = nullptr;

  inline explicit aff_list(__isl_take isl_aff_list *ptr);

public:
  inline /* implicit */ aff_list();
  inline /* implicit */ aff_list(const aff_list &obj);
  inline explicit aff_list(aff el);
  inline explicit aff_list(ctx ctx, int n);
  inline aff_list &operator=(aff_list obj);
  inline ~aff_list();
  inline __isl_give isl_aff_list *copy() const &;
  inline __isl_give isl_aff_list *copy() && = delete;
  inline __isl_keep isl_aff_list *get() const;
  inline __isl_give isl_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline aff_list add(aff el) const;
  inline aff_list concat(aff_list list2) const;
  inline aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(aff)> &fn) const;
  inline aff get_at(int index) const;
  inline aff_list reverse() const;
  inline int size() const;
  typedef isl_aff_list* isl_ptr_t;
};

// declarations for isl::ast_build
inline ast_build manage(__isl_take isl_ast_build *ptr);
inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

class ast_build {
  friend inline ast_build manage(__isl_take isl_ast_build *ptr);
  friend inline ast_build manage_copy(__isl_keep isl_ast_build *ptr);

protected:
  isl_ast_build *ptr = nullptr;

  inline explicit ast_build(__isl_take isl_ast_build *ptr);

public:
  inline /* implicit */ ast_build();
  inline /* implicit */ ast_build(const ast_build &obj);
  inline explicit ast_build(ctx ctx);
  inline ast_build &operator=(ast_build obj);
  inline ~ast_build();
  inline __isl_give isl_ast_build *copy() const &;
  inline __isl_give isl_ast_build *copy() && = delete;
  inline __isl_keep isl_ast_build *get() const;
  inline __isl_give isl_ast_build *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

private:
  inline ast_build &copy_callbacks(const ast_build &obj);
  struct at_each_domain_data {
    std::function<ast_node(ast_node, ast_build)> func;
    std::exception_ptr eptr;
  };
  std::shared_ptr<at_each_domain_data> at_each_domain_data;
  static inline isl_ast_node *at_each_domain(isl_ast_node *arg_0, isl_ast_build *arg_1, void *arg_2);
  inline void set_at_each_domain_data(const std::function<ast_node(ast_node, ast_build)> &fn);
public:
  inline ast_build set_at_each_domain(const std::function<ast_node(ast_node, ast_build)> &fn) const;
  inline ast_expr access_from(pw_multi_aff pma) const;
  inline ast_expr access_from(multi_pw_aff mpa) const;
  inline ast_node ast_from_schedule(union_map schedule) const;
  inline ast_expr call_from(pw_multi_aff pma) const;
  inline ast_expr call_from(multi_pw_aff mpa) const;
  inline ast_expr expr_from(set set) const;
  inline ast_expr expr_from(pw_aff pa) const;
  static inline ast_build from_context(set set);
  inline union_map get_schedule() const;
  inline space get_schedule_space() const;
  inline ast_node node_from(schedule schedule) const;
  inline ast_node node_from_schedule_map(union_map schedule) const;
  inline ast_build set_iterators(id_list iterators) const;
  typedef isl_ast_build* isl_ptr_t;
};

// declarations for isl::ast_expr
inline ast_expr manage(__isl_take isl_ast_expr *ptr);
inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

class ast_expr {
  friend inline ast_expr manage(__isl_take isl_ast_expr *ptr);
  friend inline ast_expr manage_copy(__isl_keep isl_ast_expr *ptr);

protected:
  isl_ast_expr *ptr = nullptr;

  inline explicit ast_expr(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr();
  inline /* implicit */ ast_expr(const ast_expr &obj);
  inline ast_expr &operator=(ast_expr obj);
  inline ~ast_expr();
  inline __isl_give isl_ast_expr *copy() const &;
  inline __isl_give isl_ast_expr *copy() && = delete;
  inline __isl_keep isl_ast_expr *get() const;
  inline __isl_give isl_ast_expr *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  template <class T> inline bool isa();
  template <class T> inline T as();
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline bool is_equal(const ast_expr &expr2) const;
  inline ast_expr set_op_arg(int pos, ast_expr arg) const;
  inline std::string to_C_str() const;
  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_expr_id

class ast_expr_id : public ast_expr {
  friend bool ast_expr::isa<ast_expr_id>();
  friend ast_expr_id ast_expr::as<ast_expr_id>();
  static const auto type = isl_ast_expr_id;

protected:
  inline explicit ast_expr_id(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_id();
  inline /* implicit */ ast_expr_id(const ast_expr_id &obj);
  inline ast_expr_id &operator=(ast_expr_id obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline id get_id() const;
  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_expr_int

class ast_expr_int : public ast_expr {
  friend bool ast_expr::isa<ast_expr_int>();
  friend ast_expr_int ast_expr::as<ast_expr_int>();
  static const auto type = isl_ast_expr_int;

protected:
  inline explicit ast_expr_int(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_int();
  inline /* implicit */ ast_expr_int(const ast_expr_int &obj);
  inline ast_expr_int &operator=(ast_expr_int obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline val get_val() const;
  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_expr_op

class ast_expr_op : public ast_expr {
  friend bool ast_expr::isa<ast_expr_op>();
  friend ast_expr_op ast_expr::as<ast_expr_op>();
  static const auto type = isl_ast_expr_op;

protected:
  inline explicit ast_expr_op(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_expr_op();
  inline /* implicit */ ast_expr_op(const ast_expr_op &obj);
  inline ast_expr_op &operator=(ast_expr_op obj);
  template <class T> inline bool isa();
  template <class T> inline T as();
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline ast_expr get_arg(int pos) const;
  inline int get_n_arg() const;
  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_node
inline ast_node manage(__isl_take isl_ast_node *ptr);
inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

class ast_node {
  friend inline ast_node manage(__isl_take isl_ast_node *ptr);
  friend inline ast_node manage_copy(__isl_keep isl_ast_node *ptr);

protected:
  isl_ast_node *ptr = nullptr;

  inline explicit ast_node(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node();
  inline /* implicit */ ast_node(const ast_node &obj);
  inline ast_node &operator=(ast_node obj);
  inline ~ast_node();
  inline __isl_give isl_ast_node *copy() const &;
  inline __isl_give isl_ast_node *copy() && = delete;
  inline __isl_keep isl_ast_node *get() const;
  inline __isl_give isl_ast_node *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  template <class T> inline bool isa();
  template <class T> inline T as();
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline id get_annotation() const;
  inline ast_node set_annotation(id annotation) const;
  inline std::string to_C_str() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_node_block

class ast_node_block : public ast_node {
  friend bool ast_node::isa<ast_node_block>();
  friend ast_node_block ast_node::as<ast_node_block>();
  static const auto type = isl_ast_node_block;

protected:
  inline explicit ast_node_block(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_block();
  inline /* implicit */ ast_node_block(const ast_node_block &obj);
  inline ast_node_block &operator=(ast_node_block obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline ast_node_list get_children() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_node_for

class ast_node_for : public ast_node {
  friend bool ast_node::isa<ast_node_for>();
  friend ast_node_for ast_node::as<ast_node_for>();
  static const auto type = isl_ast_node_for;

protected:
  inline explicit ast_node_for(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_for();
  inline /* implicit */ ast_node_for(const ast_node_for &obj);
  inline ast_node_for &operator=(ast_node_for obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline ast_node get_body() const;
  inline ast_expr get_cond() const;
  inline ast_expr get_inc() const;
  inline ast_expr get_init() const;
  inline ast_expr get_iterator() const;
  inline bool is_coincident() const;
  inline bool is_degenerate() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_node_if

class ast_node_if : public ast_node {
  friend bool ast_node::isa<ast_node_if>();
  friend ast_node_if ast_node::as<ast_node_if>();
  static const auto type = isl_ast_node_if;

protected:
  inline explicit ast_node_if(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_if();
  inline /* implicit */ ast_node_if(const ast_node_if &obj);
  inline ast_node_if &operator=(ast_node_if obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline ast_expr get_cond() const;
  inline ast_node get_else() const;
  inline ast_node get_then() const;
  inline bool has_else() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_node_list
inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

class ast_node_list {
  friend inline ast_node_list manage(__isl_take isl_ast_node_list *ptr);
  friend inline ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr);

protected:
  isl_ast_node_list *ptr = nullptr;

  inline explicit ast_node_list(__isl_take isl_ast_node_list *ptr);

public:
  inline /* implicit */ ast_node_list();
  inline /* implicit */ ast_node_list(const ast_node_list &obj);
  inline explicit ast_node_list(ast_node el);
  inline explicit ast_node_list(ctx ctx, int n);
  inline ast_node_list &operator=(ast_node_list obj);
  inline ~ast_node_list();
  inline __isl_give isl_ast_node_list *copy() const &;
  inline __isl_give isl_ast_node_list *copy() && = delete;
  inline __isl_keep isl_ast_node_list *get() const;
  inline __isl_give isl_ast_node_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline ast_node_list add(ast_node el) const;
  inline ast_node_list concat(ast_node_list list2) const;
  inline ast_node_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(ast_node)> &fn) const;
  inline ast_node get_at(int index) const;
  inline ast_node_list reverse() const;
  inline int size() const;
  typedef isl_ast_node_list* isl_ptr_t;
};

// declarations for isl::ast_node_mark

class ast_node_mark : public ast_node {
  friend bool ast_node::isa<ast_node_mark>();
  friend ast_node_mark ast_node::as<ast_node_mark>();
  static const auto type = isl_ast_node_mark;

protected:
  inline explicit ast_node_mark(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_mark();
  inline /* implicit */ ast_node_mark(const ast_node_mark &obj);
  inline ast_node_mark &operator=(ast_node_mark obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline id get_id() const;
  inline ast_node get_node() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_node_user

class ast_node_user : public ast_node {
  friend bool ast_node::isa<ast_node_user>();
  friend ast_node_user ast_node::as<ast_node_user>();
  static const auto type = isl_ast_node_user;

protected:
  inline explicit ast_node_user(__isl_take isl_ast_node *ptr);

public:
  inline /* implicit */ ast_node_user();
  inline /* implicit */ ast_node_user(const ast_node_user &obj);
  inline ast_node_user &operator=(ast_node_user obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline ast_expr get_expr() const;
  typedef isl_ast_node* isl_ptr_t;
};

// declarations for isl::ast_op_access

class ast_op_access : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_access>();
  friend ast_op_access ast_expr_op::as<ast_op_access>();
  static const auto type = isl_ast_op_access;

protected:
  inline explicit ast_op_access(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_access();
  inline /* implicit */ ast_op_access(const ast_op_access &obj);
  inline ast_op_access &operator=(ast_op_access obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_add

class ast_op_add : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_add>();
  friend ast_op_add ast_expr_op::as<ast_op_add>();
  static const auto type = isl_ast_op_add;

protected:
  inline explicit ast_op_add(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_add();
  inline /* implicit */ ast_op_add(const ast_op_add &obj);
  inline ast_op_add &operator=(ast_op_add obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_address_of

class ast_op_address_of : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_address_of>();
  friend ast_op_address_of ast_expr_op::as<ast_op_address_of>();
  static const auto type = isl_ast_op_address_of;

protected:
  inline explicit ast_op_address_of(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_address_of();
  inline /* implicit */ ast_op_address_of(const ast_op_address_of &obj);
  inline ast_op_address_of &operator=(ast_op_address_of obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_and

class ast_op_and : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_and>();
  friend ast_op_and ast_expr_op::as<ast_op_and>();
  static const auto type = isl_ast_op_and;

protected:
  inline explicit ast_op_and(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_and();
  inline /* implicit */ ast_op_and(const ast_op_and &obj);
  inline ast_op_and &operator=(ast_op_and obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_and_then

class ast_op_and_then : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_and_then>();
  friend ast_op_and_then ast_expr_op::as<ast_op_and_then>();
  static const auto type = isl_ast_op_and_then;

protected:
  inline explicit ast_op_and_then(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_and_then();
  inline /* implicit */ ast_op_and_then(const ast_op_and_then &obj);
  inline ast_op_and_then &operator=(ast_op_and_then obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_call

class ast_op_call : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_call>();
  friend ast_op_call ast_expr_op::as<ast_op_call>();
  static const auto type = isl_ast_op_call;

protected:
  inline explicit ast_op_call(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_call();
  inline /* implicit */ ast_op_call(const ast_op_call &obj);
  inline ast_op_call &operator=(ast_op_call obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_cond

class ast_op_cond : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_cond>();
  friend ast_op_cond ast_expr_op::as<ast_op_cond>();
  static const auto type = isl_ast_op_cond;

protected:
  inline explicit ast_op_cond(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_cond();
  inline /* implicit */ ast_op_cond(const ast_op_cond &obj);
  inline ast_op_cond &operator=(ast_op_cond obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_div

class ast_op_div : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_div>();
  friend ast_op_div ast_expr_op::as<ast_op_div>();
  static const auto type = isl_ast_op_div;

protected:
  inline explicit ast_op_div(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_div();
  inline /* implicit */ ast_op_div(const ast_op_div &obj);
  inline ast_op_div &operator=(ast_op_div obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_eq

class ast_op_eq : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_eq>();
  friend ast_op_eq ast_expr_op::as<ast_op_eq>();
  static const auto type = isl_ast_op_eq;

protected:
  inline explicit ast_op_eq(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_eq();
  inline /* implicit */ ast_op_eq(const ast_op_eq &obj);
  inline ast_op_eq &operator=(ast_op_eq obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_fdiv_q

class ast_op_fdiv_q : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_fdiv_q>();
  friend ast_op_fdiv_q ast_expr_op::as<ast_op_fdiv_q>();
  static const auto type = isl_ast_op_fdiv_q;

protected:
  inline explicit ast_op_fdiv_q(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_fdiv_q();
  inline /* implicit */ ast_op_fdiv_q(const ast_op_fdiv_q &obj);
  inline ast_op_fdiv_q &operator=(ast_op_fdiv_q obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_ge

class ast_op_ge : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_ge>();
  friend ast_op_ge ast_expr_op::as<ast_op_ge>();
  static const auto type = isl_ast_op_ge;

protected:
  inline explicit ast_op_ge(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_ge();
  inline /* implicit */ ast_op_ge(const ast_op_ge &obj);
  inline ast_op_ge &operator=(ast_op_ge obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_gt

class ast_op_gt : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_gt>();
  friend ast_op_gt ast_expr_op::as<ast_op_gt>();
  static const auto type = isl_ast_op_gt;

protected:
  inline explicit ast_op_gt(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_gt();
  inline /* implicit */ ast_op_gt(const ast_op_gt &obj);
  inline ast_op_gt &operator=(ast_op_gt obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_le

class ast_op_le : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_le>();
  friend ast_op_le ast_expr_op::as<ast_op_le>();
  static const auto type = isl_ast_op_le;

protected:
  inline explicit ast_op_le(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_le();
  inline /* implicit */ ast_op_le(const ast_op_le &obj);
  inline ast_op_le &operator=(ast_op_le obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_lt

class ast_op_lt : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_lt>();
  friend ast_op_lt ast_expr_op::as<ast_op_lt>();
  static const auto type = isl_ast_op_lt;

protected:
  inline explicit ast_op_lt(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_lt();
  inline /* implicit */ ast_op_lt(const ast_op_lt &obj);
  inline ast_op_lt &operator=(ast_op_lt obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_max

class ast_op_max : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_max>();
  friend ast_op_max ast_expr_op::as<ast_op_max>();
  static const auto type = isl_ast_op_max;

protected:
  inline explicit ast_op_max(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_max();
  inline /* implicit */ ast_op_max(const ast_op_max &obj);
  inline ast_op_max &operator=(ast_op_max obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_member

class ast_op_member : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_member>();
  friend ast_op_member ast_expr_op::as<ast_op_member>();
  static const auto type = isl_ast_op_member;

protected:
  inline explicit ast_op_member(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_member();
  inline /* implicit */ ast_op_member(const ast_op_member &obj);
  inline ast_op_member &operator=(ast_op_member obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_min

class ast_op_min : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_min>();
  friend ast_op_min ast_expr_op::as<ast_op_min>();
  static const auto type = isl_ast_op_min;

protected:
  inline explicit ast_op_min(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_min();
  inline /* implicit */ ast_op_min(const ast_op_min &obj);
  inline ast_op_min &operator=(ast_op_min obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_minus

class ast_op_minus : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_minus>();
  friend ast_op_minus ast_expr_op::as<ast_op_minus>();
  static const auto type = isl_ast_op_minus;

protected:
  inline explicit ast_op_minus(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_minus();
  inline /* implicit */ ast_op_minus(const ast_op_minus &obj);
  inline ast_op_minus &operator=(ast_op_minus obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_mul

class ast_op_mul : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_mul>();
  friend ast_op_mul ast_expr_op::as<ast_op_mul>();
  static const auto type = isl_ast_op_mul;

protected:
  inline explicit ast_op_mul(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_mul();
  inline /* implicit */ ast_op_mul(const ast_op_mul &obj);
  inline ast_op_mul &operator=(ast_op_mul obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_or

class ast_op_or : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_or>();
  friend ast_op_or ast_expr_op::as<ast_op_or>();
  static const auto type = isl_ast_op_or;

protected:
  inline explicit ast_op_or(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_or();
  inline /* implicit */ ast_op_or(const ast_op_or &obj);
  inline ast_op_or &operator=(ast_op_or obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_or_else

class ast_op_or_else : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_or_else>();
  friend ast_op_or_else ast_expr_op::as<ast_op_or_else>();
  static const auto type = isl_ast_op_or_else;

protected:
  inline explicit ast_op_or_else(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_or_else();
  inline /* implicit */ ast_op_or_else(const ast_op_or_else &obj);
  inline ast_op_or_else &operator=(ast_op_or_else obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_pdiv_q

class ast_op_pdiv_q : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_pdiv_q>();
  friend ast_op_pdiv_q ast_expr_op::as<ast_op_pdiv_q>();
  static const auto type = isl_ast_op_pdiv_q;

protected:
  inline explicit ast_op_pdiv_q(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_pdiv_q();
  inline /* implicit */ ast_op_pdiv_q(const ast_op_pdiv_q &obj);
  inline ast_op_pdiv_q &operator=(ast_op_pdiv_q obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_pdiv_r

class ast_op_pdiv_r : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_pdiv_r>();
  friend ast_op_pdiv_r ast_expr_op::as<ast_op_pdiv_r>();
  static const auto type = isl_ast_op_pdiv_r;

protected:
  inline explicit ast_op_pdiv_r(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_pdiv_r();
  inline /* implicit */ ast_op_pdiv_r(const ast_op_pdiv_r &obj);
  inline ast_op_pdiv_r &operator=(ast_op_pdiv_r obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_select

class ast_op_select : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_select>();
  friend ast_op_select ast_expr_op::as<ast_op_select>();
  static const auto type = isl_ast_op_select;

protected:
  inline explicit ast_op_select(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_select();
  inline /* implicit */ ast_op_select(const ast_op_select &obj);
  inline ast_op_select &operator=(ast_op_select obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_sub

class ast_op_sub : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_sub>();
  friend ast_op_sub ast_expr_op::as<ast_op_sub>();
  static const auto type = isl_ast_op_sub;

protected:
  inline explicit ast_op_sub(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_sub();
  inline /* implicit */ ast_op_sub(const ast_op_sub &obj);
  inline ast_op_sub &operator=(ast_op_sub obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::ast_op_zdiv_r

class ast_op_zdiv_r : public ast_expr_op {
  friend bool ast_expr_op::isa<ast_op_zdiv_r>();
  friend ast_op_zdiv_r ast_expr_op::as<ast_op_zdiv_r>();
  static const auto type = isl_ast_op_zdiv_r;

protected:
  inline explicit ast_op_zdiv_r(__isl_take isl_ast_expr *ptr);

public:
  inline /* implicit */ ast_op_zdiv_r();
  inline /* implicit */ ast_op_zdiv_r(const ast_op_zdiv_r &obj);
  inline ast_op_zdiv_r &operator=(ast_op_zdiv_r obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_ast_expr* isl_ptr_t;
};

// declarations for isl::basic_map
inline basic_map manage(__isl_take isl_basic_map *ptr);
inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

class basic_map {
  friend inline basic_map manage(__isl_take isl_basic_map *ptr);
  friend inline basic_map manage_copy(__isl_keep isl_basic_map *ptr);

protected:
  isl_basic_map *ptr = nullptr;

  inline explicit basic_map(__isl_take isl_basic_map *ptr);

public:
  inline /* implicit */ basic_map();
  inline /* implicit */ basic_map(const basic_map &obj);
  inline explicit basic_map(ctx ctx, const std::string &str);
  inline explicit basic_map(basic_set domain, basic_set range);
  inline explicit basic_map(aff aff);
  inline explicit basic_map(multi_aff maff);
  inline basic_map &operator=(basic_map obj);
  inline ~basic_map();
  inline __isl_give isl_basic_map *copy() const &;
  inline __isl_give isl_basic_map *copy() && = delete;
  inline __isl_keep isl_basic_map *get() const;
  inline __isl_give isl_basic_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline basic_map affine_hull() const;
  inline basic_map apply_domain(basic_map bmap2) const;
  inline basic_map apply_range(basic_map bmap2) const;
  inline bool can_curry() const;
  inline bool can_uncurry() const;
  inline basic_map curry() const;
  inline basic_set deltas() const;
  inline basic_map detect_equalities() const;
  inline basic_set domain() const;
  static inline basic_map empty(space space);
  inline basic_map flatten() const;
  inline basic_map flatten_domain() const;
  inline basic_map flatten_range() const;
  static inline basic_map from_domain(basic_set bset);
  static inline basic_map from_range(basic_set bset);
  inline space get_space() const;
  inline basic_map gist(basic_map context) const;
  inline basic_map intersect(basic_map bmap2) const;
  inline basic_map intersect_domain(basic_set bset) const;
  inline basic_map intersect_range(basic_set bset) const;
  inline bool is_empty() const;
  inline bool is_equal(const basic_map &bmap2) const;
  inline bool is_subset(const basic_map &bmap2) const;
  inline map lexmax() const;
  inline map lexmin() const;
  inline basic_map reverse() const;
  inline basic_map sample() const;
  inline basic_map uncurry() const;
  inline map unite(basic_map bmap2) const;
  inline basic_set wrap() const;
  typedef isl_basic_map* isl_ptr_t;
};

// declarations for isl::basic_map_list
inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

class basic_map_list {
  friend inline basic_map_list manage(__isl_take isl_basic_map_list *ptr);
  friend inline basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr);

protected:
  isl_basic_map_list *ptr = nullptr;

  inline explicit basic_map_list(__isl_take isl_basic_map_list *ptr);

public:
  inline /* implicit */ basic_map_list();
  inline /* implicit */ basic_map_list(const basic_map_list &obj);
  inline explicit basic_map_list(basic_map el);
  inline explicit basic_map_list(ctx ctx, int n);
  inline basic_map_list &operator=(basic_map_list obj);
  inline ~basic_map_list();
  inline __isl_give isl_basic_map_list *copy() const &;
  inline __isl_give isl_basic_map_list *copy() && = delete;
  inline __isl_keep isl_basic_map_list *get() const;
  inline __isl_give isl_basic_map_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline basic_map_list add(basic_map el) const;
  inline basic_map_list concat(basic_map_list list2) const;
  inline basic_map_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(basic_map)> &fn) const;
  inline basic_map get_at(int index) const;
  inline basic_map intersect() const;
  inline basic_map_list reverse() const;
  inline int size() const;
  typedef isl_basic_map_list* isl_ptr_t;
};

// declarations for isl::basic_set
inline basic_set manage(__isl_take isl_basic_set *ptr);
inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

class basic_set {
  friend inline basic_set manage(__isl_take isl_basic_set *ptr);
  friend inline basic_set manage_copy(__isl_keep isl_basic_set *ptr);

protected:
  isl_basic_set *ptr = nullptr;

  inline explicit basic_set(__isl_take isl_basic_set *ptr);

public:
  inline /* implicit */ basic_set();
  inline /* implicit */ basic_set(const basic_set &obj);
  inline /* implicit */ basic_set(point pnt);
  inline explicit basic_set(ctx ctx, const std::string &str);
  inline basic_set &operator=(basic_set obj);
  inline ~basic_set();
  inline __isl_give isl_basic_set *copy() const &;
  inline __isl_give isl_basic_set *copy() && = delete;
  inline __isl_keep isl_basic_set *get() const;
  inline __isl_give isl_basic_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline basic_set affine_hull() const;
  inline basic_set apply(basic_map bmap) const;
  inline set compute_divs() const;
  inline basic_set detect_equalities() const;
  inline val dim_max_val(int pos) const;
  inline basic_set flatten() const;
  inline basic_set from_params() const;
  inline space get_space() const;
  inline basic_set gist(basic_set context) const;
  inline basic_set intersect(basic_set bset2) const;
  inline basic_set intersect_params(basic_set bset2) const;
  inline bool is_empty() const;
  inline bool is_equal(const basic_set &bset2) const;
  inline bool is_subset(const basic_set &bset2) const;
  inline bool is_universe() const;
  inline bool is_wrapping() const;
  inline set lexmax() const;
  inline set lexmin() const;
  inline val max_val(const aff &obj) const;
  inline unsigned int n_dim() const;
  inline unsigned int n_param() const;
  static inline basic_set nat_universe(space dim);
  inline basic_set params() const;
  inline bool plain_is_universe() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline basic_set set_tuple_id(id id) const;
  inline set unite(basic_set bset2) const;
  static inline basic_set universe(space space);
  inline basic_map unwrap() const;
  typedef isl_basic_set* isl_ptr_t;
};

// declarations for isl::basic_set_list
inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

class basic_set_list {
  friend inline basic_set_list manage(__isl_take isl_basic_set_list *ptr);
  friend inline basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr);

protected:
  isl_basic_set_list *ptr = nullptr;

  inline explicit basic_set_list(__isl_take isl_basic_set_list *ptr);

public:
  inline /* implicit */ basic_set_list();
  inline /* implicit */ basic_set_list(const basic_set_list &obj);
  inline explicit basic_set_list(basic_set el);
  inline explicit basic_set_list(ctx ctx, int n);
  inline basic_set_list &operator=(basic_set_list obj);
  inline ~basic_set_list();
  inline __isl_give isl_basic_set_list *copy() const &;
  inline __isl_give isl_basic_set_list *copy() && = delete;
  inline __isl_keep isl_basic_set_list *get() const;
  inline __isl_give isl_basic_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline basic_set_list add(basic_set el) const;
  inline basic_set_list concat(basic_set_list list2) const;
  inline basic_set_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(basic_set)> &fn) const;
  inline basic_set get_at(int index) const;
  inline basic_set_list reverse() const;
  inline int size() const;
  typedef isl_basic_set_list* isl_ptr_t;
};

// declarations for isl::fixed_box
inline fixed_box manage(__isl_take isl_fixed_box *ptr);
inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

class fixed_box {
  friend inline fixed_box manage(__isl_take isl_fixed_box *ptr);
  friend inline fixed_box manage_copy(__isl_keep isl_fixed_box *ptr);

protected:
  isl_fixed_box *ptr = nullptr;

  inline explicit fixed_box(__isl_take isl_fixed_box *ptr);

public:
  inline /* implicit */ fixed_box();
  inline /* implicit */ fixed_box(const fixed_box &obj);
  inline fixed_box &operator=(fixed_box obj);
  inline ~fixed_box();
  inline __isl_give isl_fixed_box *copy() const &;
  inline __isl_give isl_fixed_box *copy() && = delete;
  inline __isl_keep isl_fixed_box *get() const;
  inline __isl_give isl_fixed_box *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline multi_aff get_offset() const;
  inline multi_val get_size() const;
  inline space get_space() const;
  inline bool is_valid() const;
  typedef isl_fixed_box* isl_ptr_t;
};

// declarations for isl::id
inline id manage(__isl_take isl_id *ptr);
inline id manage_copy(__isl_keep isl_id *ptr);

class id {
  friend inline id manage(__isl_take isl_id *ptr);
  friend inline id manage_copy(__isl_keep isl_id *ptr);

protected:
  isl_id *ptr = nullptr;

  inline explicit id(__isl_take isl_id *ptr);

public:
  inline /* implicit */ id();
  inline /* implicit */ id(const id &obj);
  inline explicit id(ctx ctx, const std::string &str);
  inline id &operator=(id obj);
  inline ~id();
  inline __isl_give isl_id *copy() const &;
  inline __isl_give isl_id *copy() && = delete;
  inline __isl_keep isl_id *get() const;
  inline __isl_give isl_id *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline std::string get_name() const;
  typedef isl_id* isl_ptr_t;
};

// declarations for isl::id_list
inline id_list manage(__isl_take isl_id_list *ptr);
inline id_list manage_copy(__isl_keep isl_id_list *ptr);

class id_list {
  friend inline id_list manage(__isl_take isl_id_list *ptr);
  friend inline id_list manage_copy(__isl_keep isl_id_list *ptr);

protected:
  isl_id_list *ptr = nullptr;

  inline explicit id_list(__isl_take isl_id_list *ptr);

public:
  inline /* implicit */ id_list();
  inline /* implicit */ id_list(const id_list &obj);
  inline explicit id_list(id el);
  inline explicit id_list(ctx ctx, int n);
  inline id_list &operator=(id_list obj);
  inline ~id_list();
  inline __isl_give isl_id_list *copy() const &;
  inline __isl_give isl_id_list *copy() && = delete;
  inline __isl_keep isl_id_list *get() const;
  inline __isl_give isl_id_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline id_list add(id el) const;
  inline id_list concat(id_list list2) const;
  inline id_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(id)> &fn) const;
  inline id get_at(int index) const;
  inline id_list reverse() const;
  inline int size() const;
  typedef isl_id_list* isl_ptr_t;
};

// declarations for isl::local_space
inline local_space manage(__isl_take isl_local_space *ptr);
inline local_space manage_copy(__isl_keep isl_local_space *ptr);

class local_space {
  friend inline local_space manage(__isl_take isl_local_space *ptr);
  friend inline local_space manage_copy(__isl_keep isl_local_space *ptr);

protected:
  isl_local_space *ptr = nullptr;

  inline explicit local_space(__isl_take isl_local_space *ptr);

public:
  inline /* implicit */ local_space();
  inline /* implicit */ local_space(const local_space &obj);
  inline explicit local_space(space dim);
  inline local_space &operator=(local_space obj);
  inline ~local_space();
  inline __isl_give isl_local_space *copy() const &;
  inline __isl_give isl_local_space *copy() && = delete;
  inline __isl_keep isl_local_space *get() const;
  inline __isl_give isl_local_space *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline local_space domain() const;
  inline local_space flatten_domain() const;
  inline local_space flatten_range() const;
  inline local_space from_domain() const;
  inline aff get_div(int pos) const;
  inline space get_space() const;
  inline local_space intersect(local_space ls2) const;
  inline bool is_equal(const local_space &ls2) const;
  inline bool is_params() const;
  inline bool is_set() const;
  inline basic_map lifting() const;
  inline local_space range() const;
  inline local_space wrap() const;
  typedef isl_local_space* isl_ptr_t;
};

// declarations for isl::map
inline map manage(__isl_take isl_map *ptr);
inline map manage_copy(__isl_keep isl_map *ptr);

class map {
  friend inline map manage(__isl_take isl_map *ptr);
  friend inline map manage_copy(__isl_keep isl_map *ptr);

protected:
  isl_map *ptr = nullptr;

  inline explicit map(__isl_take isl_map *ptr);

public:
  inline /* implicit */ map();
  inline /* implicit */ map(const map &obj);
  inline explicit map(ctx ctx, const std::string &str);
  inline /* implicit */ map(basic_map bmap);
  inline explicit map(set domain, set range);
  inline explicit map(aff aff);
  inline explicit map(multi_aff maff);
  inline map &operator=(map obj);
  inline ~map();
  inline __isl_give isl_map *copy() const &;
  inline __isl_give isl_map *copy() && = delete;
  inline __isl_keep isl_map *get() const;
  inline __isl_give isl_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline basic_map affine_hull() const;
  inline map apply_domain(map map2) const;
  inline map apply_range(map map2) const;
  inline bool can_curry() const;
  inline bool can_range_curry() const;
  inline bool can_uncurry() const;
  inline map coalesce() const;
  inline map complement() const;
  inline map compute_divs() const;
  inline map curry() const;
  inline set deltas() const;
  inline map detect_equalities() const;
  inline set domain() const;
  inline map domain_factor_domain() const;
  inline map domain_factor_range() const;
  inline map domain_map() const;
  inline map domain_product(map map2) const;
  static inline map empty(space space);
  inline map flatten() const;
  inline map flatten_domain() const;
  inline map flatten_range() const;
  inline void foreach_basic_map(const std::function<void(basic_map)> &fn) const;
  static inline map from(pw_multi_aff pma);
  static inline map from(union_map umap);
  static inline map from_domain(set set);
  static inline map from_range(set set);
  inline basic_map_list get_basic_map_list() const;
  inline fixed_box get_range_simple_fixed_box_hull() const;
  inline stride_info get_range_stride_info(int pos) const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  inline map gist(map context) const;
  inline map gist_domain(set context) const;
  static inline map identity(space dim);
  inline map intersect(map map2) const;
  inline map intersect_domain(set set) const;
  inline map intersect_params(set params) const;
  inline map intersect_range(set set) const;
  inline bool is_bijective() const;
  inline bool is_disjoint(const map &map2) const;
  inline bool is_empty() const;
  inline bool is_equal(const map &map2) const;
  inline bool is_injective() const;
  inline bool is_single_valued() const;
  inline bool is_strict_subset(const map &map2) const;
  inline bool is_subset(const map &map2) const;
  inline map lexmax() const;
  inline map lexmin() const;
  inline int n_basic_map() const;
  inline set params() const;
  inline basic_map polyhedral_hull() const;
  inline map preimage_domain(multi_aff ma) const;
  inline map preimage_range(multi_aff ma) const;
  inline set range() const;
  inline map range_curry() const;
  inline map range_factor_domain() const;
  inline map range_factor_range() const;
  inline map range_map() const;
  inline map range_product(map map2) const;
  inline map reverse() const;
  inline basic_map sample() const;
  inline map set_range_tuple_id(id id) const;
  inline basic_map simple_hull() const;
  inline map subtract(map map2) const;
  inline map sum(map map2) const;
  inline map uncurry() const;
  inline map unite(map map2) const;
  static inline map universe(space space);
  inline basic_map unshifted_simple_hull() const;
  inline set wrap() const;
  inline map zip() const;
  typedef isl_map* isl_ptr_t;
};

// declarations for isl::map_list
inline map_list manage(__isl_take isl_map_list *ptr);
inline map_list manage_copy(__isl_keep isl_map_list *ptr);

class map_list {
  friend inline map_list manage(__isl_take isl_map_list *ptr);
  friend inline map_list manage_copy(__isl_keep isl_map_list *ptr);

protected:
  isl_map_list *ptr = nullptr;

  inline explicit map_list(__isl_take isl_map_list *ptr);

public:
  inline /* implicit */ map_list();
  inline /* implicit */ map_list(const map_list &obj);
  inline explicit map_list(map el);
  inline explicit map_list(ctx ctx, int n);
  inline map_list &operator=(map_list obj);
  inline ~map_list();
  inline __isl_give isl_map_list *copy() const &;
  inline __isl_give isl_map_list *copy() && = delete;
  inline __isl_keep isl_map_list *get() const;
  inline __isl_give isl_map_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline map_list add(map el) const;
  inline map_list concat(map_list list2) const;
  inline map_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(map)> &fn) const;
  inline map get_at(int index) const;
  inline map_list reverse() const;
  inline int size() const;
  typedef isl_map_list* isl_ptr_t;
};

// declarations for isl::multi_aff
inline multi_aff manage(__isl_take isl_multi_aff *ptr);
inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

class multi_aff {
  friend inline multi_aff manage(__isl_take isl_multi_aff *ptr);
  friend inline multi_aff manage_copy(__isl_keep isl_multi_aff *ptr);

protected:
  isl_multi_aff *ptr = nullptr;

  inline explicit multi_aff(__isl_take isl_multi_aff *ptr);

public:
  inline /* implicit */ multi_aff();
  inline /* implicit */ multi_aff(const multi_aff &obj);
  inline explicit multi_aff(ctx ctx, const std::string &str);
  inline explicit multi_aff(space space, aff_list list);
  inline /* implicit */ multi_aff(aff aff);
  inline multi_aff &operator=(multi_aff obj);
  inline ~multi_aff();
  inline __isl_give isl_multi_aff *copy() const &;
  inline __isl_give isl_multi_aff *copy() && = delete;
  inline __isl_keep isl_multi_aff *get() const;
  inline __isl_give isl_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline multi_aff add(multi_aff multi2) const;
  inline multi_aff align_params(space model) const;
  static inline multi_aff domain_map(space space);
  inline multi_aff factor_domain() const;
  inline multi_aff factor_range() const;
  inline multi_aff flat_range_product(multi_aff multi2) const;
  inline multi_aff flatten_range() const;
  inline multi_aff floor() const;
  inline multi_aff from_range() const;
  inline aff get_aff(int pos) const;
  inline aff_list get_aff_list() const;
  inline space get_domain_space() const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  static inline multi_aff identity(space space);
  inline multi_aff mod(multi_val mv) const;
  inline multi_aff neg() const;
  inline multi_aff product(multi_aff multi2) const;
  inline multi_aff pullback(multi_aff ma2) const;
  inline multi_aff range_factor_domain() const;
  inline multi_aff range_factor_range() const;
  static inline multi_aff range_map(space space);
  inline multi_aff range_product(multi_aff multi2) const;
  inline multi_aff range_splice(unsigned int pos, multi_aff multi2) const;
  inline multi_aff reset_user() const;
  inline multi_aff scale(val v) const;
  inline multi_aff scale(multi_val mv) const;
  inline multi_aff scale_down(val v) const;
  inline multi_aff scale_down(multi_val mv) const;
  inline multi_aff set_aff(int pos, aff el) const;
  inline multi_aff set_range_tuple_id(id id) const;
  inline int size() const;
  inline multi_aff splice(unsigned int in_pos, unsigned int out_pos, multi_aff multi2) const;
  inline multi_aff sub(multi_aff multi2) const;
  static inline multi_aff wrapped_range_map(space space);
  static inline multi_aff zero(space space);
  typedef isl_multi_aff* isl_ptr_t;
};

// declarations for isl::multi_id
inline multi_id manage(__isl_take isl_multi_id *ptr);
inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

class multi_id {
  friend inline multi_id manage(__isl_take isl_multi_id *ptr);
  friend inline multi_id manage_copy(__isl_keep isl_multi_id *ptr);

protected:
  isl_multi_id *ptr = nullptr;

  inline explicit multi_id(__isl_take isl_multi_id *ptr);

public:
  inline /* implicit */ multi_id();
  inline /* implicit */ multi_id(const multi_id &obj);
  inline explicit multi_id(space space, id_list list);
  inline multi_id &operator=(multi_id obj);
  inline ~multi_id();
  inline __isl_give isl_multi_id *copy() const &;
  inline __isl_give isl_multi_id *copy() && = delete;
  inline __isl_keep isl_multi_id *get() const;
  inline __isl_give isl_multi_id *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline multi_id align_params(space model) const;
  inline multi_id factor_domain() const;
  inline multi_id factor_range() const;
  inline multi_id flat_range_product(multi_id multi2) const;
  inline multi_id flatten_range() const;
  inline multi_id from_range() const;
  inline space get_domain_space() const;
  inline id get_id(int pos) const;
  inline id_list get_id_list() const;
  inline space get_space() const;
  inline multi_id range_factor_domain() const;
  inline multi_id range_factor_range() const;
  inline multi_id range_product(multi_id multi2) const;
  inline multi_id range_splice(unsigned int pos, multi_id multi2) const;
  inline multi_id reset_user() const;
  inline multi_id set_id(int pos, id el) const;
  inline int size() const;
  typedef isl_multi_id* isl_ptr_t;
};

// declarations for isl::multi_pw_aff
inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

class multi_pw_aff {
  friend inline multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr);
  friend inline multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr);

protected:
  isl_multi_pw_aff *ptr = nullptr;

  inline explicit multi_pw_aff(__isl_take isl_multi_pw_aff *ptr);

public:
  inline /* implicit */ multi_pw_aff();
  inline /* implicit */ multi_pw_aff(const multi_pw_aff &obj);
  inline explicit multi_pw_aff(space space, pw_aff_list list);
  inline /* implicit */ multi_pw_aff(multi_aff ma);
  inline /* implicit */ multi_pw_aff(pw_aff pa);
  inline /* implicit */ multi_pw_aff(pw_multi_aff pma);
  inline explicit multi_pw_aff(ctx ctx, const std::string &str);
  inline multi_pw_aff &operator=(multi_pw_aff obj);
  inline ~multi_pw_aff();
  inline __isl_give isl_multi_pw_aff *copy() const &;
  inline __isl_give isl_multi_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_pw_aff *get() const;
  inline __isl_give isl_multi_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline multi_pw_aff add(multi_pw_aff multi2) const;
  inline multi_pw_aff align_params(space model) const;
  inline set domain() const;
  inline multi_pw_aff factor_domain() const;
  inline multi_pw_aff factor_range() const;
  inline multi_pw_aff flat_range_product(multi_pw_aff multi2) const;
  inline multi_pw_aff flatten_range() const;
  inline multi_pw_aff from_range() const;
  inline space get_domain_space() const;
  inline pw_aff get_pw_aff(int pos) const;
  inline pw_aff_list get_pw_aff_list() const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  static inline multi_pw_aff identity(space space);
  inline bool is_equal(const multi_pw_aff &mpa2) const;
  inline multi_pw_aff mod(multi_val mv) const;
  inline multi_pw_aff neg() const;
  inline multi_pw_aff product(multi_pw_aff multi2) const;
  inline multi_pw_aff pullback(multi_aff ma) const;
  inline multi_pw_aff pullback(pw_multi_aff pma) const;
  inline multi_pw_aff pullback(multi_pw_aff mpa2) const;
  inline multi_pw_aff range_factor_domain() const;
  inline multi_pw_aff range_factor_range() const;
  inline multi_pw_aff range_product(multi_pw_aff multi2) const;
  inline multi_pw_aff range_splice(unsigned int pos, multi_pw_aff multi2) const;
  inline multi_pw_aff reset_user() const;
  inline multi_pw_aff scale(val v) const;
  inline multi_pw_aff scale(multi_val mv) const;
  inline multi_pw_aff scale_down(val v) const;
  inline multi_pw_aff scale_down(multi_val mv) const;
  inline multi_pw_aff set_pw_aff(int pos, pw_aff el) const;
  inline multi_pw_aff set_range_tuple_id(id id) const;
  inline int size() const;
  inline multi_pw_aff splice(unsigned int in_pos, unsigned int out_pos, multi_pw_aff multi2) const;
  inline multi_pw_aff sub(multi_pw_aff multi2) const;
  static inline multi_pw_aff zero(space space);
  typedef isl_multi_pw_aff* isl_ptr_t;
};

// declarations for isl::multi_union_pw_aff
inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

class multi_union_pw_aff {
  friend inline multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr);
  friend inline multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr);

protected:
  isl_multi_union_pw_aff *ptr = nullptr;

  inline explicit multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr);

public:
  inline /* implicit */ multi_union_pw_aff();
  inline /* implicit */ multi_union_pw_aff(const multi_union_pw_aff &obj);
  inline /* implicit */ multi_union_pw_aff(union_pw_aff upa);
  inline /* implicit */ multi_union_pw_aff(multi_pw_aff mpa);
  inline explicit multi_union_pw_aff(union_set domain, multi_val mv);
  inline explicit multi_union_pw_aff(union_set domain, multi_aff ma);
  inline explicit multi_union_pw_aff(space space, union_pw_aff_list list);
  inline explicit multi_union_pw_aff(ctx ctx, const std::string &str);
  inline multi_union_pw_aff &operator=(multi_union_pw_aff obj);
  inline ~multi_union_pw_aff();
  inline __isl_give isl_multi_union_pw_aff *copy() const &;
  inline __isl_give isl_multi_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_multi_union_pw_aff *get() const;
  inline __isl_give isl_multi_union_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline multi_union_pw_aff add(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff align_params(space model) const;
  inline multi_union_pw_aff apply(multi_aff ma) const;
  inline multi_union_pw_aff apply(pw_multi_aff pma) const;
  inline union_set domain() const;
  inline multi_pw_aff extract_multi_pw_aff(space space) const;
  inline multi_union_pw_aff factor_domain() const;
  inline multi_union_pw_aff factor_range() const;
  inline multi_union_pw_aff flat_range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff flatten_range() const;
  inline multi_union_pw_aff floor() const;
  inline multi_union_pw_aff from_range() const;
  static inline multi_union_pw_aff from_union_map(union_map umap);
  inline space get_domain_space() const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  inline union_pw_aff get_union_pw_aff(int pos) const;
  inline union_pw_aff_list get_union_pw_aff_list() const;
  inline multi_union_pw_aff gist(union_set context) const;
  inline multi_union_pw_aff intersect_domain(union_set uset) const;
  inline multi_union_pw_aff intersect_params(set params) const;
  inline bool involves_param(const id &id) const;
  inline multi_val max_multi_val() const;
  inline multi_val min_multi_val() const;
  inline multi_union_pw_aff mod(multi_val mv) const;
  inline multi_union_pw_aff neg() const;
  inline multi_union_pw_aff pullback(union_pw_multi_aff upma) const;
  inline multi_union_pw_aff range_factor_domain() const;
  inline multi_union_pw_aff range_factor_range() const;
  inline multi_union_pw_aff range_product(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff range_splice(unsigned int pos, multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff reset_user() const;
  inline multi_union_pw_aff scale(val v) const;
  inline multi_union_pw_aff scale(multi_val mv) const;
  inline multi_union_pw_aff scale_down(val v) const;
  inline multi_union_pw_aff scale_down(multi_val mv) const;
  inline multi_union_pw_aff set_range_tuple_id(id id) const;
  inline multi_union_pw_aff set_union_pw_aff(int pos, union_pw_aff el) const;
  inline int size() const;
  inline multi_union_pw_aff sub(multi_union_pw_aff multi2) const;
  inline multi_union_pw_aff union_add(multi_union_pw_aff mupa2) const;
  static inline multi_union_pw_aff zero(space space);
  inline union_set zero_union_set() const;
  typedef isl_multi_union_pw_aff* isl_ptr_t;
};

// declarations for isl::multi_val
inline multi_val manage(__isl_take isl_multi_val *ptr);
inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

class multi_val {
  friend inline multi_val manage(__isl_take isl_multi_val *ptr);
  friend inline multi_val manage_copy(__isl_keep isl_multi_val *ptr);

protected:
  isl_multi_val *ptr = nullptr;

  inline explicit multi_val(__isl_take isl_multi_val *ptr);

public:
  inline /* implicit */ multi_val();
  inline /* implicit */ multi_val(const multi_val &obj);
  inline explicit multi_val(space space, val_list list);
  inline multi_val &operator=(multi_val obj);
  inline ~multi_val();
  inline __isl_give isl_multi_val *copy() const &;
  inline __isl_give isl_multi_val *copy() && = delete;
  inline __isl_keep isl_multi_val *get() const;
  inline __isl_give isl_multi_val *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline multi_val add(multi_val multi2) const;
  inline multi_val align_params(space model) const;
  inline multi_val factor_domain() const;
  inline multi_val factor_range() const;
  inline multi_val flat_range_product(multi_val multi2) const;
  inline multi_val flatten_range() const;
  inline multi_val from_range() const;
  inline space get_domain_space() const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  inline val get_val(int pos) const;
  inline val_list get_val_list() const;
  inline multi_val mod(multi_val mv) const;
  inline multi_val neg() const;
  inline multi_val product(multi_val multi2) const;
  inline multi_val range_factor_domain() const;
  inline multi_val range_factor_range() const;
  inline multi_val range_product(multi_val multi2) const;
  inline multi_val range_splice(unsigned int pos, multi_val multi2) const;
  inline multi_val reset_user() const;
  inline multi_val scale(val v) const;
  inline multi_val scale(multi_val mv) const;
  inline multi_val scale_down(val v) const;
  inline multi_val scale_down(multi_val mv) const;
  inline multi_val set_range_tuple_id(id id) const;
  inline multi_val set_val(int pos, val el) const;
  inline int size() const;
  inline multi_val splice(unsigned int in_pos, unsigned int out_pos, multi_val multi2) const;
  inline multi_val sub(multi_val multi2) const;
  static inline multi_val zero(space space);
  typedef isl_multi_val* isl_ptr_t;
};

// declarations for isl::point
inline point manage(__isl_take isl_point *ptr);
inline point manage_copy(__isl_keep isl_point *ptr);

class point {
  friend inline point manage(__isl_take isl_point *ptr);
  friend inline point manage_copy(__isl_keep isl_point *ptr);

protected:
  isl_point *ptr = nullptr;

  inline explicit point(__isl_take isl_point *ptr);

public:
  inline /* implicit */ point();
  inline /* implicit */ point(const point &obj);
  inline explicit point(space dim);
  inline point &operator=(point obj);
  inline ~point();
  inline __isl_give isl_point *copy() const &;
  inline __isl_give isl_point *copy() && = delete;
  inline __isl_keep isl_point *get() const;
  inline __isl_give isl_point *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline space get_space() const;
  inline bool is_void() const;
  typedef isl_point* isl_ptr_t;
};

// declarations for isl::pw_aff
inline pw_aff manage(__isl_take isl_pw_aff *ptr);
inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

class pw_aff {
  friend inline pw_aff manage(__isl_take isl_pw_aff *ptr);
  friend inline pw_aff manage_copy(__isl_keep isl_pw_aff *ptr);

protected:
  isl_pw_aff *ptr = nullptr;

  inline explicit pw_aff(__isl_take isl_pw_aff *ptr);

public:
  inline /* implicit */ pw_aff();
  inline /* implicit */ pw_aff(const pw_aff &obj);
  inline /* implicit */ pw_aff(aff aff);
  inline explicit pw_aff(local_space ls);
  inline explicit pw_aff(set domain, val v);
  inline explicit pw_aff(ctx ctx, const std::string &str);
  inline pw_aff &operator=(pw_aff obj);
  inline ~pw_aff();
  inline __isl_give isl_pw_aff *copy() const &;
  inline __isl_give isl_pw_aff *copy() && = delete;
  inline __isl_keep isl_pw_aff *get() const;
  inline __isl_give isl_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline pw_aff add(pw_aff pwaff2) const;
  inline pw_aff ceil() const;
  inline pw_aff cond(pw_aff pwaff_true, pw_aff pwaff_false) const;
  inline pw_aff div(pw_aff pa2) const;
  inline set domain() const;
  inline map eq_map(pw_aff pa2) const;
  inline set eq_set(pw_aff pwaff2) const;
  inline pw_aff floor() const;
  inline void foreach_piece(const std::function<void(set, aff)> &fn) const;
  inline set ge_set(pw_aff pwaff2) const;
  inline space get_space() const;
  inline map gt_map(pw_aff pa2) const;
  inline set gt_set(pw_aff pwaff2) const;
  inline pw_aff intersect_domain(set set) const;
  inline pw_aff intersect_params(set set) const;
  inline bool involves_nan() const;
  inline bool is_cst() const;
  inline bool is_equal(const pw_aff &pa2) const;
  inline set le_set(pw_aff pwaff2) const;
  inline map lt_map(pw_aff pa2) const;
  inline set lt_set(pw_aff pwaff2) const;
  inline pw_aff max(pw_aff pwaff2) const;
  inline pw_aff min(pw_aff pwaff2) const;
  inline pw_aff mod(val mod) const;
  inline pw_aff mul(pw_aff pwaff2) const;
  inline int n_piece() const;
  inline set ne_set(pw_aff pwaff2) const;
  inline pw_aff neg() const;
  inline set nonneg_set() const;
  inline set params() const;
  inline set pos_set() const;
  inline pw_aff project_domain_on_params() const;
  inline pw_aff pullback(multi_aff ma) const;
  inline pw_aff pullback(pw_multi_aff pma) const;
  inline pw_aff pullback(multi_pw_aff mpa) const;
  inline pw_aff scale(val v) const;
  inline pw_aff scale_down(val f) const;
  inline pw_aff sub(pw_aff pwaff2) const;
  inline pw_aff tdiv_q(pw_aff pa2) const;
  inline pw_aff tdiv_r(pw_aff pa2) const;
  inline pw_aff union_add(pw_aff pwaff2) const;
  inline set zero_set() const;
  typedef isl_pw_aff* isl_ptr_t;
};

// declarations for isl::pw_aff_list
inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

class pw_aff_list {
  friend inline pw_aff_list manage(__isl_take isl_pw_aff_list *ptr);
  friend inline pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr);

protected:
  isl_pw_aff_list *ptr = nullptr;

  inline explicit pw_aff_list(__isl_take isl_pw_aff_list *ptr);

public:
  inline /* implicit */ pw_aff_list();
  inline /* implicit */ pw_aff_list(const pw_aff_list &obj);
  inline explicit pw_aff_list(pw_aff el);
  inline explicit pw_aff_list(ctx ctx, int n);
  inline pw_aff_list &operator=(pw_aff_list obj);
  inline ~pw_aff_list();
  inline __isl_give isl_pw_aff_list *copy() const &;
  inline __isl_give isl_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_pw_aff_list *get() const;
  inline __isl_give isl_pw_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline pw_aff_list add(pw_aff el) const;
  inline pw_aff_list concat(pw_aff_list list2) const;
  inline pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(pw_aff)> &fn) const;
  inline pw_aff get_at(int index) const;
  inline pw_aff_list reverse() const;
  inline int size() const;
  typedef isl_pw_aff_list* isl_ptr_t;
};

// declarations for isl::pw_multi_aff
inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

class pw_multi_aff {
  friend inline pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr);
  friend inline pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr);

protected:
  isl_pw_multi_aff *ptr = nullptr;

  inline explicit pw_multi_aff(__isl_take isl_pw_multi_aff *ptr);

public:
  inline /* implicit */ pw_multi_aff();
  inline /* implicit */ pw_multi_aff(const pw_multi_aff &obj);
  inline /* implicit */ pw_multi_aff(multi_aff ma);
  inline /* implicit */ pw_multi_aff(pw_aff pa);
  inline explicit pw_multi_aff(set domain, multi_val mv);
  inline explicit pw_multi_aff(map map);
  inline explicit pw_multi_aff(ctx ctx, const std::string &str);
  inline pw_multi_aff &operator=(pw_multi_aff obj);
  inline ~pw_multi_aff();
  inline __isl_give isl_pw_multi_aff *copy() const &;
  inline __isl_give isl_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_pw_multi_aff *get() const;
  inline __isl_give isl_pw_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline pw_multi_aff add(pw_multi_aff pma2) const;
  inline set domain() const;
  inline pw_multi_aff flat_range_product(pw_multi_aff pma2) const;
  inline void foreach_piece(const std::function<void(set, multi_aff)> &fn) const;
  static inline pw_multi_aff from(multi_pw_aff mpa);
  inline pw_aff get_pw_aff(int pos) const;
  inline id get_range_tuple_id() const;
  inline space get_space() const;
  static inline pw_multi_aff identity(space space);
  inline bool is_equal(const pw_multi_aff &pma2) const;
  inline int n_piece() const;
  inline pw_multi_aff product(pw_multi_aff pma2) const;
  inline pw_multi_aff project_domain_on_params() const;
  inline pw_multi_aff pullback(multi_aff ma) const;
  inline pw_multi_aff pullback(pw_multi_aff pma2) const;
  inline pw_multi_aff range_factor_domain() const;
  inline pw_multi_aff range_factor_range() const;
  inline pw_multi_aff range_product(pw_multi_aff pma2) const;
  inline pw_multi_aff scale(val v) const;
  inline pw_multi_aff scale_down(val v) const;
  inline pw_multi_aff set_pw_aff(unsigned int pos, pw_aff pa) const;
  inline pw_multi_aff union_add(pw_multi_aff pma2) const;
  typedef isl_pw_multi_aff* isl_ptr_t;
};

// declarations for isl::schedule
inline schedule manage(__isl_take isl_schedule *ptr);
inline schedule manage_copy(__isl_keep isl_schedule *ptr);

class schedule {
  friend inline schedule manage(__isl_take isl_schedule *ptr);
  friend inline schedule manage_copy(__isl_keep isl_schedule *ptr);

protected:
  isl_schedule *ptr = nullptr;

  inline explicit schedule(__isl_take isl_schedule *ptr);

public:
  inline /* implicit */ schedule();
  inline /* implicit */ schedule(const schedule &obj);
  inline explicit schedule(ctx ctx, const std::string &str);
  inline schedule &operator=(schedule obj);
  inline ~schedule();
  inline __isl_give isl_schedule *copy() const &;
  inline __isl_give isl_schedule *copy() && = delete;
  inline __isl_keep isl_schedule *get() const;
  inline __isl_give isl_schedule *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  static inline schedule from_domain(union_set domain);
  inline union_set get_domain() const;
  inline union_map get_map() const;
  inline schedule_node get_root() const;
  inline schedule insert_partial_schedule(multi_union_pw_aff partial) const;
  inline bool plain_is_equal(const schedule &schedule2) const;
  inline schedule pullback(union_pw_multi_aff upma) const;
  inline schedule reset_user() const;
  inline schedule sequence(schedule schedule2) const;
  inline schedule set(schedule schedule2) const;
  typedef isl_schedule* isl_ptr_t;
};

// declarations for isl::schedule_constraints
inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

class schedule_constraints {
  friend inline schedule_constraints manage(__isl_take isl_schedule_constraints *ptr);
  friend inline schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr);

protected:
  isl_schedule_constraints *ptr = nullptr;

  inline explicit schedule_constraints(__isl_take isl_schedule_constraints *ptr);

public:
  inline /* implicit */ schedule_constraints();
  inline /* implicit */ schedule_constraints(const schedule_constraints &obj);
  inline explicit schedule_constraints(ctx ctx, const std::string &str);
  inline schedule_constraints &operator=(schedule_constraints obj);
  inline ~schedule_constraints();
  inline __isl_give isl_schedule_constraints *copy() const &;
  inline __isl_give isl_schedule_constraints *copy() && = delete;
  inline __isl_keep isl_schedule_constraints *get() const;
  inline __isl_give isl_schedule_constraints *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline schedule compute_schedule() const;
  inline union_map get_coincidence() const;
  inline union_map get_conditional_validity() const;
  inline union_map get_conditional_validity_condition() const;
  inline set get_context() const;
  inline union_set get_domain() const;
  inline multi_union_pw_aff get_prefix() const;
  inline union_map get_proximity() const;
  inline union_map get_validity() const;
  inline schedule_constraints intersect_domain(union_set domain) const;
  static inline schedule_constraints on_domain(union_set domain);
  inline schedule_constraints set_coincidence(union_map coincidence) const;
  inline schedule_constraints set_conditional_validity(union_map condition, union_map validity) const;
  inline schedule_constraints set_context(set context) const;
  inline schedule_constraints set_prefix(multi_union_pw_aff prefix) const;
  inline schedule_constraints set_proximity(union_map proximity) const;
  inline schedule_constraints set_validity(union_map validity) const;
  typedef isl_schedule_constraints* isl_ptr_t;
};

// declarations for isl::schedule_node
inline schedule_node manage(__isl_take isl_schedule_node *ptr);
inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

class schedule_node {
  friend inline schedule_node manage(__isl_take isl_schedule_node *ptr);
  friend inline schedule_node manage_copy(__isl_keep isl_schedule_node *ptr);

protected:
  isl_schedule_node *ptr = nullptr;

  inline explicit schedule_node(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node();
  inline /* implicit */ schedule_node(const schedule_node &obj);
  inline schedule_node &operator=(schedule_node obj);
  inline ~schedule_node();
  inline __isl_give isl_schedule_node *copy() const &;
  inline __isl_give isl_schedule_node *copy() && = delete;
  inline __isl_keep isl_schedule_node *get() const;
  inline __isl_give isl_schedule_node *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  template <class T> inline bool isa();
  template <class T> inline T as();
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline schedule_node ancestor(int generation) const;
  inline schedule_node child(int pos) const;
  inline schedule_node cut() const;
  inline schedule_node del() const;
  inline bool every_descendant(const std::function<bool(schedule_node)> &test) const;
  inline schedule_node first_child() const;
  inline void foreach_descendant_top_down(const std::function<bool(schedule_node)> &fn) const;
  static inline schedule_node from_domain(union_set domain);
  static inline schedule_node from_extension(union_map extension);
  inline int get_ancestor_child_position(const schedule_node &ancestor) const;
  inline schedule_node get_child(int pos) const;
  inline int get_child_position() const;
  inline union_set get_domain() const;
  inline multi_union_pw_aff get_prefix_schedule_multi_union_pw_aff() const;
  inline union_map get_prefix_schedule_relation() const;
  inline union_map get_prefix_schedule_union_map() const;
  inline union_pw_multi_aff get_prefix_schedule_union_pw_multi_aff() const;
  inline schedule get_schedule() const;
  inline int get_schedule_depth() const;
  inline schedule_node get_shared_ancestor(const schedule_node &node2) const;
  inline int get_tree_depth() const;
  inline union_set get_universe_domain() const;
  inline schedule_node graft_after(schedule_node graft) const;
  inline schedule_node graft_before(schedule_node graft) const;
  inline bool has_children() const;
  inline bool has_next_sibling() const;
  inline bool has_parent() const;
  inline bool has_previous_sibling() const;
  inline schedule_node insert_context(set context) const;
  inline schedule_node insert_filter(union_set filter) const;
  inline schedule_node insert_guard(set context) const;
  inline schedule_node insert_mark(id mark) const;
  inline schedule_node insert_partial_schedule(multi_union_pw_aff schedule) const;
  inline schedule_node insert_sequence(union_set_list filters) const;
  inline schedule_node insert_set(union_set_list filters) const;
  inline bool is_equal(const schedule_node &node2) const;
  inline bool is_subtree_anchored() const;
  inline schedule_node map_descendant_bottom_up(const std::function<schedule_node(schedule_node)> &fn) const;
  inline int n_children() const;
  inline schedule_node next_sibling() const;
  inline schedule_node order_after(union_set filter) const;
  inline schedule_node order_before(union_set filter) const;
  inline schedule_node parent() const;
  inline schedule_node previous_sibling() const;
  inline schedule_node root() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_band

class schedule_node_band : public schedule_node {
  friend bool schedule_node::isa<schedule_node_band>();
  friend schedule_node_band schedule_node::as<schedule_node_band>();
  static const auto type = isl_schedule_node_band;

protected:
  inline explicit schedule_node_band(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_band();
  inline /* implicit */ schedule_node_band(const schedule_node_band &obj);
  inline schedule_node_band &operator=(schedule_node_band obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_set get_ast_build_options() const;
  inline set get_ast_isolate_option() const;
  inline multi_union_pw_aff get_partial_schedule() const;
  inline union_map get_partial_schedule_union_map() const;
  inline bool get_permutable() const;
  inline space get_space() const;
  inline bool member_get_coincident(int pos) const;
  inline schedule_node_band member_set_coincident(int pos, int coincident) const;
  inline schedule_node_band mod(multi_val mv) const;
  inline unsigned int n_member() const;
  inline schedule_node_band scale(multi_val mv) const;
  inline schedule_node_band scale_down(multi_val mv) const;
  inline schedule_node_band set_ast_build_options(union_set options) const;
  inline schedule_node_band set_permutable(int permutable) const;
  inline schedule_node_band shift(multi_union_pw_aff shift) const;
  inline schedule_node_band split(int pos) const;
  inline schedule_node_band tile(multi_val sizes) const;
  inline schedule_node_band member_set_ast_loop_default(int pos) const;
  inline schedule_node_band member_set_ast_loop_atomic(int pos) const;
  inline schedule_node_band member_set_ast_loop_unroll(int pos) const;
  inline schedule_node_band member_set_ast_loop_separate(int pos) const;
  inline schedule_node_band member_set_isolate_ast_loop_default(int pos) const;
  inline schedule_node_band member_set_isolate_ast_loop_atomic(int pos) const;
  inline schedule_node_band member_set_isolate_ast_loop_unroll(int pos) const;
  inline schedule_node_band member_set_isolate_ast_loop_separate(int pos) const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_context

class schedule_node_context : public schedule_node {
  friend bool schedule_node::isa<schedule_node_context>();
  friend schedule_node_context schedule_node::as<schedule_node_context>();
  static const auto type = isl_schedule_node_context;

protected:
  inline explicit schedule_node_context(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_context();
  inline /* implicit */ schedule_node_context(const schedule_node_context &obj);
  inline schedule_node_context &operator=(schedule_node_context obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline set get_context() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_domain

class schedule_node_domain : public schedule_node {
  friend bool schedule_node::isa<schedule_node_domain>();
  friend schedule_node_domain schedule_node::as<schedule_node_domain>();
  static const auto type = isl_schedule_node_domain;

protected:
  inline explicit schedule_node_domain(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_domain();
  inline /* implicit */ schedule_node_domain(const schedule_node_domain &obj);
  inline schedule_node_domain &operator=(schedule_node_domain obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_set get_domain() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_expansion

class schedule_node_expansion : public schedule_node {
  friend bool schedule_node::isa<schedule_node_expansion>();
  friend schedule_node_expansion schedule_node::as<schedule_node_expansion>();
  static const auto type = isl_schedule_node_expansion;

protected:
  inline explicit schedule_node_expansion(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_expansion();
  inline /* implicit */ schedule_node_expansion(const schedule_node_expansion &obj);
  inline schedule_node_expansion &operator=(schedule_node_expansion obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_pw_multi_aff get_contraction() const;
  inline union_map get_expansion() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_extension

class schedule_node_extension : public schedule_node {
  friend bool schedule_node::isa<schedule_node_extension>();
  friend schedule_node_extension schedule_node::as<schedule_node_extension>();
  static const auto type = isl_schedule_node_extension;

protected:
  inline explicit schedule_node_extension(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_extension();
  inline /* implicit */ schedule_node_extension(const schedule_node_extension &obj);
  inline schedule_node_extension &operator=(schedule_node_extension obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_map get_extension() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_filter

class schedule_node_filter : public schedule_node {
  friend bool schedule_node::isa<schedule_node_filter>();
  friend schedule_node_filter schedule_node::as<schedule_node_filter>();
  static const auto type = isl_schedule_node_filter;

protected:
  inline explicit schedule_node_filter(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_filter();
  inline /* implicit */ schedule_node_filter(const schedule_node_filter &obj);
  inline schedule_node_filter &operator=(schedule_node_filter obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_set get_filter() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_guard

class schedule_node_guard : public schedule_node {
  friend bool schedule_node::isa<schedule_node_guard>();
  friend schedule_node_guard schedule_node::as<schedule_node_guard>();
  static const auto type = isl_schedule_node_guard;

protected:
  inline explicit schedule_node_guard(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_guard();
  inline /* implicit */ schedule_node_guard(const schedule_node_guard &obj);
  inline schedule_node_guard &operator=(schedule_node_guard obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline set get_guard() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_leaf

class schedule_node_leaf : public schedule_node {
  friend bool schedule_node::isa<schedule_node_leaf>();
  friend schedule_node_leaf schedule_node::as<schedule_node_leaf>();
  static const auto type = isl_schedule_node_leaf;

protected:
  inline explicit schedule_node_leaf(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_leaf();
  inline /* implicit */ schedule_node_leaf(const schedule_node_leaf &obj);
  inline schedule_node_leaf &operator=(schedule_node_leaf obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_mark

class schedule_node_mark : public schedule_node {
  friend bool schedule_node::isa<schedule_node_mark>();
  friend schedule_node_mark schedule_node::as<schedule_node_mark>();
  static const auto type = isl_schedule_node_mark;

protected:
  inline explicit schedule_node_mark(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_mark();
  inline /* implicit */ schedule_node_mark(const schedule_node_mark &obj);
  inline schedule_node_mark &operator=(schedule_node_mark obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline id get_id() const;
  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_sequence

class schedule_node_sequence : public schedule_node {
  friend bool schedule_node::isa<schedule_node_sequence>();
  friend schedule_node_sequence schedule_node::as<schedule_node_sequence>();
  static const auto type = isl_schedule_node_sequence;

protected:
  inline explicit schedule_node_sequence(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_sequence();
  inline /* implicit */ schedule_node_sequence(const schedule_node_sequence &obj);
  inline schedule_node_sequence &operator=(schedule_node_sequence obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::schedule_node_set

class schedule_node_set : public schedule_node {
  friend bool schedule_node::isa<schedule_node_set>();
  friend schedule_node_set schedule_node::as<schedule_node_set>();
  static const auto type = isl_schedule_node_set;

protected:
  inline explicit schedule_node_set(__isl_take isl_schedule_node *ptr);

public:
  inline /* implicit */ schedule_node_set();
  inline /* implicit */ schedule_node_set(const schedule_node_set &obj);
  inline schedule_node_set &operator=(schedule_node_set obj);
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  typedef isl_schedule_node* isl_ptr_t;
};

// declarations for isl::set
inline set manage(__isl_take isl_set *ptr);
inline set manage_copy(__isl_keep isl_set *ptr);

class set {
  friend inline set manage(__isl_take isl_set *ptr);
  friend inline set manage_copy(__isl_keep isl_set *ptr);

protected:
  isl_set *ptr = nullptr;

  inline explicit set(__isl_take isl_set *ptr);

public:
  inline /* implicit */ set();
  inline /* implicit */ set(const set &obj);
  inline /* implicit */ set(point pnt);
  inline explicit set(ctx ctx, const std::string &str);
  inline /* implicit */ set(basic_set bset);
  inline set &operator=(set obj);
  inline ~set();
  inline __isl_give isl_set *copy() const &;
  inline __isl_give isl_set *copy() && = delete;
  inline __isl_keep isl_set *get() const;
  inline __isl_give isl_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline basic_set affine_hull() const;
  inline set align_params(space model) const;
  inline set apply(map map) const;
  inline set coalesce() const;
  inline set complement() const;
  inline set compute_divs() const;
  inline set detect_equalities() const;
  inline pw_aff dim_max(int pos) const;
  inline pw_aff dim_min(int pos) const;
  static inline set empty(space space);
  inline set flatten() const;
  inline map flatten_map() const;
  inline void foreach_basic_set(const std::function<void(basic_set)> &fn) const;
  static inline set from(multi_aff ma);
  inline set from_params() const;
  static inline set from_union_set(union_set uset);
  inline basic_set_list get_basic_set_list() const;
  inline space get_space() const;
  inline val get_stride(int pos) const;
  inline id get_tuple_id() const;
  inline std::string get_tuple_name() const;
  inline set gist(set context) const;
  inline bool has_tuple_id() const;
  inline bool has_tuple_name() const;
  inline map identity() const;
  inline set intersect(set set2) const;
  inline set intersect_params(set params) const;
  inline bool involves_param(const id &id) const;
  inline bool is_disjoint(const set &set2) const;
  inline bool is_empty() const;
  inline bool is_equal(const set &set2) const;
  inline bool is_singleton() const;
  inline bool is_strict_subset(const set &set2) const;
  inline bool is_subset(const set &set2) const;
  inline bool is_wrapping() const;
  inline set lexmax() const;
  inline set lexmin() const;
  inline val max_val(const aff &obj) const;
  inline val min_val(const aff &obj) const;
  inline int n_basic_set() const;
  inline unsigned int n_dim() const;
  inline unsigned int n_param() const;
  static inline set nat_universe(space dim);
  inline set params() const;
  inline bool plain_is_universe() const;
  inline basic_set polyhedral_hull() const;
  inline set preimage_multi_aff(multi_aff ma) const;
  inline set product(set set2) const;
  inline set reset_tuple_id() const;
  inline basic_set sample() const;
  inline point sample_point() const;
  inline set set_tuple_id(id id) const;
  inline set set_tuple_name(const std::string &s) const;
  inline basic_set simple_hull() const;
  inline set subtract(set set2) const;
  inline set unbind_params(multi_id tuple) const;
  inline map unbind_params_insert_domain(multi_id domain) const;
  inline set unite(set set2) const;
  static inline set universe(space space);
  inline basic_set unshifted_simple_hull() const;
  inline map unwrap() const;
  inline map wrapped_domain_map() const;
  typedef isl_set* isl_ptr_t;
};

// declarations for isl::set_list
inline set_list manage(__isl_take isl_set_list *ptr);
inline set_list manage_copy(__isl_keep isl_set_list *ptr);

class set_list {
  friend inline set_list manage(__isl_take isl_set_list *ptr);
  friend inline set_list manage_copy(__isl_keep isl_set_list *ptr);

protected:
  isl_set_list *ptr = nullptr;

  inline explicit set_list(__isl_take isl_set_list *ptr);

public:
  inline /* implicit */ set_list();
  inline /* implicit */ set_list(const set_list &obj);
  inline explicit set_list(set el);
  inline explicit set_list(ctx ctx, int n);
  inline set_list &operator=(set_list obj);
  inline ~set_list();
  inline __isl_give isl_set_list *copy() const &;
  inline __isl_give isl_set_list *copy() && = delete;
  inline __isl_keep isl_set_list *get() const;
  inline __isl_give isl_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline set_list add(set el) const;
  inline set_list concat(set_list list2) const;
  inline set_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(set)> &fn) const;
  inline set get_at(int index) const;
  inline set_list reverse() const;
  inline int size() const;
  typedef isl_set_list* isl_ptr_t;
};

// declarations for isl::space
inline space manage(__isl_take isl_space *ptr);
inline space manage_copy(__isl_keep isl_space *ptr);

class space {
  friend inline space manage(__isl_take isl_space *ptr);
  friend inline space manage_copy(__isl_keep isl_space *ptr);

protected:
  isl_space *ptr = nullptr;

  inline explicit space(__isl_take isl_space *ptr);

public:
  inline /* implicit */ space();
  inline /* implicit */ space(const space &obj);
  inline explicit space(ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out);
  inline explicit space(ctx ctx, unsigned int nparam, unsigned int dim);
  inline explicit space(ctx ctx, unsigned int nparam);
  inline space &operator=(space obj);
  inline ~space();
  inline __isl_give isl_space *copy() const &;
  inline __isl_give isl_space *copy() && = delete;
  inline __isl_keep isl_space *get() const;
  inline __isl_give isl_space *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline space add_named_tuple_id_ui(id tuple_id, unsigned int dim) const;
  inline space add_param(id id) const;
  inline space add_unnamed_tuple_ui(unsigned int dim) const;
  inline space align_params(space dim2) const;
  inline bool can_curry() const;
  inline bool can_uncurry() const;
  inline space curry() const;
  inline space domain() const;
  inline space domain_map() const;
  inline space domain_product(space right) const;
  inline space from_domain() const;
  inline space from_range() const;
  inline id get_map_range_tuple_id() const;
  inline bool has_equal_params(const space &space2) const;
  inline bool has_equal_tuples(const space &space2) const;
  inline bool has_param(const id &id) const;
  inline bool is_equal(const space &space2) const;
  inline bool is_params() const;
  inline bool is_set() const;
  inline bool is_wrapping() const;
  inline space map_from_domain_and_range(space range) const;
  inline space map_from_set() const;
  inline space params() const;
  inline space product(space right) const;
  inline space range() const;
  inline space range_map() const;
  inline space range_product(space right) const;
  inline space set_from_params() const;
  inline space set_set_tuple_id(id id) const;
  inline space uncurry() const;
  inline space unwrap() const;
  inline space wrap() const;
  typedef isl_space* isl_ptr_t;
};

// declarations for isl::stride_info
inline stride_info manage(__isl_take isl_stride_info *ptr);
inline stride_info manage_copy(__isl_keep isl_stride_info *ptr);

class stride_info {
  friend inline stride_info manage(__isl_take isl_stride_info *ptr);
  friend inline stride_info manage_copy(__isl_keep isl_stride_info *ptr);

protected:
  isl_stride_info *ptr = nullptr;

  inline explicit stride_info(__isl_take isl_stride_info *ptr);

public:
  inline /* implicit */ stride_info();
  inline /* implicit */ stride_info(const stride_info &obj);
  inline stride_info &operator=(stride_info obj);
  inline ~stride_info();
  inline __isl_give isl_stride_info *copy() const &;
  inline __isl_give isl_stride_info *copy() && = delete;
  inline __isl_keep isl_stride_info *get() const;
  inline __isl_give isl_stride_info *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline aff get_offset() const;
  inline val get_stride() const;
  typedef isl_stride_info* isl_ptr_t;
};

// declarations for isl::union_access_info
inline union_access_info manage(__isl_take isl_union_access_info *ptr);
inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

class union_access_info {
  friend inline union_access_info manage(__isl_take isl_union_access_info *ptr);
  friend inline union_access_info manage_copy(__isl_keep isl_union_access_info *ptr);

protected:
  isl_union_access_info *ptr = nullptr;

  inline explicit union_access_info(__isl_take isl_union_access_info *ptr);

public:
  inline /* implicit */ union_access_info();
  inline /* implicit */ union_access_info(const union_access_info &obj);
  inline explicit union_access_info(union_map sink);
  inline union_access_info &operator=(union_access_info obj);
  inline ~union_access_info();
  inline __isl_give isl_union_access_info *copy() const &;
  inline __isl_give isl_union_access_info *copy() && = delete;
  inline __isl_keep isl_union_access_info *get() const;
  inline __isl_give isl_union_access_info *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_flow compute_flow() const;
  inline union_access_info set_kill(union_map kill) const;
  inline union_access_info set_may_source(union_map may_source) const;
  inline union_access_info set_must_source(union_map must_source) const;
  inline union_access_info set_schedule(schedule schedule) const;
  inline union_access_info set_schedule_map(union_map schedule_map) const;
  typedef isl_union_access_info* isl_ptr_t;
};

// declarations for isl::union_flow
inline union_flow manage(__isl_take isl_union_flow *ptr);
inline union_flow manage_copy(__isl_keep isl_union_flow *ptr);

class union_flow {
  friend inline union_flow manage(__isl_take isl_union_flow *ptr);
  friend inline union_flow manage_copy(__isl_keep isl_union_flow *ptr);

protected:
  isl_union_flow *ptr = nullptr;

  inline explicit union_flow(__isl_take isl_union_flow *ptr);

public:
  inline /* implicit */ union_flow();
  inline /* implicit */ union_flow(const union_flow &obj);
  inline union_flow &operator=(union_flow obj);
  inline ~union_flow();
  inline __isl_give isl_union_flow *copy() const &;
  inline __isl_give isl_union_flow *copy() && = delete;
  inline __isl_keep isl_union_flow *get() const;
  inline __isl_give isl_union_flow *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_map get_full_may_dependence() const;
  inline union_map get_full_must_dependence() const;
  inline union_map get_may_dependence() const;
  inline union_map get_may_no_source() const;
  inline union_map get_must_dependence() const;
  inline union_map get_must_no_source() const;
  typedef isl_union_flow* isl_ptr_t;
};

// declarations for isl::union_map
inline union_map manage(__isl_take isl_union_map *ptr);
inline union_map manage_copy(__isl_keep isl_union_map *ptr);

class union_map {
  friend inline union_map manage(__isl_take isl_union_map *ptr);
  friend inline union_map manage_copy(__isl_keep isl_union_map *ptr);

protected:
  isl_union_map *ptr = nullptr;

  inline explicit union_map(__isl_take isl_union_map *ptr);

public:
  inline /* implicit */ union_map();
  inline /* implicit */ union_map(const union_map &obj);
  inline /* implicit */ union_map(basic_map bmap);
  inline /* implicit */ union_map(map map);
  inline explicit union_map(ctx ctx, const std::string &str);
  inline union_map &operator=(union_map obj);
  inline ~union_map();
  inline __isl_give isl_union_map *copy() const &;
  inline __isl_give isl_union_map *copy() && = delete;
  inline __isl_keep isl_union_map *get() const;
  inline __isl_give isl_union_map *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_map add_map(map map) const;
  inline union_map affine_hull() const;
  inline union_map apply_domain(union_map umap2) const;
  inline union_map apply_range(union_map umap2) const;
  inline union_map coalesce() const;
  inline union_map compute_divs() const;
  inline union_map curry() const;
  inline union_set deltas() const;
  inline union_map detect_equalities() const;
  inline union_set domain() const;
  inline union_map domain_factor_domain() const;
  inline union_map domain_factor_range() const;
  inline union_map domain_map() const;
  inline union_pw_multi_aff domain_map_union_pw_multi_aff() const;
  inline union_map domain_product(union_map umap2) const;
  static inline union_map empty(space space);
  inline union_map eq_at(multi_union_pw_aff mupa) const;
  inline map extract_map(space dim) const;
  inline union_map factor_domain() const;
  inline union_map factor_range() const;
  inline union_map fixed_power(val exp) const;
  inline union_map flat_range_product(union_map umap2) const;
  inline void foreach_map(const std::function<void(map)> &fn) const;
  static inline union_map from(union_pw_multi_aff upma);
  static inline union_map from(multi_union_pw_aff mupa);
  static inline union_map from_domain(union_set uset);
  static inline union_map from_domain_and_range(union_set domain, union_set range);
  static inline union_map from_range(union_set uset);
  inline map_list get_map_list() const;
  inline space get_space() const;
  inline union_map gist(union_map context) const;
  inline union_map gist_domain(union_set uset) const;
  inline union_map gist_params(set set) const;
  inline union_map gist_range(union_set uset) const;
  inline union_map intersect(union_map umap2) const;
  inline union_map intersect_domain(union_set uset) const;
  inline union_map intersect_params(set set) const;
  inline union_map intersect_range(union_set uset) const;
  inline bool is_bijective() const;
  inline bool is_empty() const;
  inline bool is_equal(const union_map &umap2) const;
  inline bool is_injective() const;
  inline bool is_single_valued() const;
  inline bool is_strict_subset(const union_map &umap2) const;
  inline bool is_subset(const union_map &umap2) const;
  inline union_map lex_gt_at(multi_union_pw_aff mupa) const;
  inline union_map lex_lt_at(multi_union_pw_aff mupa) const;
  inline union_map lexmax() const;
  inline union_map lexmin() const;
  inline int n_map() const;
  inline union_map polyhedral_hull() const;
  inline union_map preimage_range_multi_aff(multi_aff ma) const;
  inline union_map product(union_map umap2) const;
  inline union_map project_out_all_params() const;
  inline union_set range() const;
  inline union_map range_factor_domain() const;
  inline union_map range_factor_range() const;
  inline union_map range_map() const;
  inline union_map range_product(union_map umap2) const;
  inline union_map reverse() const;
  inline union_map subtract(union_map umap2) const;
  inline union_map subtract_domain(union_set dom) const;
  inline union_map subtract_range(union_set dom) const;
  inline union_map uncurry() const;
  inline union_map unite(union_map umap2) const;
  inline union_map universe() const;
  inline union_set wrap() const;
  inline union_map zip() const;
  typedef isl_union_map* isl_ptr_t;
};

// declarations for isl::union_pw_aff
inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

class union_pw_aff {
  friend inline union_pw_aff manage(__isl_take isl_union_pw_aff *ptr);
  friend inline union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr);

protected:
  isl_union_pw_aff *ptr = nullptr;

  inline explicit union_pw_aff(__isl_take isl_union_pw_aff *ptr);

public:
  inline /* implicit */ union_pw_aff();
  inline /* implicit */ union_pw_aff(const union_pw_aff &obj);
  inline /* implicit */ union_pw_aff(pw_aff pa);
  inline explicit union_pw_aff(union_set domain, val v);
  inline explicit union_pw_aff(union_set domain, aff aff);
  inline explicit union_pw_aff(ctx ctx, const std::string &str);
  inline union_pw_aff &operator=(union_pw_aff obj);
  inline ~union_pw_aff();
  inline __isl_give isl_union_pw_aff *copy() const &;
  inline __isl_give isl_union_pw_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_aff *get() const;
  inline __isl_give isl_union_pw_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_pw_aff add(union_pw_aff upa2) const;
  inline union_set domain() const;
  static inline union_pw_aff empty(space space);
  inline pw_aff extract_on_domain(space space) const;
  inline pw_aff extract_pw_aff(space space) const;
  inline union_pw_aff floor() const;
  inline void foreach_pw_aff(const std::function<void(pw_aff)> &fn) const;
  inline pw_aff_list get_pw_aff_list() const;
  inline space get_space() const;
  inline union_pw_aff intersect_domain(union_set uset) const;
  inline bool involves_param(const id &id) const;
  inline val max_val() const;
  inline val min_val() const;
  inline union_pw_aff mod(val f) const;
  inline int n_pw_aff() const;
  static inline union_pw_aff param_on_domain(union_set domain, id id);
  inline bool plain_is_equal(const union_pw_aff &upa2) const;
  inline union_pw_aff pullback(union_pw_multi_aff upma) const;
  inline union_pw_aff scale(val v) const;
  inline union_pw_aff scale_down(val v) const;
  inline union_pw_aff sub(union_pw_aff upa2) const;
  inline union_pw_aff union_add(union_pw_aff upa2) const;
  inline union_set zero_union_set() const;
  typedef isl_union_pw_aff* isl_ptr_t;
};

// declarations for isl::union_pw_aff_list
inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

class union_pw_aff_list {
  friend inline union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr);
  friend inline union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr);

protected:
  isl_union_pw_aff_list *ptr = nullptr;

  inline explicit union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr);

public:
  inline /* implicit */ union_pw_aff_list();
  inline /* implicit */ union_pw_aff_list(const union_pw_aff_list &obj);
  inline explicit union_pw_aff_list(union_pw_aff el);
  inline explicit union_pw_aff_list(ctx ctx, int n);
  inline union_pw_aff_list &operator=(union_pw_aff_list obj);
  inline ~union_pw_aff_list();
  inline __isl_give isl_union_pw_aff_list *copy() const &;
  inline __isl_give isl_union_pw_aff_list *copy() && = delete;
  inline __isl_keep isl_union_pw_aff_list *get() const;
  inline __isl_give isl_union_pw_aff_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline union_pw_aff_list add(union_pw_aff el) const;
  inline union_pw_aff_list concat(union_pw_aff_list list2) const;
  inline union_pw_aff_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(union_pw_aff)> &fn) const;
  inline union_pw_aff get_at(int index) const;
  inline union_pw_aff_list reverse() const;
  inline int size() const;
  typedef isl_union_pw_aff_list* isl_ptr_t;
};

// declarations for isl::union_pw_multi_aff
inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

class union_pw_multi_aff {
  friend inline union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr);
  friend inline union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr);

protected:
  isl_union_pw_multi_aff *ptr = nullptr;

  inline explicit union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr);

public:
  inline /* implicit */ union_pw_multi_aff();
  inline /* implicit */ union_pw_multi_aff(const union_pw_multi_aff &obj);
  inline /* implicit */ union_pw_multi_aff(pw_multi_aff pma);
  inline explicit union_pw_multi_aff(union_set domain, multi_val mv);
  inline explicit union_pw_multi_aff(ctx ctx, const std::string &str);
  inline /* implicit */ union_pw_multi_aff(union_pw_aff upa);
  inline union_pw_multi_aff &operator=(union_pw_multi_aff obj);
  inline ~union_pw_multi_aff();
  inline __isl_give isl_union_pw_multi_aff *copy() const &;
  inline __isl_give isl_union_pw_multi_aff *copy() && = delete;
  inline __isl_keep isl_union_pw_multi_aff *get() const;
  inline __isl_give isl_union_pw_multi_aff *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_pw_multi_aff add(union_pw_multi_aff upma2) const;
  inline union_set domain() const;
  inline pw_multi_aff extract_pw_multi_aff(space space) const;
  inline union_pw_multi_aff flat_range_product(union_pw_multi_aff upma2) const;
  inline void foreach_pw_multi_aff(const std::function<void(pw_multi_aff)> &fn) const;
  static inline union_pw_multi_aff from(union_map umap);
  static inline union_pw_multi_aff from_multi_union_pw_aff(multi_union_pw_aff mupa);
  inline space get_space() const;
  inline union_pw_aff get_union_pw_aff(int pos) const;
  inline int n_pw_multi_aff() const;
  inline union_pw_multi_aff pullback(union_pw_multi_aff upma2) const;
  inline union_pw_multi_aff scale(val val) const;
  inline union_pw_multi_aff scale_down(val val) const;
  inline union_pw_multi_aff union_add(union_pw_multi_aff upma2) const;
  typedef isl_union_pw_multi_aff* isl_ptr_t;
};

// declarations for isl::union_set
inline union_set manage(__isl_take isl_union_set *ptr);
inline union_set manage_copy(__isl_keep isl_union_set *ptr);

class union_set {
  friend inline union_set manage(__isl_take isl_union_set *ptr);
  friend inline union_set manage_copy(__isl_keep isl_union_set *ptr);

protected:
  isl_union_set *ptr = nullptr;

  inline explicit union_set(__isl_take isl_union_set *ptr);

public:
  inline /* implicit */ union_set();
  inline /* implicit */ union_set(const union_set &obj);
  inline /* implicit */ union_set(basic_set bset);
  inline /* implicit */ union_set(set set);
  inline /* implicit */ union_set(point pnt);
  inline explicit union_set(ctx ctx, const std::string &str);
  inline union_set &operator=(union_set obj);
  inline ~union_set();
  inline __isl_give isl_union_set *copy() const &;
  inline __isl_give isl_union_set *copy() && = delete;
  inline __isl_keep isl_union_set *get() const;
  inline __isl_give isl_union_set *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline union_set add_set(set set) const;
  inline union_set affine_hull() const;
  inline union_set apply(union_map umap) const;
  inline union_set coalesce() const;
  inline union_set compute_divs() const;
  inline union_set detect_equalities() const;
  static inline union_set empty(space space);
  inline bool every_set(const std::function<bool(set)> &test) const;
  inline set extract_set(space dim) const;
  inline void foreach_point(const std::function<void(point)> &fn) const;
  inline void foreach_set(const std::function<void(set)> &fn) const;
  inline set_list get_set_list() const;
  inline space get_space() const;
  inline union_set gist(union_set context) const;
  inline union_set gist_params(set set) const;
  inline union_map identity() const;
  inline union_set intersect(union_set uset2) const;
  inline union_set intersect_params(set set) const;
  inline bool involves_param(const id &id) const;
  inline bool is_disjoint(const union_set &uset2) const;
  inline bool is_empty() const;
  inline bool is_equal(const union_set &uset2) const;
  inline bool is_params() const;
  inline bool is_strict_subset(const union_set &uset2) const;
  inline bool is_subset(const union_set &uset2) const;
  inline union_set lexmax() const;
  inline union_set lexmin() const;
  inline int n_set() const;
  inline set params() const;
  inline union_set polyhedral_hull() const;
  inline union_set preimage(multi_aff ma) const;
  inline union_set preimage(pw_multi_aff pma) const;
  inline union_set preimage(union_pw_multi_aff upma) const;
  inline point sample_point() const;
  inline union_set subtract(union_set uset2) const;
  inline union_set unite(union_set uset2) const;
  inline union_set universe() const;
  inline union_map unwrap() const;
  inline union_map wrapped_domain_map() const;
  typedef isl_union_set* isl_ptr_t;
};

// declarations for isl::union_set_list
inline union_set_list manage(__isl_take isl_union_set_list *ptr);
inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

class union_set_list {
  friend inline union_set_list manage(__isl_take isl_union_set_list *ptr);
  friend inline union_set_list manage_copy(__isl_keep isl_union_set_list *ptr);

protected:
  isl_union_set_list *ptr = nullptr;

  inline explicit union_set_list(__isl_take isl_union_set_list *ptr);

public:
  inline /* implicit */ union_set_list();
  inline /* implicit */ union_set_list(const union_set_list &obj);
  inline explicit union_set_list(union_set el);
  inline explicit union_set_list(ctx ctx, int n);
  inline union_set_list &operator=(union_set_list obj);
  inline ~union_set_list();
  inline __isl_give isl_union_set_list *copy() const &;
  inline __isl_give isl_union_set_list *copy() && = delete;
  inline __isl_keep isl_union_set_list *get() const;
  inline __isl_give isl_union_set_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline union_set_list add(union_set el) const;
  inline union_set_list concat(union_set_list list2) const;
  inline union_set_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(union_set)> &fn) const;
  inline union_set get_at(int index) const;
  inline union_set_list reverse() const;
  inline int size() const;
  typedef isl_union_set_list* isl_ptr_t;
};

// declarations for isl::val
inline val manage(__isl_take isl_val *ptr);
inline val manage_copy(__isl_keep isl_val *ptr);

class val {
  friend inline val manage(__isl_take isl_val *ptr);
  friend inline val manage_copy(__isl_keep isl_val *ptr);

protected:
  isl_val *ptr = nullptr;

  inline explicit val(__isl_take isl_val *ptr);

public:
  inline /* implicit */ val();
  inline /* implicit */ val(const val &obj);
  inline explicit val(ctx ctx, const std::string &str);
  inline explicit val(ctx ctx, long i);
  inline val &operator=(val obj);
  inline ~val();
  inline __isl_give isl_val *copy() const &;
  inline __isl_give isl_val *copy() && = delete;
  inline __isl_keep isl_val *get() const;
  inline __isl_give isl_val *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;
  inline std::string to_str() const;

  inline val abs() const;
  inline bool abs_eq(const val &v2) const;
  inline val add(val v2) const;
  inline val ceil() const;
  inline int cmp_si(long i) const;
  inline val div(val v2) const;
  inline bool eq(const val &v2) const;
  inline val floor() const;
  inline val gcd(val v2) const;
  inline bool ge(const val &v2) const;
  inline long get_den_si() const;
  inline long get_num_si() const;
  inline bool gt(const val &v2) const;
  static inline val infty(ctx ctx);
  inline val inv() const;
  inline bool is_divisible_by(const val &v2) const;
  inline bool is_infty() const;
  inline bool is_int() const;
  inline bool is_nan() const;
  inline bool is_neg() const;
  inline bool is_neginfty() const;
  inline bool is_negone() const;
  inline bool is_nonneg() const;
  inline bool is_nonpos() const;
  inline bool is_one() const;
  inline bool is_pos() const;
  inline bool is_rat() const;
  inline bool is_zero() const;
  inline bool le(const val &v2) const;
  inline bool lt(const val &v2) const;
  inline val max(val v2) const;
  inline val min(val v2) const;
  inline val mod(val v2) const;
  inline val mul(val v2) const;
  static inline val nan(ctx ctx);
  inline bool ne(const val &v2) const;
  inline val neg() const;
  static inline val neginfty(ctx ctx);
  static inline val negone(ctx ctx);
  static inline val one(ctx ctx);
  inline int sgn() const;
  inline val sub(val v2) const;
  inline val trunc() const;
  static inline val zero(ctx ctx);
  typedef isl_val* isl_ptr_t;
};

// declarations for isl::val_list
inline val_list manage(__isl_take isl_val_list *ptr);
inline val_list manage_copy(__isl_keep isl_val_list *ptr);

class val_list {
  friend inline val_list manage(__isl_take isl_val_list *ptr);
  friend inline val_list manage_copy(__isl_keep isl_val_list *ptr);

protected:
  isl_val_list *ptr = nullptr;

  inline explicit val_list(__isl_take isl_val_list *ptr);

public:
  inline /* implicit */ val_list();
  inline /* implicit */ val_list(const val_list &obj);
  inline explicit val_list(val el);
  inline explicit val_list(ctx ctx, int n);
  inline val_list &operator=(val_list obj);
  inline ~val_list();
  inline __isl_give isl_val_list *copy() const &;
  inline __isl_give isl_val_list *copy() && = delete;
  inline __isl_keep isl_val_list *get() const;
  inline __isl_give isl_val_list *release();
  inline bool is_null() const;
  inline explicit operator bool() const;
  inline ctx get_ctx() const;

  inline val_list add(val el) const;
  inline val_list concat(val_list list2) const;
  inline val_list drop(unsigned int first, unsigned int n) const;
  inline void foreach(const std::function<void(val)> &fn) const;
  inline val get_at(int index) const;
  inline val_list reverse() const;
  inline int size() const;
  typedef isl_val_list* isl_ptr_t;
};

// implementations for isl::aff
aff manage(__isl_take isl_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return aff(ptr);
}
aff manage_copy(__isl_keep isl_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return aff(ptr);
}

aff::aff()
    : ptr(nullptr) {}

aff::aff(const aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

aff::aff(__isl_take isl_aff *ptr)
    : ptr(ptr) {}

aff::aff(local_space ls)
{
  if (ls.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ls.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_zero_on_domain(ls.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
aff::aff(local_space ls, val val)
{
  if (ls.is_null() || val.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ls.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_val_on_domain(ls.release(), val.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
aff::aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

aff &aff::operator=(aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

aff::~aff() {
  if (ptr)
    isl_aff_free(ptr);
}

__isl_give isl_aff *aff::copy() const & {
  return isl_aff_copy(ptr);
}

__isl_keep isl_aff *aff::get() const {
  return ptr;
}

__isl_give isl_aff *aff::release() {
  isl_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool aff::is_null() const {
  return ptr == nullptr;
}
aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const aff& C) {
  os << C.to_str();
  return os;
}


std::string aff::to_str() const {
  char *Tmp = isl_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx aff::get_ctx() const {
  return ctx(isl_aff_get_ctx(ptr));
}

aff aff::add(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_add(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::add_constant(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_add_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::add_constant_si(int v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_add_constant_si(copy(), v);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_ceil(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::div(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_div(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::eq_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_eq_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val aff::eval(point pnt) const
{
  if (!ptr || pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_eval(copy(), pnt.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::ge_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_ge_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val aff::get_constant_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_get_constant_val(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val aff::get_denominator_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_get_denominator_val(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::get_div(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_get_div(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space aff::get_local_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_get_local_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::gt_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_gt_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::le_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_le_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::lt_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_lt_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::mod(val mod) const
{
  if (!ptr || mod.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_mod_val(copy(), mod.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::mul(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_mul(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set aff::ne_set(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_ne_set(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::param_on_domain_space(space space, id id)
{
  if (space.is_null() || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_param_on_domain_space_id(space.release(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool aff::plain_is_equal(const aff &aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_plain_is_equal(get(), aff2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

aff aff::project_domain_on_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_project_domain_on_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::pullback(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::scale_down_ui(unsigned int f) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_scale_down_ui(copy(), f);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::set_constant_si(int v) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_set_constant_si(copy(), v);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::set_constant_val(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_set_constant_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::sub(aff aff2) const
{
  if (!ptr || aff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_sub(copy(), aff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::unbind_params_insert_domain(multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff aff::zero_on_domain(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_zero_on_domain_space(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::aff_list
aff_list manage(__isl_take isl_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return aff_list(ptr);
}
aff_list manage_copy(__isl_keep isl_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return aff_list(ptr);
}

aff_list::aff_list()
    : ptr(nullptr) {}

aff_list::aff_list(const aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

aff_list::aff_list(__isl_take isl_aff_list *ptr)
    : ptr(ptr) {}

aff_list::aff_list(aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_from_aff(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
aff_list::aff_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

aff_list &aff_list::operator=(aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

aff_list::~aff_list() {
  if (ptr)
    isl_aff_list_free(ptr);
}

__isl_give isl_aff_list *aff_list::copy() const & {
  return isl_aff_list_copy(ptr);
}

__isl_keep isl_aff_list *aff_list::get() const {
  return ptr;
}

__isl_give isl_aff_list *aff_list::release() {
  isl_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool aff_list::is_null() const {
  return ptr == nullptr;
}
aff_list::operator bool() const
{
  return !is_null();
}



ctx aff_list::get_ctx() const {
  return ctx(isl_aff_list_get_ctx(ptr));
}

aff_list aff_list::add(aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff_list aff_list::concat(aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff_list aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void aff_list::foreach(const std::function<void(aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

aff aff_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff_list aff_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_aff_list_size(get());
  return res;
}

// implementations for isl::ast_build
ast_build manage(__isl_take isl_ast_build *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_build(ptr);
}
ast_build manage_copy(__isl_keep isl_ast_build *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_build_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_ast_build_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return ast_build(ptr);
}

ast_build::ast_build()
    : ptr(nullptr) {}

ast_build::ast_build(const ast_build &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_build_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  copy_callbacks(obj);
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

ast_build::ast_build(__isl_take isl_ast_build *ptr)
    : ptr(ptr) {}

ast_build::ast_build(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_alloc(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

ast_build &ast_build::operator=(ast_build obj) {
  std::swap(this->ptr, obj.ptr);
  copy_callbacks(obj);
  return *this;
}

ast_build::~ast_build() {
  if (ptr)
    isl_ast_build_free(ptr);
}

__isl_give isl_ast_build *ast_build::copy() const & {
  return isl_ast_build_copy(ptr);
}

__isl_keep isl_ast_build *ast_build::get() const {
  return ptr;
}

__isl_give isl_ast_build *ast_build::release() {
  if (at_each_domain_data)
    exception::throw_invalid("cannot release object with persistent callbacks", __FILE__, __LINE__);
  isl_ast_build *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_build::is_null() const {
  return ptr == nullptr;
}
ast_build::operator bool() const
{
  return !is_null();
}



ctx ast_build::get_ctx() const {
  return ctx(isl_ast_build_get_ctx(ptr));
}

ast_build &ast_build::copy_callbacks(const ast_build &obj)
{
  at_each_domain_data = obj.at_each_domain_data;
  return *this;
}

isl_ast_node *ast_build::at_each_domain(isl_ast_node *arg_0, isl_ast_build *arg_1, void *arg_2)
{
  auto *data = static_cast<struct at_each_domain_data *>(arg_2);
    ISL_CPP_TRY {
    auto ret = (data->func)(manage(arg_0), manage_copy(arg_1));
    return ret.release();
  } ISL_CPP_CATCH_ALL {
    data->eptr = std::current_exception();
    return NULL;
  }
}

void ast_build::set_at_each_domain_data(const std::function<ast_node(ast_node, ast_build)> &fn)
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_build_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  at_each_domain_data = std::make_shared<struct at_each_domain_data>();
  at_each_domain_data->func = fn;
  ptr = isl_ast_build_set_at_each_domain(ptr, &at_each_domain, at_each_domain_data.get());
  if (!ptr)
    exception::throw_last_error(ctx);
}

ast_build ast_build::set_at_each_domain(const std::function<ast_node(ast_node, ast_build)> &fn) const
{
  auto copy = *this;
  copy.set_at_each_domain_data(fn);
  return copy;
}

ast_expr ast_build::access_from(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_access_from_pw_multi_aff(get(), pma.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_build::access_from(multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_access_from_multi_pw_aff(get(), mpa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_build::ast_from_schedule(union_map schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_ast_from_schedule(get(), schedule.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_build::call_from(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_call_from_pw_multi_aff(get(), pma.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_build::call_from(multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_call_from_multi_pw_aff(get(), mpa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_build::expr_from(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_expr_from_set(get(), set.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_build::expr_from(pw_aff pa) const
{
  if (!ptr || pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_expr_from_pw_aff(get(), pa.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_build ast_build::from_context(set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = set.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_from_context(set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map ast_build::get_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_get_schedule(get());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space ast_build::get_schedule_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_get_schedule_space(get());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_build::node_from(schedule schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_node_from_schedule(get(), schedule.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_build::node_from_schedule_map(union_map schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_node_from_schedule_map(get(), schedule.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_build ast_build::set_iterators(id_list iterators) const
{
  if (!ptr || iterators.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_build_set_iterators(copy(), iterators.release());
  if (at_each_domain_data && at_each_domain_data->eptr) {
    std::exception_ptr eptr = at_each_domain_data->eptr;
    at_each_domain_data->eptr = nullptr;
    std::rethrow_exception(eptr);
  }
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).copy_callbacks(*this);
}

// implementations for isl::ast_expr
ast_expr manage(__isl_take isl_ast_expr *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_expr(ptr);
}
ast_expr manage_copy(__isl_keep isl_ast_expr *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_expr_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_ast_expr_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return ast_expr(ptr);
}

ast_expr::ast_expr()
    : ptr(nullptr) {}

ast_expr::ast_expr(const ast_expr &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_expr_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

ast_expr::ast_expr(__isl_take isl_ast_expr *ptr)
    : ptr(ptr) {}


ast_expr &ast_expr::operator=(ast_expr obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_expr::~ast_expr() {
  if (ptr)
    isl_ast_expr_free(ptr);
}

__isl_give isl_ast_expr *ast_expr::copy() const & {
  return isl_ast_expr_copy(ptr);
}

__isl_keep isl_ast_expr *ast_expr::get() const {
  return ptr;
}

__isl_give isl_ast_expr *ast_expr::release() {
  isl_ast_expr *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_expr::is_null() const {
  return ptr == nullptr;
}
ast_expr::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const ast_expr& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_expr& C1, const ast_expr& C2) {
  return C1.is_equal(C2);
}


std::string ast_expr::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


template <class T>
bool ast_expr::isa()
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_expr_get_type(get()) == T::type;
}
template <class T>
T ast_expr::as()
{
  return isa<T>() ? T(copy()) : T();
}

ctx ast_expr::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

bool ast_expr::is_equal(const ast_expr &expr2) const
{
  if (!ptr || expr2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_is_equal(get(), expr2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

ast_expr ast_expr::set_op_arg(int pos, ast_expr arg) const
{
  if (!ptr || arg.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_set_op_arg(copy(), pos, arg.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

std::string ast_expr::to_C_str() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

// implementations for isl::ast_expr_id

ast_expr_id::ast_expr_id()
    : ast_expr() {}

ast_expr_id::ast_expr_id(const ast_expr_id &obj)
    : ast_expr(obj)
{
}

ast_expr_id::ast_expr_id(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}


ast_expr_id &ast_expr_id::operator=(ast_expr_id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_expr_id& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_expr_id& C1, const ast_expr_id& C2) {
  return C1.is_equal(C2);
}


std::string ast_expr_id::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_expr_id::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

id ast_expr_id::get_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_id_get_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::ast_expr_int

ast_expr_int::ast_expr_int()
    : ast_expr() {}

ast_expr_int::ast_expr_int(const ast_expr_int &obj)
    : ast_expr(obj)
{
}

ast_expr_int::ast_expr_int(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}


ast_expr_int &ast_expr_int::operator=(ast_expr_int obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_expr_int& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_expr_int& C1, const ast_expr_int& C2) {
  return C1.is_equal(C2);
}


std::string ast_expr_int::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_expr_int::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

val ast_expr_int::get_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_int_get_val(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::ast_expr_op

ast_expr_op::ast_expr_op()
    : ast_expr() {}

ast_expr_op::ast_expr_op(const ast_expr_op &obj)
    : ast_expr(obj)
{
}

ast_expr_op::ast_expr_op(__isl_take isl_ast_expr *ptr)
    : ast_expr(ptr) {}


ast_expr_op &ast_expr_op::operator=(ast_expr_op obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_expr_op& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_expr_op& C1, const ast_expr_op& C2) {
  return C1.is_equal(C2);
}


std::string ast_expr_op::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


template <class T>
bool ast_expr_op::isa()
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_expr_op_get_type(get()) == T::type;
}
template <class T>
T ast_expr_op::as()
{
  return isa<T>() ? T(copy()) : T();
}

ctx ast_expr_op::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}

ast_expr ast_expr_op::get_arg(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_op_get_arg(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int ast_expr_op::get_n_arg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_expr_op_get_n_arg(get());
  return res;
}

// implementations for isl::ast_node
ast_node manage(__isl_take isl_ast_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_node(ptr);
}
ast_node manage_copy(__isl_keep isl_ast_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_node_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_ast_node_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return ast_node(ptr);
}

ast_node::ast_node()
    : ptr(nullptr) {}

ast_node::ast_node(const ast_node &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_node_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

ast_node::ast_node(__isl_take isl_ast_node *ptr)
    : ptr(ptr) {}


ast_node &ast_node::operator=(ast_node obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_node::~ast_node() {
  if (ptr)
    isl_ast_node_free(ptr);
}

__isl_give isl_ast_node *ast_node::copy() const & {
  return isl_ast_node_copy(ptr);
}

__isl_keep isl_ast_node *ast_node::get() const {
  return ptr;
}

__isl_give isl_ast_node *ast_node::release() {
  isl_ast_node *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_node::is_null() const {
  return ptr == nullptr;
}
ast_node::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const ast_node& C) {
  os << C.to_str();
  return os;
}


std::string ast_node::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


template <class T>
bool ast_node::isa()
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_ast_node_get_type(get()) == T::type;
}
template <class T>
T ast_node::as()
{
  return isa<T>() ? T(copy()) : T();
}

ctx ast_node::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

id ast_node::get_annotation() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_get_annotation(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_node::set_annotation(id annotation) const
{
  if (!ptr || annotation.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_set_annotation(copy(), annotation.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

std::string ast_node::to_C_str() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_to_C_str(get());
  std::string tmp(res);
  free(res);
  return tmp;
}

// implementations for isl::ast_node_block

ast_node_block::ast_node_block()
    : ast_node() {}

ast_node_block::ast_node_block(const ast_node_block &obj)
    : ast_node(obj)
{
}

ast_node_block::ast_node_block(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}


ast_node_block &ast_node_block::operator=(ast_node_block obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_node_block& C) {
  os << C.to_str();
  return os;
}


std::string ast_node_block::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_node_block::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

ast_node_list ast_node_block::get_children() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_block_get_children(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::ast_node_for

ast_node_for::ast_node_for()
    : ast_node() {}

ast_node_for::ast_node_for(const ast_node_for &obj)
    : ast_node(obj)
{
}

ast_node_for::ast_node_for(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}


ast_node_for &ast_node_for::operator=(ast_node_for obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_node_for& C) {
  os << C.to_str();
  return os;
}


std::string ast_node_for::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_node_for::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

ast_node ast_node_for::get_body() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_get_body(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_node_for::get_cond() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_get_cond(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_node_for::get_inc() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_get_inc(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_node_for::get_init() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_get_init(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_expr ast_node_for::get_iterator() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_get_iterator(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool ast_node_for::is_coincident() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_is_coincident(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool ast_node_for::is_degenerate() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_for_is_degenerate(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

// implementations for isl::ast_node_if

ast_node_if::ast_node_if()
    : ast_node() {}

ast_node_if::ast_node_if(const ast_node_if &obj)
    : ast_node(obj)
{
}

ast_node_if::ast_node_if(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}


ast_node_if &ast_node_if::operator=(ast_node_if obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_node_if& C) {
  os << C.to_str();
  return os;
}


std::string ast_node_if::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_node_if::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

ast_expr ast_node_if::get_cond() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_if_get_cond(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_node_if::get_else() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_if_get_else(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_node_if::get_then() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_if_get_then(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool ast_node_if::has_else() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_if_has_else(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

// implementations for isl::ast_node_list
ast_node_list manage(__isl_take isl_ast_node_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return ast_node_list(ptr);
}
ast_node_list manage_copy(__isl_keep isl_ast_node_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_node_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_ast_node_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return ast_node_list(ptr);
}

ast_node_list::ast_node_list()
    : ptr(nullptr) {}

ast_node_list::ast_node_list(const ast_node_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_ast_node_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

ast_node_list::ast_node_list(__isl_take isl_ast_node_list *ptr)
    : ptr(ptr) {}

ast_node_list::ast_node_list(ast_node el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_from_ast_node(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
ast_node_list::ast_node_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

ast_node_list &ast_node_list::operator=(ast_node_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

ast_node_list::~ast_node_list() {
  if (ptr)
    isl_ast_node_list_free(ptr);
}

__isl_give isl_ast_node_list *ast_node_list::copy() const & {
  return isl_ast_node_list_copy(ptr);
}

__isl_keep isl_ast_node_list *ast_node_list::get() const {
  return ptr;
}

__isl_give isl_ast_node_list *ast_node_list::release() {
  isl_ast_node_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool ast_node_list::is_null() const {
  return ptr == nullptr;
}
ast_node_list::operator bool() const
{
  return !is_null();
}



ctx ast_node_list::get_ctx() const {
  return ctx(isl_ast_node_list_get_ctx(ptr));
}

ast_node_list ast_node_list::add(ast_node el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node_list ast_node_list::concat(ast_node_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node_list ast_node_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void ast_node_list::foreach(const std::function<void(ast_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(ast_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_ast_node *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_ast_node_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

ast_node ast_node_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node_list ast_node_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int ast_node_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_list_size(get());
  return res;
}

// implementations for isl::ast_node_mark

ast_node_mark::ast_node_mark()
    : ast_node() {}

ast_node_mark::ast_node_mark(const ast_node_mark &obj)
    : ast_node(obj)
{
}

ast_node_mark::ast_node_mark(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}


ast_node_mark &ast_node_mark::operator=(ast_node_mark obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_node_mark& C) {
  os << C.to_str();
  return os;
}


std::string ast_node_mark::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_node_mark::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

id ast_node_mark::get_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_mark_get_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

ast_node ast_node_mark::get_node() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_mark_get_node(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::ast_node_user

ast_node_user::ast_node_user()
    : ast_node() {}

ast_node_user::ast_node_user(const ast_node_user &obj)
    : ast_node(obj)
{
}

ast_node_user::ast_node_user(__isl_take isl_ast_node *ptr)
    : ast_node(ptr) {}


ast_node_user &ast_node_user::operator=(ast_node_user obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_node_user& C) {
  os << C.to_str();
  return os;
}


std::string ast_node_user::to_str() const {
  char *Tmp = isl_ast_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_node_user::get_ctx() const {
  return ctx(isl_ast_node_get_ctx(ptr));
}

ast_expr ast_node_user::get_expr() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_ast_node_user_get_expr(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::ast_op_access

ast_op_access::ast_op_access()
    : ast_expr_op() {}

ast_op_access::ast_op_access(const ast_op_access &obj)
    : ast_expr_op(obj)
{
}

ast_op_access::ast_op_access(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_access &ast_op_access::operator=(ast_op_access obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_access& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_access& C1, const ast_op_access& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_access::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_access::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_add

ast_op_add::ast_op_add()
    : ast_expr_op() {}

ast_op_add::ast_op_add(const ast_op_add &obj)
    : ast_expr_op(obj)
{
}

ast_op_add::ast_op_add(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_add &ast_op_add::operator=(ast_op_add obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_add& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_add& C1, const ast_op_add& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_add::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_add::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_address_of

ast_op_address_of::ast_op_address_of()
    : ast_expr_op() {}

ast_op_address_of::ast_op_address_of(const ast_op_address_of &obj)
    : ast_expr_op(obj)
{
}

ast_op_address_of::ast_op_address_of(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_address_of &ast_op_address_of::operator=(ast_op_address_of obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_address_of& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_address_of& C1, const ast_op_address_of& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_address_of::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_address_of::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_and

ast_op_and::ast_op_and()
    : ast_expr_op() {}

ast_op_and::ast_op_and(const ast_op_and &obj)
    : ast_expr_op(obj)
{
}

ast_op_and::ast_op_and(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_and &ast_op_and::operator=(ast_op_and obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_and& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_and& C1, const ast_op_and& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_and::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_and::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_and_then

ast_op_and_then::ast_op_and_then()
    : ast_expr_op() {}

ast_op_and_then::ast_op_and_then(const ast_op_and_then &obj)
    : ast_expr_op(obj)
{
}

ast_op_and_then::ast_op_and_then(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_and_then &ast_op_and_then::operator=(ast_op_and_then obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_and_then& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_and_then& C1, const ast_op_and_then& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_and_then::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_and_then::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_call

ast_op_call::ast_op_call()
    : ast_expr_op() {}

ast_op_call::ast_op_call(const ast_op_call &obj)
    : ast_expr_op(obj)
{
}

ast_op_call::ast_op_call(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_call &ast_op_call::operator=(ast_op_call obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_call& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_call& C1, const ast_op_call& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_call::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_call::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_cond

ast_op_cond::ast_op_cond()
    : ast_expr_op() {}

ast_op_cond::ast_op_cond(const ast_op_cond &obj)
    : ast_expr_op(obj)
{
}

ast_op_cond::ast_op_cond(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_cond &ast_op_cond::operator=(ast_op_cond obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_cond& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_cond& C1, const ast_op_cond& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_cond::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_cond::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_div

ast_op_div::ast_op_div()
    : ast_expr_op() {}

ast_op_div::ast_op_div(const ast_op_div &obj)
    : ast_expr_op(obj)
{
}

ast_op_div::ast_op_div(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_div &ast_op_div::operator=(ast_op_div obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_div& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_div& C1, const ast_op_div& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_div::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_div::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_eq

ast_op_eq::ast_op_eq()
    : ast_expr_op() {}

ast_op_eq::ast_op_eq(const ast_op_eq &obj)
    : ast_expr_op(obj)
{
}

ast_op_eq::ast_op_eq(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_eq &ast_op_eq::operator=(ast_op_eq obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_eq& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_eq& C1, const ast_op_eq& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_eq::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_eq::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_fdiv_q

ast_op_fdiv_q::ast_op_fdiv_q()
    : ast_expr_op() {}

ast_op_fdiv_q::ast_op_fdiv_q(const ast_op_fdiv_q &obj)
    : ast_expr_op(obj)
{
}

ast_op_fdiv_q::ast_op_fdiv_q(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_fdiv_q &ast_op_fdiv_q::operator=(ast_op_fdiv_q obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_fdiv_q& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_fdiv_q& C1, const ast_op_fdiv_q& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_fdiv_q::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_fdiv_q::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_ge

ast_op_ge::ast_op_ge()
    : ast_expr_op() {}

ast_op_ge::ast_op_ge(const ast_op_ge &obj)
    : ast_expr_op(obj)
{
}

ast_op_ge::ast_op_ge(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_ge &ast_op_ge::operator=(ast_op_ge obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_ge& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_ge& C1, const ast_op_ge& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_ge::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_ge::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_gt

ast_op_gt::ast_op_gt()
    : ast_expr_op() {}

ast_op_gt::ast_op_gt(const ast_op_gt &obj)
    : ast_expr_op(obj)
{
}

ast_op_gt::ast_op_gt(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_gt &ast_op_gt::operator=(ast_op_gt obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_gt& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_gt& C1, const ast_op_gt& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_gt::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_gt::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_le

ast_op_le::ast_op_le()
    : ast_expr_op() {}

ast_op_le::ast_op_le(const ast_op_le &obj)
    : ast_expr_op(obj)
{
}

ast_op_le::ast_op_le(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_le &ast_op_le::operator=(ast_op_le obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_le& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_le& C1, const ast_op_le& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_le::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_le::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_lt

ast_op_lt::ast_op_lt()
    : ast_expr_op() {}

ast_op_lt::ast_op_lt(const ast_op_lt &obj)
    : ast_expr_op(obj)
{
}

ast_op_lt::ast_op_lt(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_lt &ast_op_lt::operator=(ast_op_lt obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_lt& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_lt& C1, const ast_op_lt& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_lt::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_lt::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_max

ast_op_max::ast_op_max()
    : ast_expr_op() {}

ast_op_max::ast_op_max(const ast_op_max &obj)
    : ast_expr_op(obj)
{
}

ast_op_max::ast_op_max(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_max &ast_op_max::operator=(ast_op_max obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_max& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_max& C1, const ast_op_max& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_max::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_max::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_member

ast_op_member::ast_op_member()
    : ast_expr_op() {}

ast_op_member::ast_op_member(const ast_op_member &obj)
    : ast_expr_op(obj)
{
}

ast_op_member::ast_op_member(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_member &ast_op_member::operator=(ast_op_member obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_member& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_member& C1, const ast_op_member& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_member::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_member::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_min

ast_op_min::ast_op_min()
    : ast_expr_op() {}

ast_op_min::ast_op_min(const ast_op_min &obj)
    : ast_expr_op(obj)
{
}

ast_op_min::ast_op_min(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_min &ast_op_min::operator=(ast_op_min obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_min& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_min& C1, const ast_op_min& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_min::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_min::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_minus

ast_op_minus::ast_op_minus()
    : ast_expr_op() {}

ast_op_minus::ast_op_minus(const ast_op_minus &obj)
    : ast_expr_op(obj)
{
}

ast_op_minus::ast_op_minus(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_minus &ast_op_minus::operator=(ast_op_minus obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_minus& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_minus& C1, const ast_op_minus& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_minus::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_minus::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_mul

ast_op_mul::ast_op_mul()
    : ast_expr_op() {}

ast_op_mul::ast_op_mul(const ast_op_mul &obj)
    : ast_expr_op(obj)
{
}

ast_op_mul::ast_op_mul(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_mul &ast_op_mul::operator=(ast_op_mul obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_mul& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_mul& C1, const ast_op_mul& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_mul::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_mul::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_or

ast_op_or::ast_op_or()
    : ast_expr_op() {}

ast_op_or::ast_op_or(const ast_op_or &obj)
    : ast_expr_op(obj)
{
}

ast_op_or::ast_op_or(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_or &ast_op_or::operator=(ast_op_or obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_or& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_or& C1, const ast_op_or& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_or::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_or::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_or_else

ast_op_or_else::ast_op_or_else()
    : ast_expr_op() {}

ast_op_or_else::ast_op_or_else(const ast_op_or_else &obj)
    : ast_expr_op(obj)
{
}

ast_op_or_else::ast_op_or_else(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_or_else &ast_op_or_else::operator=(ast_op_or_else obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_or_else& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_or_else& C1, const ast_op_or_else& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_or_else::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_or_else::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_pdiv_q

ast_op_pdiv_q::ast_op_pdiv_q()
    : ast_expr_op() {}

ast_op_pdiv_q::ast_op_pdiv_q(const ast_op_pdiv_q &obj)
    : ast_expr_op(obj)
{
}

ast_op_pdiv_q::ast_op_pdiv_q(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_pdiv_q &ast_op_pdiv_q::operator=(ast_op_pdiv_q obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_pdiv_q& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_pdiv_q& C1, const ast_op_pdiv_q& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_pdiv_q::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_pdiv_q::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_pdiv_r

ast_op_pdiv_r::ast_op_pdiv_r()
    : ast_expr_op() {}

ast_op_pdiv_r::ast_op_pdiv_r(const ast_op_pdiv_r &obj)
    : ast_expr_op(obj)
{
}

ast_op_pdiv_r::ast_op_pdiv_r(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_pdiv_r &ast_op_pdiv_r::operator=(ast_op_pdiv_r obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_pdiv_r& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_pdiv_r& C1, const ast_op_pdiv_r& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_pdiv_r::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_pdiv_r::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_select

ast_op_select::ast_op_select()
    : ast_expr_op() {}

ast_op_select::ast_op_select(const ast_op_select &obj)
    : ast_expr_op(obj)
{
}

ast_op_select::ast_op_select(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_select &ast_op_select::operator=(ast_op_select obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_select& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_select& C1, const ast_op_select& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_select::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_select::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_sub

ast_op_sub::ast_op_sub()
    : ast_expr_op() {}

ast_op_sub::ast_op_sub(const ast_op_sub &obj)
    : ast_expr_op(obj)
{
}

ast_op_sub::ast_op_sub(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_sub &ast_op_sub::operator=(ast_op_sub obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_sub& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_sub& C1, const ast_op_sub& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_sub::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_sub::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::ast_op_zdiv_r

ast_op_zdiv_r::ast_op_zdiv_r()
    : ast_expr_op() {}

ast_op_zdiv_r::ast_op_zdiv_r(const ast_op_zdiv_r &obj)
    : ast_expr_op(obj)
{
}

ast_op_zdiv_r::ast_op_zdiv_r(__isl_take isl_ast_expr *ptr)
    : ast_expr_op(ptr) {}


ast_op_zdiv_r &ast_op_zdiv_r::operator=(ast_op_zdiv_r obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const ast_op_zdiv_r& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const ast_op_zdiv_r& C1, const ast_op_zdiv_r& C2) {
  return C1.is_equal(C2);
}


std::string ast_op_zdiv_r::to_str() const {
  char *Tmp = isl_ast_expr_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx ast_op_zdiv_r::get_ctx() const {
  return ctx(isl_ast_expr_get_ctx(ptr));
}


// implementations for isl::basic_map
basic_map manage(__isl_take isl_basic_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_map(ptr);
}
basic_map manage_copy(__isl_keep isl_basic_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_basic_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return basic_map(ptr);
}

basic_map::basic_map()
    : ptr(nullptr) {}

basic_map::basic_map(const basic_map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

basic_map::basic_map(__isl_take isl_basic_map *ptr)
    : ptr(ptr) {}

basic_map::basic_map(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_map::basic_map(basic_set domain, basic_set range)
{
  if (domain.is_null() || range.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_from_domain_and_range(domain.release(), range.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_map::basic_map(aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = aff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_map::basic_map(multi_aff maff)
{
  if (maff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = maff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_from_multi_aff(maff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

basic_map &basic_map::operator=(basic_map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_map::~basic_map() {
  if (ptr)
    isl_basic_map_free(ptr);
}

__isl_give isl_basic_map *basic_map::copy() const & {
  return isl_basic_map_copy(ptr);
}

__isl_keep isl_basic_map *basic_map::get() const {
  return ptr;
}

__isl_give isl_basic_map *basic_map::release() {
  isl_basic_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_map::is_null() const {
  return ptr == nullptr;
}
basic_map::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const basic_map& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const basic_map& C1, const basic_map& C2) {
  return C1.is_equal(C2);
}


std::string basic_map::to_str() const {
  char *Tmp = isl_basic_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx basic_map::get_ctx() const {
  return ctx(isl_basic_map_get_ctx(ptr));
}

basic_map basic_map::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::apply_domain(basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_apply_domain(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::apply_range(basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_apply_range(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool basic_map::can_curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_can_curry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_map::can_uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_can_uncurry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

basic_map basic_map::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_curry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_deltas(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_map::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_flatten(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::from_domain(basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_from_domain(bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::from_range(basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_from_range(bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space basic_map::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::gist(basic_map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::intersect(basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_intersect(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::intersect_domain(basic_set bset) const
{
  if (!ptr || bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_intersect_domain(copy(), bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::intersect_range(basic_set bset) const
{
  if (!ptr || bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_intersect_range(copy(), bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool basic_map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_map::is_equal(const basic_map &bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_is_equal(get(), bmap2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_map::is_subset(const basic_map &bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_is_subset(get(), bmap2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

map basic_map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map basic_map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_sample(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_uncurry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map basic_map::unite(basic_map bmap2) const
{
  if (!ptr || bmap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_union(copy(), bmap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_map::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_wrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::basic_map_list
basic_map_list manage(__isl_take isl_basic_map_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_map_list(ptr);
}
basic_map_list manage_copy(__isl_keep isl_basic_map_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_map_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_basic_map_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return basic_map_list(ptr);
}

basic_map_list::basic_map_list()
    : ptr(nullptr) {}

basic_map_list::basic_map_list(const basic_map_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_map_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

basic_map_list::basic_map_list(__isl_take isl_basic_map_list *ptr)
    : ptr(ptr) {}

basic_map_list::basic_map_list(basic_map el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_from_basic_map(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_map_list::basic_map_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

basic_map_list &basic_map_list::operator=(basic_map_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_map_list::~basic_map_list() {
  if (ptr)
    isl_basic_map_list_free(ptr);
}

__isl_give isl_basic_map_list *basic_map_list::copy() const & {
  return isl_basic_map_list_copy(ptr);
}

__isl_keep isl_basic_map_list *basic_map_list::get() const {
  return ptr;
}

__isl_give isl_basic_map_list *basic_map_list::release() {
  isl_basic_map_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_map_list::is_null() const {
  return ptr == nullptr;
}
basic_map_list::operator bool() const
{
  return !is_null();
}



ctx basic_map_list::get_ctx() const {
  return ctx(isl_basic_map_list_get_ctx(ptr));
}

basic_map_list basic_map_list::add(basic_map el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map_list basic_map_list::concat(basic_map_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map_list basic_map_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void basic_map_list::foreach(const std::function<void(basic_map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(basic_map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_basic_map_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

basic_map basic_map_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_map_list::intersect() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_intersect(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map_list basic_map_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int basic_map_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_map_list_size(get());
  return res;
}

// implementations for isl::basic_set
basic_set manage(__isl_take isl_basic_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_set(ptr);
}
basic_set manage_copy(__isl_keep isl_basic_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_basic_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return basic_set(ptr);
}

basic_set::basic_set()
    : ptr(nullptr) {}

basic_set::basic_set(const basic_set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

basic_set::basic_set(__isl_take isl_basic_set *ptr)
    : ptr(ptr) {}

basic_set::basic_set(point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pnt.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_set::basic_set(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

basic_set &basic_set::operator=(basic_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_set::~basic_set() {
  if (ptr)
    isl_basic_set_free(ptr);
}

__isl_give isl_basic_set *basic_set::copy() const & {
  return isl_basic_set_copy(ptr);
}

__isl_keep isl_basic_set *basic_set::get() const {
  return ptr;
}

__isl_give isl_basic_set *basic_set::release() {
  isl_basic_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_set::is_null() const {
  return ptr == nullptr;
}
basic_set::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const basic_set& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const basic_set& C1, const basic_set& C2) {
  return C1.is_equal(C2);
}


std::string basic_set::to_str() const {
  char *Tmp = isl_basic_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx basic_set::get_ctx() const {
  return ctx(isl_basic_set_get_ctx(ptr));
}

basic_set basic_set::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::apply(basic_map bmap) const
{
  if (!ptr || bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_apply(copy(), bmap.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set basic_set::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_compute_divs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val basic_set::dim_max_val(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_dim_max_val(copy(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_flatten(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::from_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_from_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space basic_set::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::gist(basic_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::intersect(basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_intersect(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::intersect_params(basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_intersect_params(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool basic_set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_set::is_equal(const basic_set &bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_is_equal(get(), bset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_set::is_subset(const basic_set &bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_is_subset(get(), bset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_set::is_universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_is_universe(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool basic_set::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

set basic_set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set basic_set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val basic_set::max_val(const aff &obj) const
{
  if (!ptr || obj.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_max_val(get(), obj.get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

unsigned int basic_set::n_dim() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_n_dim(get());
  return res;
}

unsigned int basic_set::n_param() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_n_param(get());
  return res;
}

basic_set basic_set::nat_universe(space dim)
{
  if (dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = dim.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_nat_universe(dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool basic_set::plain_is_universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_plain_is_universe(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

basic_set basic_set::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_sample(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

point basic_set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::set_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_set_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set basic_set::unite(basic_set bset2) const
{
  if (!ptr || bset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_union(copy(), bset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set basic_set::universe(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_universe(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map basic_set::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_unwrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::basic_set_list
basic_set_list manage(__isl_take isl_basic_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return basic_set_list(ptr);
}
basic_set_list manage_copy(__isl_keep isl_basic_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_set_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_basic_set_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return basic_set_list(ptr);
}

basic_set_list::basic_set_list()
    : ptr(nullptr) {}

basic_set_list::basic_set_list(const basic_set_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_basic_set_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

basic_set_list::basic_set_list(__isl_take isl_basic_set_list *ptr)
    : ptr(ptr) {}

basic_set_list::basic_set_list(basic_set el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_from_basic_set(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
basic_set_list::basic_set_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

basic_set_list &basic_set_list::operator=(basic_set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

basic_set_list::~basic_set_list() {
  if (ptr)
    isl_basic_set_list_free(ptr);
}

__isl_give isl_basic_set_list *basic_set_list::copy() const & {
  return isl_basic_set_list_copy(ptr);
}

__isl_keep isl_basic_set_list *basic_set_list::get() const {
  return ptr;
}

__isl_give isl_basic_set_list *basic_set_list::release() {
  isl_basic_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool basic_set_list::is_null() const {
  return ptr == nullptr;
}
basic_set_list::operator bool() const
{
  return !is_null();
}



ctx basic_set_list::get_ctx() const {
  return ctx(isl_basic_set_list_get_ctx(ptr));
}

basic_set_list basic_set_list::add(basic_set el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set_list basic_set_list::concat(basic_set_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set_list basic_set_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void basic_set_list::foreach(const std::function<void(basic_set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(basic_set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_basic_set_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

basic_set basic_set_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set_list basic_set_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int basic_set_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_basic_set_list_size(get());
  return res;
}

// implementations for isl::fixed_box
fixed_box manage(__isl_take isl_fixed_box *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return fixed_box(ptr);
}
fixed_box manage_copy(__isl_keep isl_fixed_box *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_fixed_box_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_fixed_box_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return fixed_box(ptr);
}

fixed_box::fixed_box()
    : ptr(nullptr) {}

fixed_box::fixed_box(const fixed_box &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_fixed_box_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

fixed_box::fixed_box(__isl_take isl_fixed_box *ptr)
    : ptr(ptr) {}


fixed_box &fixed_box::operator=(fixed_box obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

fixed_box::~fixed_box() {
  if (ptr)
    isl_fixed_box_free(ptr);
}

__isl_give isl_fixed_box *fixed_box::copy() const & {
  return isl_fixed_box_copy(ptr);
}

__isl_keep isl_fixed_box *fixed_box::get() const {
  return ptr;
}

__isl_give isl_fixed_box *fixed_box::release() {
  isl_fixed_box *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool fixed_box::is_null() const {
  return ptr == nullptr;
}
fixed_box::operator bool() const
{
  return !is_null();
}



ctx fixed_box::get_ctx() const {
  return ctx(isl_fixed_box_get_ctx(ptr));
}

multi_aff fixed_box::get_offset() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_fixed_box_get_offset(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val fixed_box::get_size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_fixed_box_get_size(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space fixed_box::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_fixed_box_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool fixed_box::is_valid() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_fixed_box_is_valid(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

// implementations for isl::id
id manage(__isl_take isl_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return id(ptr);
}
id manage_copy(__isl_keep isl_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_id_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_id_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return id(ptr);
}

id::id()
    : ptr(nullptr) {}

id::id(const id &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_id_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

id::id(__isl_take isl_id *ptr)
    : ptr(ptr) {}

id::id(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

id &id::operator=(id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

id::~id() {
  if (ptr)
    isl_id_free(ptr);
}

__isl_give isl_id *id::copy() const & {
  return isl_id_copy(ptr);
}

__isl_keep isl_id *id::get() const {
  return ptr;
}

__isl_give isl_id *id::release() {
  isl_id *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool id::is_null() const {
  return ptr == nullptr;
}
id::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const id& C) {
  os << C.to_str();
  return os;
}


std::string id::to_str() const {
  char *Tmp = isl_id_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx id::get_ctx() const {
  return ctx(isl_id_get_ctx(ptr));
}

std::string id::get_name() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_get_name(get());
  std::string tmp(res);
  return tmp;
}

// implementations for isl::id_list
id_list manage(__isl_take isl_id_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return id_list(ptr);
}
id_list manage_copy(__isl_keep isl_id_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_id_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_id_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return id_list(ptr);
}

id_list::id_list()
    : ptr(nullptr) {}

id_list::id_list(const id_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_id_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

id_list::id_list(__isl_take isl_id_list *ptr)
    : ptr(ptr) {}

id_list::id_list(id el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_from_id(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
id_list::id_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

id_list &id_list::operator=(id_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

id_list::~id_list() {
  if (ptr)
    isl_id_list_free(ptr);
}

__isl_give isl_id_list *id_list::copy() const & {
  return isl_id_list_copy(ptr);
}

__isl_keep isl_id_list *id_list::get() const {
  return ptr;
}

__isl_give isl_id_list *id_list::release() {
  isl_id_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool id_list::is_null() const {
  return ptr == nullptr;
}
id_list::operator bool() const
{
  return !is_null();
}



ctx id_list::get_ctx() const {
  return ctx(isl_id_list_get_ctx(ptr));
}

id_list id_list::add(id el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id_list id_list::concat(id_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id_list id_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void id_list::foreach(const std::function<void(id)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(id)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_id *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_id_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

id id_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id_list id_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int id_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_id_list_size(get());
  return res;
}

// implementations for isl::local_space
local_space manage(__isl_take isl_local_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return local_space(ptr);
}
local_space manage_copy(__isl_keep isl_local_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_local_space_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_local_space_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return local_space(ptr);
}

local_space::local_space()
    : ptr(nullptr) {}

local_space::local_space(const local_space &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_local_space_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

local_space::local_space(__isl_take isl_local_space *ptr)
    : ptr(ptr) {}

local_space::local_space(space dim)
{
  if (dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = dim.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_from_space(dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

local_space &local_space::operator=(local_space obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

local_space::~local_space() {
  if (ptr)
    isl_local_space_free(ptr);
}

__isl_give isl_local_space *local_space::copy() const & {
  return isl_local_space_copy(ptr);
}

__isl_keep isl_local_space *local_space::get() const {
  return ptr;
}

__isl_give isl_local_space *local_space::release() {
  isl_local_space *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool local_space::is_null() const {
  return ptr == nullptr;
}
local_space::operator bool() const
{
  return !is_null();
}

inline bool operator==(const local_space& C1, const local_space& C2) {
  return C1.is_equal(C2);
}



ctx local_space::get_ctx() const {
  return ctx(isl_local_space_get_ctx(ptr));
}

local_space local_space::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::from_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_from_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff local_space::get_div(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_get_div(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space local_space::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::intersect(local_space ls2) const
{
  if (!ptr || ls2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_intersect(copy(), ls2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool local_space::is_equal(const local_space &ls2) const
{
  if (!ptr || ls2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_is_equal(get(), ls2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool local_space::is_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_is_params(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool local_space::is_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_is_set(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

basic_map local_space::lifting() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_lifting(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

local_space local_space::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_local_space_wrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::map
map manage(__isl_take isl_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return map(ptr);
}
map manage_copy(__isl_keep isl_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return map(ptr);
}

map::map()
    : ptr(nullptr) {}

map::map(const map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

map::map(__isl_take isl_map *ptr)
    : ptr(ptr) {}

map::map(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
map::map(basic_map bmap)
{
  if (bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bmap.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_basic_map(bmap.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
map::map(set domain, set range)
{
  if (domain.is_null() || range.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_domain_and_range(domain.release(), range.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
map::map(aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = aff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
map::map(multi_aff maff)
{
  if (maff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = maff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_multi_aff(maff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

map &map::operator=(map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

map::~map() {
  if (ptr)
    isl_map_free(ptr);
}

__isl_give isl_map *map::copy() const & {
  return isl_map_copy(ptr);
}

__isl_keep isl_map *map::get() const {
  return ptr;
}

__isl_give isl_map *map::release() {
  isl_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool map::is_null() const {
  return ptr == nullptr;
}
map::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const map& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const map& C1, const map& C2) {
  return C1.is_equal(C2);
}


std::string map::to_str() const {
  char *Tmp = isl_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx map::get_ctx() const {
  return ctx(isl_map_get_ctx(ptr));
}

basic_map map::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::apply_domain(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_apply_domain(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::apply_range(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_apply_range(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool map::can_curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_can_curry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::can_range_curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_can_range_curry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::can_uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_can_uncurry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

map map::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_coalesce(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::complement() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_complement(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_compute_divs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_curry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_deltas(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set map::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::domain_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_domain_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::domain_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_domain_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_domain_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::domain_product(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_domain_product(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_flatten(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::flatten_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_flatten_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void map::foreach_basic_map(const std::function<void(basic_map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(basic_map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_map_foreach_basic_map(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

map map::from(pw_multi_aff pma)
{
  if (pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_pw_multi_aff(pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::from(union_map umap)
{
  if (umap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = umap.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_union_map(umap.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::from_domain(set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = set.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_domain(set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::from_range(set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = set.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_from_range(set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map_list map::get_basic_map_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_get_basic_map_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

fixed_box map::get_range_simple_fixed_box_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_get_range_simple_fixed_box_hull(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

stride_info map::get_range_stride_info(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_get_range_stride_info(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id map::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space map::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::gist(map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::gist_domain(set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_gist_domain(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::identity(space dim)
{
  if (dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = dim.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_identity(dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::intersect(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_intersect(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::intersect_domain(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_intersect_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::intersect_params(set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::intersect_range(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_intersect_range(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool map::is_bijective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_bijective(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_disjoint(const map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_disjoint(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_equal(const map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_equal(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_injective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_injective(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_single_valued() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_single_valued(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_strict_subset(const map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_strict_subset(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool map::is_subset(const map &map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_is_subset(get(), map2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

map map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int map::n_basic_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_n_basic_map(get());
  return res;
}

set map::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map map::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::preimage_domain(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_preimage_domain_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::preimage_range(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_preimage_range_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set map::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::range_curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range_curry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::range_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::range_product(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_range_product(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map map::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_sample(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::set_range_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_set_range_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map map::simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_simple_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::subtract(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_subtract(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::sum(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_sum(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_uncurry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::unite(map map2) const
{
  if (!ptr || map2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_union(copy(), map2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::universe(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_universe(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_map map::unshifted_simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_unshifted_simple_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set map::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_wrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map map::zip() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_zip(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::map_list
map_list manage(__isl_take isl_map_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return map_list(ptr);
}
map_list manage_copy(__isl_keep isl_map_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_map_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_map_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return map_list(ptr);
}

map_list::map_list()
    : ptr(nullptr) {}

map_list::map_list(const map_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_map_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

map_list::map_list(__isl_take isl_map_list *ptr)
    : ptr(ptr) {}

map_list::map_list(map el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_from_map(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
map_list::map_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

map_list &map_list::operator=(map_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

map_list::~map_list() {
  if (ptr)
    isl_map_list_free(ptr);
}

__isl_give isl_map_list *map_list::copy() const & {
  return isl_map_list_copy(ptr);
}

__isl_keep isl_map_list *map_list::get() const {
  return ptr;
}

__isl_give isl_map_list *map_list::release() {
  isl_map_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool map_list::is_null() const {
  return ptr == nullptr;
}
map_list::operator bool() const
{
  return !is_null();
}



ctx map_list::get_ctx() const {
  return ctx(isl_map_list_get_ctx(ptr));
}

map_list map_list::add(map el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map_list map_list::concat(map_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map_list map_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void map_list::foreach(const std::function<void(map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_map_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

map map_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map_list map_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int map_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_map_list_size(get());
  return res;
}

// implementations for isl::multi_aff
multi_aff manage(__isl_take isl_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_aff(ptr);
}
multi_aff manage_copy(__isl_keep isl_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return multi_aff(ptr);
}

multi_aff::multi_aff()
    : ptr(nullptr) {}

multi_aff::multi_aff(const multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

multi_aff::multi_aff(__isl_take isl_multi_aff *ptr)
    : ptr(ptr) {}

multi_aff::multi_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_aff::multi_aff(space space, aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_from_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_aff::multi_aff(aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = aff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

multi_aff &multi_aff::operator=(multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_aff::~multi_aff() {
  if (ptr)
    isl_multi_aff_free(ptr);
}

__isl_give isl_multi_aff *multi_aff::copy() const & {
  return isl_multi_aff_copy(ptr);
}

__isl_keep isl_multi_aff *multi_aff::get() const {
  return ptr;
}

__isl_give isl_multi_aff *multi_aff::release() {
  isl_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_aff::is_null() const {
  return ptr == nullptr;
}
multi_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const multi_aff& C) {
  os << C.to_str();
  return os;
}


std::string multi_aff::to_str() const {
  char *Tmp = isl_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx multi_aff::get_ctx() const {
  return ctx(isl_multi_aff_get_ctx(ptr));
}

multi_aff multi_aff::add(multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::domain_map(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_domain_map(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::flat_range_product(multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff multi_aff::get_aff(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_get_aff(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

aff_list multi_aff::get_aff_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_get_aff_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_aff::get_domain_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_get_domain_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id multi_aff::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::identity(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_identity(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::mod(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_mod_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::product(multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::pullback(multi_aff ma2) const
{
  if (!ptr || ma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_pullback_multi_aff(copy(), ma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::range_map(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_range_map(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::range_product(multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::range_splice(unsigned int pos, multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_range_splice(copy(), pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::scale(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::scale_down(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::set_aff(int pos, aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_set_aff(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::set_range_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_set_range_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int multi_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_size(get());
  return res;
}

multi_aff multi_aff::splice(unsigned int in_pos, unsigned int out_pos, multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_splice(copy(), in_pos, out_pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::sub(multi_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::wrapped_range_map(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_wrapped_range_map(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_aff multi_aff::zero(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::multi_id
multi_id manage(__isl_take isl_multi_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_id(ptr);
}
multi_id manage_copy(__isl_keep isl_multi_id *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_id_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_multi_id_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return multi_id(ptr);
}

multi_id::multi_id()
    : ptr(nullptr) {}

multi_id::multi_id(const multi_id &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_id_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

multi_id::multi_id(__isl_take isl_multi_id *ptr)
    : ptr(ptr) {}

multi_id::multi_id(space space, id_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_from_id_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

multi_id &multi_id::operator=(multi_id obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_id::~multi_id() {
  if (ptr)
    isl_multi_id_free(ptr);
}

__isl_give isl_multi_id *multi_id::copy() const & {
  return isl_multi_id_copy(ptr);
}

__isl_keep isl_multi_id *multi_id::get() const {
  return ptr;
}

__isl_give isl_multi_id *multi_id::release() {
  isl_multi_id *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_id::is_null() const {
  return ptr == nullptr;
}
multi_id::operator bool() const
{
  return !is_null();
}



ctx multi_id::get_ctx() const {
  return ctx(isl_multi_id_get_ctx(ptr));
}

multi_id multi_id::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::flat_range_product(multi_id multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_id::get_domain_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_get_domain_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id multi_id::get_id(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_get_id(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id_list multi_id::get_id_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_get_id_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_id::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::range_product(multi_id multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::range_splice(unsigned int pos, multi_id multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_range_splice(copy(), pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_id multi_id::set_id(int pos, id el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_set_id(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int multi_id::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_id_size(get());
  return res;
}

// implementations for isl::multi_pw_aff
multi_pw_aff manage(__isl_take isl_multi_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_pw_aff(ptr);
}
multi_pw_aff manage_copy(__isl_keep isl_multi_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_multi_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return multi_pw_aff(ptr);
}

multi_pw_aff::multi_pw_aff()
    : ptr(nullptr) {}

multi_pw_aff::multi_pw_aff(const multi_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

multi_pw_aff::multi_pw_aff(__isl_take isl_multi_pw_aff *ptr)
    : ptr(ptr) {}

multi_pw_aff::multi_pw_aff(space space, pw_aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_pw_aff::multi_pw_aff(multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_pw_aff::multi_pw_aff(pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_pw_aff::multi_pw_aff(pw_multi_aff pma)
{
  if (pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_pw_multi_aff(pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_pw_aff::multi_pw_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

multi_pw_aff &multi_pw_aff::operator=(multi_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_pw_aff::~multi_pw_aff() {
  if (ptr)
    isl_multi_pw_aff_free(ptr);
}

__isl_give isl_multi_pw_aff *multi_pw_aff::copy() const & {
  return isl_multi_pw_aff_copy(ptr);
}

__isl_keep isl_multi_pw_aff *multi_pw_aff::get() const {
  return ptr;
}

__isl_give isl_multi_pw_aff *multi_pw_aff::release() {
  isl_multi_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_pw_aff::is_null() const {
  return ptr == nullptr;
}
multi_pw_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const multi_pw_aff& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const multi_pw_aff& C1, const multi_pw_aff& C2) {
  return C1.is_equal(C2);
}


std::string multi_pw_aff::to_str() const {
  char *Tmp = isl_multi_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx multi_pw_aff::get_ctx() const {
  return ctx(isl_multi_pw_aff_get_ctx(ptr));
}

multi_pw_aff multi_pw_aff::add(multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set multi_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::flat_range_product(multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_pw_aff::get_domain_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_domain_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff multi_pw_aff::get_pw_aff(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_pw_aff(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff_list multi_pw_aff::get_pw_aff_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_pw_aff_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id multi_pw_aff::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_pw_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::identity(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_identity(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool multi_pw_aff::is_equal(const multi_pw_aff &mpa2) const
{
  if (!ptr || mpa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_is_equal(get(), mpa2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

multi_pw_aff multi_pw_aff::mod(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_mod_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::product(multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::pullback(multi_pw_aff mpa2) const
{
  if (!ptr || mpa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_pullback_multi_pw_aff(copy(), mpa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_product(multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::range_splice(unsigned int pos, multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_range_splice(copy(), pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::scale_down(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_pw_aff(int pos, pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_set_pw_aff(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::set_range_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_set_range_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int multi_pw_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_size(get());
  return res;
}

multi_pw_aff multi_pw_aff::splice(unsigned int in_pos, unsigned int out_pos, multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_splice(copy(), in_pos, out_pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::sub(multi_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_pw_aff::zero(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_pw_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::multi_union_pw_aff
multi_union_pw_aff manage(__isl_take isl_multi_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_union_pw_aff(ptr);
}
multi_union_pw_aff manage_copy(__isl_keep isl_multi_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_union_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_multi_union_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return multi_union_pw_aff(ptr);
}

multi_union_pw_aff::multi_union_pw_aff()
    : ptr(nullptr) {}

multi_union_pw_aff::multi_union_pw_aff(const multi_union_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_union_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

multi_union_pw_aff::multi_union_pw_aff(__isl_take isl_multi_union_pw_aff *ptr)
    : ptr(ptr) {}

multi_union_pw_aff::multi_union_pw_aff(union_pw_aff upa)
{
  if (upa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = upa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_union_pw_aff(upa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(multi_pw_aff mpa)
{
  if (mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = mpa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_multi_pw_aff(mpa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(union_set domain, multi_val mv)
{
  if (domain.is_null() || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_multi_val_on_domain(domain.release(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(union_set domain, multi_aff ma)
{
  if (domain.is_null() || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_multi_aff_on_domain(domain.release(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(space space, union_pw_aff_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_union_pw_aff_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
multi_union_pw_aff::multi_union_pw_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

multi_union_pw_aff &multi_union_pw_aff::operator=(multi_union_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_union_pw_aff::~multi_union_pw_aff() {
  if (ptr)
    isl_multi_union_pw_aff_free(ptr);
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::copy() const & {
  return isl_multi_union_pw_aff_copy(ptr);
}

__isl_keep isl_multi_union_pw_aff *multi_union_pw_aff::get() const {
  return ptr;
}

__isl_give isl_multi_union_pw_aff *multi_union_pw_aff::release() {
  isl_multi_union_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_union_pw_aff::is_null() const {
  return ptr == nullptr;
}
multi_union_pw_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const multi_union_pw_aff& C) {
  os << C.to_str();
  return os;
}


std::string multi_union_pw_aff::to_str() const {
  char *Tmp = isl_multi_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx multi_union_pw_aff::get_ctx() const {
  return ctx(isl_multi_union_pw_aff_get_ctx(ptr));
}

multi_union_pw_aff multi_union_pw_aff::add(multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::apply(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_apply_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::apply(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_apply_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set multi_union_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_pw_aff multi_union_pw_aff::extract_multi_pw_aff(space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_extract_multi_pw_aff(get(), space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::flat_range_product(multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::from_union_map(union_map umap)
{
  if (umap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = umap.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_from_union_map(umap.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_union_pw_aff::get_domain_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_domain_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id multi_union_pw_aff::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_union_pw_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff multi_union_pw_aff::get_union_pw_aff(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_union_pw_aff(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff_list multi_union_pw_aff::get_union_pw_aff_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_get_union_pw_aff_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::gist(union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::intersect_domain(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_intersect_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::intersect_params(set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool multi_union_pw_aff::involves_param(const id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_involves_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

multi_val multi_union_pw_aff::max_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_max_multi_val(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_union_pw_aff::min_multi_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_min_multi_val(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::mod(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_mod_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_product(multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::range_splice(unsigned int pos, multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_range_splice(copy(), pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::scale_down(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_range_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_set_range_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::set_union_pw_aff(int pos, union_pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_set_union_pw_aff(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int multi_union_pw_aff::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_size(get());
  return res;
}

multi_union_pw_aff multi_union_pw_aff::sub(multi_union_pw_aff multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::union_add(multi_union_pw_aff mupa2) const
{
  if (!ptr || mupa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_union_add(copy(), mupa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff multi_union_pw_aff::zero(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_zero(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set multi_union_pw_aff::zero_union_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_union_pw_aff_zero_union_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::multi_val
multi_val manage(__isl_take isl_multi_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return multi_val(ptr);
}
multi_val manage_copy(__isl_keep isl_multi_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_val_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_multi_val_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return multi_val(ptr);
}

multi_val::multi_val()
    : ptr(nullptr) {}

multi_val::multi_val(const multi_val &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_multi_val_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

multi_val::multi_val(__isl_take isl_multi_val *ptr)
    : ptr(ptr) {}

multi_val::multi_val(space space, val_list list)
{
  if (space.is_null() || list.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_from_val_list(space.release(), list.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

multi_val &multi_val::operator=(multi_val obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

multi_val::~multi_val() {
  if (ptr)
    isl_multi_val_free(ptr);
}

__isl_give isl_multi_val *multi_val::copy() const & {
  return isl_multi_val_copy(ptr);
}

__isl_keep isl_multi_val *multi_val::get() const {
  return ptr;
}

__isl_give isl_multi_val *multi_val::release() {
  isl_multi_val *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool multi_val::is_null() const {
  return ptr == nullptr;
}
multi_val::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const multi_val& C) {
  os << C.to_str();
  return os;
}


std::string multi_val::to_str() const {
  char *Tmp = isl_multi_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx multi_val::get_ctx() const {
  return ctx(isl_multi_val_get_ctx(ptr));
}

multi_val multi_val::add(multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_add(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::flat_range_product(multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_flat_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::flatten_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_flatten_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_val::get_domain_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_get_domain_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id multi_val::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space multi_val::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val multi_val::get_val(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_get_val(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val_list multi_val::get_val_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_get_val_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::mod(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_mod_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::product(multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::range_product(multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_range_product(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::range_splice(unsigned int pos, multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_range_splice(copy(), pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::scale(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_scale_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::scale_down(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_scale_down_multi_val(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::set_range_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_set_range_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::set_val(int pos, val el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_set_val(copy(), pos, el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int multi_val::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_size(get());
  return res;
}

multi_val multi_val::splice(unsigned int in_pos, unsigned int out_pos, multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_splice(copy(), in_pos, out_pos, multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::sub(multi_val multi2) const
{
  if (!ptr || multi2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_sub(copy(), multi2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_val multi_val::zero(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_multi_val_zero(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::point
point manage(__isl_take isl_point *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return point(ptr);
}
point manage_copy(__isl_keep isl_point *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_point_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_point_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return point(ptr);
}

point::point()
    : ptr(nullptr) {}

point::point(const point &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_point_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

point::point(__isl_take isl_point *ptr)
    : ptr(ptr) {}

point::point(space dim)
{
  if (dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = dim.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_point_zero(dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

point &point::operator=(point obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

point::~point() {
  if (ptr)
    isl_point_free(ptr);
}

__isl_give isl_point *point::copy() const & {
  return isl_point_copy(ptr);
}

__isl_keep isl_point *point::get() const {
  return ptr;
}

__isl_give isl_point *point::release() {
  isl_point *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool point::is_null() const {
  return ptr == nullptr;
}
point::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const point& C) {
  os << C.to_str();
  return os;
}


std::string point::to_str() const {
  char *Tmp = isl_point_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx point::get_ctx() const {
  return ctx(isl_point_get_ctx(ptr));
}

space point::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_point_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool point::is_void() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_point_is_void(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

// implementations for isl::pw_aff
pw_aff manage(__isl_take isl_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_aff(ptr);
}
pw_aff manage_copy(__isl_keep isl_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return pw_aff(ptr);
}

pw_aff::pw_aff()
    : ptr(nullptr) {}

pw_aff::pw_aff(const pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

pw_aff::pw_aff(__isl_take isl_pw_aff *ptr)
    : ptr(ptr) {}

pw_aff::pw_aff(aff aff)
{
  if (aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = aff.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_from_aff(aff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_aff::pw_aff(local_space ls)
{
  if (ls.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ls.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_zero_on_domain(ls.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_aff::pw_aff(set domain, val v)
{
  if (domain.is_null() || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_val_on_domain(domain.release(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_aff::pw_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

pw_aff &pw_aff::operator=(pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_aff::~pw_aff() {
  if (ptr)
    isl_pw_aff_free(ptr);
}

__isl_give isl_pw_aff *pw_aff::copy() const & {
  return isl_pw_aff_copy(ptr);
}

__isl_keep isl_pw_aff *pw_aff::get() const {
  return ptr;
}

__isl_give isl_pw_aff *pw_aff::release() {
  isl_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_aff::is_null() const {
  return ptr == nullptr;
}
pw_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const pw_aff& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const pw_aff& C1, const pw_aff& C2) {
  return C1.is_equal(C2);
}


std::string pw_aff::to_str() const {
  char *Tmp = isl_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx pw_aff::get_ctx() const {
  return ctx(isl_pw_aff_get_ctx(ptr));
}

pw_aff pw_aff::add(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_add(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_ceil(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::cond(pw_aff pwaff_true, pw_aff pwaff_false) const
{
  if (!ptr || pwaff_true.is_null() || pwaff_false.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_cond(copy(), pwaff_true.release(), pwaff_false.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::div(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_div(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map pw_aff::eq_map(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_eq_map(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::eq_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_eq_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void pw_aff::foreach_piece(const std::function<void(set, aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(set, aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, isl_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0), manage(arg_1));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_aff_foreach_piece(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

set pw_aff::ge_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_ge_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space pw_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map pw_aff::gt_map(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_gt_map(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::gt_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_gt_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::intersect_domain(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_intersect_domain(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::intersect_params(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool pw_aff::involves_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_involves_nan(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool pw_aff::is_cst() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_is_cst(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool pw_aff::is_equal(const pw_aff &pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_is_equal(get(), pa2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

set pw_aff::le_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_le_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map pw_aff::lt_map(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_lt_map(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::lt_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_lt_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::max(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_max(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::min(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_min(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::mod(val mod) const
{
  if (!ptr || mod.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_mod_val(copy(), mod.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::mul(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_mul(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int pw_aff::n_piece() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_n_piece(get());
  return res;
}

set pw_aff::ne_set(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_ne_set(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::nonneg_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_nonneg_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::pos_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_pos_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::project_domain_on_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_project_domain_on_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::pullback(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::pullback(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::pullback(multi_pw_aff mpa) const
{
  if (!ptr || mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_pullback_multi_pw_aff(copy(), mpa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::scale_down(val f) const
{
  if (!ptr || f.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_scale_down_val(copy(), f.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::sub(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_sub(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::tdiv_q(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_tdiv_q(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::tdiv_r(pw_aff pa2) const
{
  if (!ptr || pa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_tdiv_r(copy(), pa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_aff::union_add(pw_aff pwaff2) const
{
  if (!ptr || pwaff2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_union_add(copy(), pwaff2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_aff::zero_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_zero_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::pw_aff_list
pw_aff_list manage(__isl_take isl_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_aff_list(ptr);
}
pw_aff_list manage_copy(__isl_keep isl_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_pw_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return pw_aff_list(ptr);
}

pw_aff_list::pw_aff_list()
    : ptr(nullptr) {}

pw_aff_list::pw_aff_list(const pw_aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

pw_aff_list::pw_aff_list(__isl_take isl_pw_aff_list *ptr)
    : ptr(ptr) {}

pw_aff_list::pw_aff_list(pw_aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_from_pw_aff(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_aff_list::pw_aff_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

pw_aff_list &pw_aff_list::operator=(pw_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_aff_list::~pw_aff_list() {
  if (ptr)
    isl_pw_aff_list_free(ptr);
}

__isl_give isl_pw_aff_list *pw_aff_list::copy() const & {
  return isl_pw_aff_list_copy(ptr);
}

__isl_keep isl_pw_aff_list *pw_aff_list::get() const {
  return ptr;
}

__isl_give isl_pw_aff_list *pw_aff_list::release() {
  isl_pw_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_aff_list::is_null() const {
  return ptr == nullptr;
}
pw_aff_list::operator bool() const
{
  return !is_null();
}



ctx pw_aff_list::get_ctx() const {
  return ctx(isl_pw_aff_list_get_ctx(ptr));
}

pw_aff_list pw_aff_list::add(pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff_list pw_aff_list::concat(pw_aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff_list pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void pw_aff_list::foreach(const std::function<void(pw_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(pw_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

pw_aff pw_aff_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff_list pw_aff_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int pw_aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_aff_list_size(get());
  return res;
}

// implementations for isl::pw_multi_aff
pw_multi_aff manage(__isl_take isl_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return pw_multi_aff(ptr);
}
pw_multi_aff manage_copy(__isl_keep isl_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_pw_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return pw_multi_aff(ptr);
}

pw_multi_aff::pw_multi_aff()
    : ptr(nullptr) {}

pw_multi_aff::pw_multi_aff(const pw_multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_pw_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

pw_multi_aff::pw_multi_aff(__isl_take isl_pw_multi_aff *ptr)
    : ptr(ptr) {}

pw_multi_aff::pw_multi_aff(multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_multi_aff::pw_multi_aff(pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_multi_aff::pw_multi_aff(set domain, multi_val mv)
{
  if (domain.is_null() || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_multi_aff::pw_multi_aff(map map)
{
  if (map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = map.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_map(map.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
pw_multi_aff::pw_multi_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

pw_multi_aff &pw_multi_aff::operator=(pw_multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

pw_multi_aff::~pw_multi_aff() {
  if (ptr)
    isl_pw_multi_aff_free(ptr);
}

__isl_give isl_pw_multi_aff *pw_multi_aff::copy() const & {
  return isl_pw_multi_aff_copy(ptr);
}

__isl_keep isl_pw_multi_aff *pw_multi_aff::get() const {
  return ptr;
}

__isl_give isl_pw_multi_aff *pw_multi_aff::release() {
  isl_pw_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool pw_multi_aff::is_null() const {
  return ptr == nullptr;
}
pw_multi_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const pw_multi_aff& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const pw_multi_aff& C1, const pw_multi_aff& C2) {
  return C1.is_equal(C2);
}


std::string pw_multi_aff::to_str() const {
  char *Tmp = isl_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx pw_multi_aff::get_ctx() const {
  return ctx(isl_pw_multi_aff_get_ctx(ptr));
}

pw_multi_aff pw_multi_aff::add(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_add(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set pw_multi_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::flat_range_product(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_flat_range_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void pw_multi_aff::foreach_piece(const std::function<void(set, multi_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(set, multi_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, isl_multi_aff *arg_1, void *arg_2) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_2);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0), manage(arg_1));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_pw_multi_aff_foreach_piece(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

pw_multi_aff pw_multi_aff::from(multi_pw_aff mpa)
{
  if (mpa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = mpa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_from_multi_pw_aff(mpa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff pw_multi_aff::get_pw_aff(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_get_pw_aff(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id pw_multi_aff::get_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_get_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space pw_multi_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::identity(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_identity(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool pw_multi_aff::is_equal(const pw_multi_aff &pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_is_equal(get(), pma2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

int pw_multi_aff::n_piece() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_n_piece(get());
  return res;
}

pw_multi_aff pw_multi_aff::product(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::project_domain_on_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_project_domain_on_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::pullback(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_pullback_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::pullback(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_pullback_pw_multi_aff(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::range_product(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_range_product(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::set_pw_aff(unsigned int pos, pw_aff pa) const
{
  if (!ptr || pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_set_pw_aff(copy(), pos, pa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff pw_multi_aff::union_add(pw_multi_aff pma2) const
{
  if (!ptr || pma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_pw_multi_aff_union_add(copy(), pma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule
schedule manage(__isl_take isl_schedule *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule(ptr);
}
schedule manage_copy(__isl_keep isl_schedule *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_schedule_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return schedule(ptr);
}

schedule::schedule()
    : ptr(nullptr) {}

schedule::schedule(const schedule &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

schedule::schedule(__isl_take isl_schedule *ptr)
    : ptr(ptr) {}

schedule::schedule(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

schedule &schedule::operator=(schedule obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule::~schedule() {
  if (ptr)
    isl_schedule_free(ptr);
}

__isl_give isl_schedule *schedule::copy() const & {
  return isl_schedule_copy(ptr);
}

__isl_keep isl_schedule *schedule::get() const {
  return ptr;
}

__isl_give isl_schedule *schedule::release() {
  isl_schedule *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule::is_null() const {
  return ptr == nullptr;
}
schedule::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const schedule& C) {
  os << C.to_str();
  return os;
}


std::string schedule::to_str() const {
  char *Tmp = isl_schedule_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule::get_ctx() const {
  return ctx(isl_schedule_get_ctx(ptr));
}

schedule schedule::from_domain(union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_from_domain(domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set schedule::get_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_get_domain(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule::get_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_get_map(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule::get_root() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_get_root(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule schedule::insert_partial_schedule(multi_union_pw_aff partial) const
{
  if (!ptr || partial.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_insert_partial_schedule(copy(), partial.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule::plain_is_equal(const schedule &schedule2) const
{
  if (!ptr || schedule2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_plain_is_equal(get(), schedule2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

schedule schedule::pullback(union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule schedule::reset_user() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_reset_user(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule schedule::sequence(schedule schedule2) const
{
  if (!ptr || schedule2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_sequence(copy(), schedule2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule schedule::set(schedule schedule2) const
{
  if (!ptr || schedule2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_set(copy(), schedule2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_constraints
schedule_constraints manage(__isl_take isl_schedule_constraints *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule_constraints(ptr);
}
schedule_constraints manage_copy(__isl_keep isl_schedule_constraints *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_constraints_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_schedule_constraints_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return schedule_constraints(ptr);
}

schedule_constraints::schedule_constraints()
    : ptr(nullptr) {}

schedule_constraints::schedule_constraints(const schedule_constraints &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_constraints_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

schedule_constraints::schedule_constraints(__isl_take isl_schedule_constraints *ptr)
    : ptr(ptr) {}

schedule_constraints::schedule_constraints(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

schedule_constraints &schedule_constraints::operator=(schedule_constraints obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule_constraints::~schedule_constraints() {
  if (ptr)
    isl_schedule_constraints_free(ptr);
}

__isl_give isl_schedule_constraints *schedule_constraints::copy() const & {
  return isl_schedule_constraints_copy(ptr);
}

__isl_keep isl_schedule_constraints *schedule_constraints::get() const {
  return ptr;
}

__isl_give isl_schedule_constraints *schedule_constraints::release() {
  isl_schedule_constraints *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule_constraints::is_null() const {
  return ptr == nullptr;
}
schedule_constraints::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const schedule_constraints& C) {
  os << C.to_str();
  return os;
}


std::string schedule_constraints::to_str() const {
  char *Tmp = isl_schedule_constraints_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_constraints::get_ctx() const {
  return ctx(isl_schedule_constraints_get_ctx(ptr));
}

schedule schedule_constraints::compute_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_compute_schedule(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_constraints::get_coincidence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_coincidence(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_constraints::get_conditional_validity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_conditional_validity(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_constraints::get_conditional_validity_condition() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_conditional_validity_condition(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set schedule_constraints::get_context() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_context(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set schedule_constraints::get_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_domain(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff schedule_constraints::get_prefix() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_prefix(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_constraints::get_proximity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_proximity(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_constraints::get_validity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_get_validity(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::intersect_domain(union_set domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_intersect_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::on_domain(union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_on_domain(domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_coincidence(union_map coincidence) const
{
  if (!ptr || coincidence.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_coincidence(copy(), coincidence.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_conditional_validity(union_map condition, union_map validity) const
{
  if (!ptr || condition.is_null() || validity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_conditional_validity(copy(), condition.release(), validity.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_context(set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_context(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_prefix(multi_union_pw_aff prefix) const
{
  if (!ptr || prefix.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_prefix(copy(), prefix.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_proximity(union_map proximity) const
{
  if (!ptr || proximity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_proximity(copy(), proximity.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_constraints schedule_constraints::set_validity(union_map validity) const
{
  if (!ptr || validity.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_constraints_set_validity(copy(), validity.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node
schedule_node manage(__isl_take isl_schedule_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return schedule_node(ptr);
}
schedule_node manage_copy(__isl_keep isl_schedule_node *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_node_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_schedule_node_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return schedule_node(ptr);
}

schedule_node::schedule_node()
    : ptr(nullptr) {}

schedule_node::schedule_node(const schedule_node &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_schedule_node_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

schedule_node::schedule_node(__isl_take isl_schedule_node *ptr)
    : ptr(ptr) {}


schedule_node &schedule_node::operator=(schedule_node obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

schedule_node::~schedule_node() {
  if (ptr)
    isl_schedule_node_free(ptr);
}

__isl_give isl_schedule_node *schedule_node::copy() const & {
  return isl_schedule_node_copy(ptr);
}

__isl_keep isl_schedule_node *schedule_node::get() const {
  return ptr;
}

__isl_give isl_schedule_node *schedule_node::release() {
  isl_schedule_node *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool schedule_node::is_null() const {
  return ptr == nullptr;
}
schedule_node::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const schedule_node& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node& C1, const schedule_node& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


template <class T>
bool schedule_node::isa()
{
  if (is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return isl_schedule_node_get_type(get()) == T::type;
}
template <class T>
T schedule_node::as()
{
  return isa<T>() ? T(copy()) : T();
}

ctx schedule_node::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

schedule_node schedule_node::ancestor(int generation) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_ancestor(copy(), generation);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::child(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_child(copy(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::cut() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_cut(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::del() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_delete(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule_node::every_descendant(const std::function<bool(schedule_node)> &test) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct test_data {
    std::function<bool(schedule_node)> func;
    std::exception_ptr eptr;
  } test_data = { test };
  auto test_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_schedule_node_every_descendant(get(), test_lambda, &test_data);
  if (test_data.eptr)
    std::rethrow_exception(test_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

schedule_node schedule_node::first_child() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_first_child(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void schedule_node::foreach_descendant_top_down(const std::function<bool(schedule_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<bool(schedule_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_schedule_node_foreach_descendant_top_down(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

schedule_node schedule_node::from_domain(union_set domain)
{
  if (domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_from_domain(domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::from_extension(union_map extension)
{
  if (extension.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = extension.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_from_extension(extension.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int schedule_node::get_ancestor_child_position(const schedule_node &ancestor) const
{
  if (!ptr || ancestor.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_ancestor_child_position(get(), ancestor.get());
  return res;
}

schedule_node schedule_node::get_child(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_child(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int schedule_node::get_child_position() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_child_position(get());
  return res;
}

union_set schedule_node::get_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_domain(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff schedule_node::get_prefix_schedule_multi_union_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_node::get_prefix_schedule_relation() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_relation(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_node::get_prefix_schedule_union_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_union_map(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff schedule_node::get_prefix_schedule_union_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_prefix_schedule_union_pw_multi_aff(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule schedule_node::get_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_schedule(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int schedule_node::get_schedule_depth() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_schedule_depth(get());
  return res;
}

schedule_node schedule_node::get_shared_ancestor(const schedule_node &node2) const
{
  if (!ptr || node2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_shared_ancestor(get(), node2.get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int schedule_node::get_tree_depth() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_tree_depth(get());
  return res;
}

union_set schedule_node::get_universe_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_get_universe_domain(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::graft_after(schedule_node graft) const
{
  if (!ptr || graft.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_graft_after(copy(), graft.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::graft_before(schedule_node graft) const
{
  if (!ptr || graft.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_graft_before(copy(), graft.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule_node::has_children() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_has_children(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool schedule_node::has_next_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_has_next_sibling(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool schedule_node::has_parent() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_has_parent(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool schedule_node::has_previous_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_has_previous_sibling(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

schedule_node schedule_node::insert_context(set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_context(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_filter(union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_filter(copy(), filter.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_guard(set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_guard(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_mark(id mark) const
{
  if (!ptr || mark.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_mark(copy(), mark.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_partial_schedule(multi_union_pw_aff schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_partial_schedule(copy(), schedule.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_sequence(union_set_list filters) const
{
  if (!ptr || filters.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_sequence(copy(), filters.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::insert_set(union_set_list filters) const
{
  if (!ptr || filters.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_insert_set(copy(), filters.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule_node::is_equal(const schedule_node &node2) const
{
  if (!ptr || node2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_is_equal(get(), node2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool schedule_node::is_subtree_anchored() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_is_subtree_anchored(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

schedule_node schedule_node::map_descendant_bottom_up(const std::function<schedule_node(schedule_node)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<schedule_node(schedule_node)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_schedule_node *arg_0, void *arg_1) -> isl_schedule_node * {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage(arg_0));
      return ret.release();
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return NULL;
    }
  };
  auto res = isl_schedule_node_map_descendant_bottom_up(copy(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int schedule_node::n_children() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_n_children(get());
  return res;
}

schedule_node schedule_node::next_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_next_sibling(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::order_after(union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_order_after(copy(), filter.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::order_before(union_set filter) const
{
  if (!ptr || filter.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_order_before(copy(), filter.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::parent() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_parent(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::previous_sibling() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_previous_sibling(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

schedule_node schedule_node::root() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_root(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_band

schedule_node_band::schedule_node_band()
    : schedule_node() {}

schedule_node_band::schedule_node_band(const schedule_node_band &obj)
    : schedule_node(obj)
{
}

schedule_node_band::schedule_node_band(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_band &schedule_node_band::operator=(schedule_node_band obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_band& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_band& C1, const schedule_node_band& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_band::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_band::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

union_set schedule_node_band::get_ast_build_options() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_ast_build_options(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set schedule_node_band::get_ast_isolate_option() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_ast_isolate_option(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

multi_union_pw_aff schedule_node_band::get_partial_schedule() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_partial_schedule(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_node_band::get_partial_schedule_union_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_partial_schedule_union_map(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule_node_band::get_permutable() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_permutable(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

space schedule_node_band::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool schedule_node_band::member_get_coincident(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_get_coincident(get(), pos);
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

schedule_node_band schedule_node_band::member_set_coincident(int pos, int coincident) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_coincident(copy(), pos, coincident);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::mod(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_mod(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

unsigned int schedule_node_band::n_member() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_n_member(get());
  return res;
}

schedule_node_band schedule_node_band::scale(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_scale(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::scale_down(multi_val mv) const
{
  if (!ptr || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_scale_down(copy(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_ast_build_options(union_set options) const
{
  if (!ptr || options.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_set_ast_build_options(copy(), options.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::set_permutable(int permutable) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_set_permutable(copy(), permutable);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::shift(multi_union_pw_aff shift) const
{
  if (!ptr || shift.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_shift(copy(), shift.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::split(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_split(copy(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::tile(multi_val sizes) const
{
  if (!ptr || sizes.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_tile(copy(), sizes.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_default(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_default);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_atomic(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_atomic);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_unroll(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_unroll);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_ast_loop_separate(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_ast_loop_type(copy(), pos, isl_ast_loop_separate);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_isolate_ast_loop_default(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_isolate_ast_loop_type(copy(), pos, isl_ast_loop_default);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_isolate_ast_loop_atomic(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_isolate_ast_loop_type(copy(), pos, isl_ast_loop_atomic);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_isolate_ast_loop_unroll(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_isolate_ast_loop_type(copy(), pos, isl_ast_loop_unroll);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

schedule_node_band schedule_node_band::member_set_isolate_ast_loop_separate(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_band_member_set_isolate_ast_loop_type(copy(), pos, isl_ast_loop_separate);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res).as<schedule_node_band>();
}

// implementations for isl::schedule_node_context

schedule_node_context::schedule_node_context()
    : schedule_node() {}

schedule_node_context::schedule_node_context(const schedule_node_context &obj)
    : schedule_node(obj)
{
}

schedule_node_context::schedule_node_context(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_context &schedule_node_context::operator=(schedule_node_context obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_context& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_context& C1, const schedule_node_context& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_context::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_context::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

set schedule_node_context::get_context() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_context_get_context(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_domain

schedule_node_domain::schedule_node_domain()
    : schedule_node() {}

schedule_node_domain::schedule_node_domain(const schedule_node_domain &obj)
    : schedule_node(obj)
{
}

schedule_node_domain::schedule_node_domain(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_domain &schedule_node_domain::operator=(schedule_node_domain obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_domain& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_domain& C1, const schedule_node_domain& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_domain::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_domain::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

union_set schedule_node_domain::get_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_domain_get_domain(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_expansion

schedule_node_expansion::schedule_node_expansion()
    : schedule_node() {}

schedule_node_expansion::schedule_node_expansion(const schedule_node_expansion &obj)
    : schedule_node(obj)
{
}

schedule_node_expansion::schedule_node_expansion(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_expansion &schedule_node_expansion::operator=(schedule_node_expansion obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_expansion& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_expansion& C1, const schedule_node_expansion& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_expansion::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_expansion::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

union_pw_multi_aff schedule_node_expansion::get_contraction() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_expansion_get_contraction(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map schedule_node_expansion::get_expansion() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_expansion_get_expansion(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_extension

schedule_node_extension::schedule_node_extension()
    : schedule_node() {}

schedule_node_extension::schedule_node_extension(const schedule_node_extension &obj)
    : schedule_node(obj)
{
}

schedule_node_extension::schedule_node_extension(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_extension &schedule_node_extension::operator=(schedule_node_extension obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_extension& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_extension& C1, const schedule_node_extension& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_extension::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_extension::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

union_map schedule_node_extension::get_extension() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_extension_get_extension(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_filter

schedule_node_filter::schedule_node_filter()
    : schedule_node() {}

schedule_node_filter::schedule_node_filter(const schedule_node_filter &obj)
    : schedule_node(obj)
{
}

schedule_node_filter::schedule_node_filter(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_filter &schedule_node_filter::operator=(schedule_node_filter obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_filter& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_filter& C1, const schedule_node_filter& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_filter::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_filter::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

union_set schedule_node_filter::get_filter() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_filter_get_filter(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_guard

schedule_node_guard::schedule_node_guard()
    : schedule_node() {}

schedule_node_guard::schedule_node_guard(const schedule_node_guard &obj)
    : schedule_node(obj)
{
}

schedule_node_guard::schedule_node_guard(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_guard &schedule_node_guard::operator=(schedule_node_guard obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_guard& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_guard& C1, const schedule_node_guard& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_guard::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_guard::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

set schedule_node_guard::get_guard() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_guard_get_guard(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_leaf

schedule_node_leaf::schedule_node_leaf()
    : schedule_node() {}

schedule_node_leaf::schedule_node_leaf(const schedule_node_leaf &obj)
    : schedule_node(obj)
{
}

schedule_node_leaf::schedule_node_leaf(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_leaf &schedule_node_leaf::operator=(schedule_node_leaf obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_leaf& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_leaf& C1, const schedule_node_leaf& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_leaf::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_leaf::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}


// implementations for isl::schedule_node_mark

schedule_node_mark::schedule_node_mark()
    : schedule_node() {}

schedule_node_mark::schedule_node_mark(const schedule_node_mark &obj)
    : schedule_node(obj)
{
}

schedule_node_mark::schedule_node_mark(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_mark &schedule_node_mark::operator=(schedule_node_mark obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_mark& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_mark& C1, const schedule_node_mark& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_mark::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_mark::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}

id schedule_node_mark::get_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_schedule_node_mark_get_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::schedule_node_sequence

schedule_node_sequence::schedule_node_sequence()
    : schedule_node() {}

schedule_node_sequence::schedule_node_sequence(const schedule_node_sequence &obj)
    : schedule_node(obj)
{
}

schedule_node_sequence::schedule_node_sequence(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_sequence &schedule_node_sequence::operator=(schedule_node_sequence obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_sequence& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_sequence& C1, const schedule_node_sequence& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_sequence::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_sequence::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}


// implementations for isl::schedule_node_set

schedule_node_set::schedule_node_set()
    : schedule_node() {}

schedule_node_set::schedule_node_set(const schedule_node_set &obj)
    : schedule_node(obj)
{
}

schedule_node_set::schedule_node_set(__isl_take isl_schedule_node *ptr)
    : schedule_node(ptr) {}


schedule_node_set &schedule_node_set::operator=(schedule_node_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}



inline std::ostream& operator<<(std::ostream& os, const schedule_node_set& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const schedule_node_set& C1, const schedule_node_set& C2) {
  return C1.is_equal(C2);
}


std::string schedule_node_set::to_str() const {
  char *Tmp = isl_schedule_node_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx schedule_node_set::get_ctx() const {
  return ctx(isl_schedule_node_get_ctx(ptr));
}


// implementations for isl::set
set manage(__isl_take isl_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return set(ptr);
}
set manage_copy(__isl_keep isl_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return set(ptr);
}

set::set()
    : ptr(nullptr) {}

set::set(const set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

set::set(__isl_take isl_set *ptr)
    : ptr(ptr) {}

set::set(point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pnt.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
set::set(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
set::set(basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_from_basic_set(bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

set &set::operator=(set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

set::~set() {
  if (ptr)
    isl_set_free(ptr);
}

__isl_give isl_set *set::copy() const & {
  return isl_set_copy(ptr);
}

__isl_keep isl_set *set::get() const {
  return ptr;
}

__isl_give isl_set *set::release() {
  isl_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool set::is_null() const {
  return ptr == nullptr;
}
set::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const set& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const set& C1, const set& C2) {
  return C1.is_equal(C2);
}


std::string set::to_str() const {
  char *Tmp = isl_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx set::get_ctx() const {
  return ctx(isl_set_get_ctx(ptr));
}

basic_set set::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::align_params(space model) const
{
  if (!ptr || model.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_align_params(copy(), model.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::apply(map map) const
{
  if (!ptr || map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_apply(copy(), map.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_coalesce(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::complement() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_complement(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_compute_divs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff set::dim_max(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_dim_max(copy(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff set::dim_min(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_dim_min(copy(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::flatten() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_flatten(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map set::flatten_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_flatten_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void set::foreach_basic_set(const std::function<void(basic_set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(basic_set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_basic_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_set_foreach_basic_set(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

set set::from(multi_aff ma)
{
  if (ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = ma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_from_multi_aff(ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::from_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_from_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::from_union_set(union_set uset)
{
  if (uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = uset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_from_union_set(uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set_list set::get_basic_set_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_get_basic_set_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space set::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val set::get_stride(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_get_stride(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id set::get_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_get_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

std::string set::get_tuple_name() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_get_tuple_name(get());
  std::string tmp(res);
  return tmp;
}

set set::gist(set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool set::has_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_has_tuple_id(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::has_tuple_name() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_has_tuple_name(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

map set::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_identity(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::intersect(set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_intersect(copy(), set2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::intersect_params(set params) const
{
  if (!ptr || params.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_intersect_params(copy(), params.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool set::involves_param(const id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_involves_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_disjoint(const set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_disjoint(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_equal(const set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_equal(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_singleton() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_singleton(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_strict_subset(const set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_strict_subset(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_subset(const set &set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_subset(get(), set2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool set::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

set set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val set::max_val(const aff &obj) const
{
  if (!ptr || obj.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_max_val(get(), obj.get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val set::min_val(const aff &obj) const
{
  if (!ptr || obj.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_min_val(get(), obj.get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int set::n_basic_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_n_basic_set(get());
  return res;
}

unsigned int set::n_dim() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_n_dim(get());
  return res;
}

unsigned int set::n_param() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_n_param(get());
  return res;
}

set set::nat_universe(space dim)
{
  if (dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = dim.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_nat_universe(dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool set::plain_is_universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_plain_is_universe(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

basic_set set::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::preimage_multi_aff(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_preimage_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::product(set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_product(copy(), set2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::reset_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_reset_tuple_id(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set set::sample() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_sample(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

point set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::set_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_set_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::set_tuple_name(const std::string &s) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_set_tuple_name(copy(), s.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set set::simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_simple_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::subtract(set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_subtract(copy(), set2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::unbind_params(multi_id tuple) const
{
  if (!ptr || tuple.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_unbind_params(copy(), tuple.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map set::unbind_params_insert_domain(multi_id domain) const
{
  if (!ptr || domain.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_unbind_params_insert_domain(copy(), domain.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::unite(set set2) const
{
  if (!ptr || set2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_union(copy(), set2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set set::universe(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_universe(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

basic_set set::unshifted_simple_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_unshifted_simple_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map set::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_unwrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map set::wrapped_domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_wrapped_domain_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::set_list
set_list manage(__isl_take isl_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return set_list(ptr);
}
set_list manage_copy(__isl_keep isl_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_set_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_set_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return set_list(ptr);
}

set_list::set_list()
    : ptr(nullptr) {}

set_list::set_list(const set_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_set_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

set_list::set_list(__isl_take isl_set_list *ptr)
    : ptr(ptr) {}

set_list::set_list(set el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_from_set(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
set_list::set_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

set_list &set_list::operator=(set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

set_list::~set_list() {
  if (ptr)
    isl_set_list_free(ptr);
}

__isl_give isl_set_list *set_list::copy() const & {
  return isl_set_list_copy(ptr);
}

__isl_keep isl_set_list *set_list::get() const {
  return ptr;
}

__isl_give isl_set_list *set_list::release() {
  isl_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool set_list::is_null() const {
  return ptr == nullptr;
}
set_list::operator bool() const
{
  return !is_null();
}



ctx set_list::get_ctx() const {
  return ctx(isl_set_list_get_ctx(ptr));
}

set_list set_list::add(set el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set_list set_list::concat(set_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set_list set_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void set_list::foreach(const std::function<void(set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_set_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

set set_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

set_list set_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int set_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_set_list_size(get());
  return res;
}

// implementations for isl::space
space manage(__isl_take isl_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return space(ptr);
}
space manage_copy(__isl_keep isl_space *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_space_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_space_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return space(ptr);
}

space::space()
    : ptr(nullptr) {}

space::space(const space &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_space_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

space::space(__isl_take isl_space *ptr)
    : ptr(ptr) {}

space::space(ctx ctx, unsigned int nparam, unsigned int n_in, unsigned int n_out)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_alloc(ctx.release(), nparam, n_in, n_out);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
space::space(ctx ctx, unsigned int nparam, unsigned int dim)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_set_alloc(ctx.release(), nparam, dim);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
space::space(ctx ctx, unsigned int nparam)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_params_alloc(ctx.release(), nparam);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

space &space::operator=(space obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

space::~space() {
  if (ptr)
    isl_space_free(ptr);
}

__isl_give isl_space *space::copy() const & {
  return isl_space_copy(ptr);
}

__isl_keep isl_space *space::get() const {
  return ptr;
}

__isl_give isl_space *space::release() {
  isl_space *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool space::is_null() const {
  return ptr == nullptr;
}
space::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const space& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const space& C1, const space& C2) {
  return C1.is_equal(C2);
}


std::string space::to_str() const {
  char *Tmp = isl_space_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx space::get_ctx() const {
  return ctx(isl_space_get_ctx(ptr));
}

space space::add_named_tuple_id_ui(id tuple_id, unsigned int dim) const
{
  if (!ptr || tuple_id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_add_named_tuple_id_ui(copy(), tuple_id.release(), dim);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::add_param(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_add_param_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::add_unnamed_tuple_ui(unsigned int dim) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_add_unnamed_tuple_ui(copy(), dim);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::align_params(space dim2) const
{
  if (!ptr || dim2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_align_params(copy(), dim2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool space::can_curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_can_curry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::can_uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_can_uncurry(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

space space::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_curry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_domain_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::domain_product(space right) const
{
  if (!ptr || right.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_domain_product(copy(), right.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::from_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_from_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::from_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_from_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

id space::get_map_range_tuple_id() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_get_map_range_tuple_id(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool space::has_equal_params(const space &space2) const
{
  if (!ptr || space2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_has_equal_params(get(), space2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::has_equal_tuples(const space &space2) const
{
  if (!ptr || space2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_has_equal_tuples(get(), space2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::has_param(const id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_has_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::is_equal(const space &space2) const
{
  if (!ptr || space2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_is_equal(get(), space2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::is_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_is_params(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::is_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_is_set(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool space::is_wrapping() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_is_wrapping(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

space space::map_from_domain_and_range(space range) const
{
  if (!ptr || range.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_map_from_domain_and_range(copy(), range.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::map_from_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_map_from_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::product(space right) const
{
  if (!ptr || right.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_product(copy(), right.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::range_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_range_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::range_product(space right) const
{
  if (!ptr || right.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_range_product(copy(), right.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::set_from_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_set_from_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::set_set_tuple_id(id id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_set_set_tuple_id(copy(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_uncurry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_unwrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space space::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_space_wrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::stride_info
stride_info manage(__isl_take isl_stride_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return stride_info(ptr);
}
stride_info manage_copy(__isl_keep isl_stride_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_stride_info_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_stride_info_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return stride_info(ptr);
}

stride_info::stride_info()
    : ptr(nullptr) {}

stride_info::stride_info(const stride_info &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_stride_info_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

stride_info::stride_info(__isl_take isl_stride_info *ptr)
    : ptr(ptr) {}


stride_info &stride_info::operator=(stride_info obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

stride_info::~stride_info() {
  if (ptr)
    isl_stride_info_free(ptr);
}

__isl_give isl_stride_info *stride_info::copy() const & {
  return isl_stride_info_copy(ptr);
}

__isl_keep isl_stride_info *stride_info::get() const {
  return ptr;
}

__isl_give isl_stride_info *stride_info::release() {
  isl_stride_info *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool stride_info::is_null() const {
  return ptr == nullptr;
}
stride_info::operator bool() const
{
  return !is_null();
}



ctx stride_info::get_ctx() const {
  return ctx(isl_stride_info_get_ctx(ptr));
}

aff stride_info::get_offset() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_stride_info_get_offset(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val stride_info::get_stride() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_stride_info_get_stride(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_access_info
union_access_info manage(__isl_take isl_union_access_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_access_info(ptr);
}
union_access_info manage_copy(__isl_keep isl_union_access_info *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_access_info_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_access_info_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_access_info(ptr);
}

union_access_info::union_access_info()
    : ptr(nullptr) {}

union_access_info::union_access_info(const union_access_info &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_access_info_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_access_info::union_access_info(__isl_take isl_union_access_info *ptr)
    : ptr(ptr) {}

union_access_info::union_access_info(union_map sink)
{
  if (sink.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = sink.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_from_sink(sink.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_access_info &union_access_info::operator=(union_access_info obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_access_info::~union_access_info() {
  if (ptr)
    isl_union_access_info_free(ptr);
}

__isl_give isl_union_access_info *union_access_info::copy() const & {
  return isl_union_access_info_copy(ptr);
}

__isl_keep isl_union_access_info *union_access_info::get() const {
  return ptr;
}

__isl_give isl_union_access_info *union_access_info::release() {
  isl_union_access_info *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_access_info::is_null() const {
  return ptr == nullptr;
}
union_access_info::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_access_info& C) {
  os << C.to_str();
  return os;
}


std::string union_access_info::to_str() const {
  char *Tmp = isl_union_access_info_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_access_info::get_ctx() const {
  return ctx(isl_union_access_info_get_ctx(ptr));
}

union_flow union_access_info::compute_flow() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_compute_flow(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_access_info union_access_info::set_kill(union_map kill) const
{
  if (!ptr || kill.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_set_kill(copy(), kill.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_access_info union_access_info::set_may_source(union_map may_source) const
{
  if (!ptr || may_source.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_set_may_source(copy(), may_source.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_access_info union_access_info::set_must_source(union_map must_source) const
{
  if (!ptr || must_source.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_set_must_source(copy(), must_source.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_access_info union_access_info::set_schedule(schedule schedule) const
{
  if (!ptr || schedule.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_set_schedule(copy(), schedule.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_access_info union_access_info::set_schedule_map(union_map schedule_map) const
{
  if (!ptr || schedule_map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_access_info_set_schedule_map(copy(), schedule_map.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_flow
union_flow manage(__isl_take isl_union_flow *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_flow(ptr);
}
union_flow manage_copy(__isl_keep isl_union_flow *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_flow_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_flow_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_flow(ptr);
}

union_flow::union_flow()
    : ptr(nullptr) {}

union_flow::union_flow(const union_flow &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_flow_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_flow::union_flow(__isl_take isl_union_flow *ptr)
    : ptr(ptr) {}


union_flow &union_flow::operator=(union_flow obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_flow::~union_flow() {
  if (ptr)
    isl_union_flow_free(ptr);
}

__isl_give isl_union_flow *union_flow::copy() const & {
  return isl_union_flow_copy(ptr);
}

__isl_keep isl_union_flow *union_flow::get() const {
  return ptr;
}

__isl_give isl_union_flow *union_flow::release() {
  isl_union_flow *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_flow::is_null() const {
  return ptr == nullptr;
}
union_flow::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_flow& C) {
  os << C.to_str();
  return os;
}


std::string union_flow::to_str() const {
  char *Tmp = isl_union_flow_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_flow::get_ctx() const {
  return ctx(isl_union_flow_get_ctx(ptr));
}

union_map union_flow::get_full_may_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_full_may_dependence(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_flow::get_full_must_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_full_must_dependence(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_flow::get_may_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_may_dependence(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_flow::get_may_no_source() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_may_no_source(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_flow::get_must_dependence() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_must_dependence(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_flow::get_must_no_source() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_flow_get_must_no_source(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_map
union_map manage(__isl_take isl_union_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_map(ptr);
}
union_map manage_copy(__isl_keep isl_union_map *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_map_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_map_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_map(ptr);
}

union_map::union_map()
    : ptr(nullptr) {}

union_map::union_map(const union_map &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_map_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_map::union_map(__isl_take isl_union_map *ptr)
    : ptr(ptr) {}

union_map::union_map(basic_map bmap)
{
  if (bmap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bmap.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_basic_map(bmap.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_map::union_map(map map)
{
  if (map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = map.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_map(map.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_map::union_map(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_map &union_map::operator=(union_map obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_map::~union_map() {
  if (ptr)
    isl_union_map_free(ptr);
}

__isl_give isl_union_map *union_map::copy() const & {
  return isl_union_map_copy(ptr);
}

__isl_keep isl_union_map *union_map::get() const {
  return ptr;
}

__isl_give isl_union_map *union_map::release() {
  isl_union_map *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_map::is_null() const {
  return ptr == nullptr;
}
union_map::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_map& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const union_map& C1, const union_map& C2) {
  return C1.is_equal(C2);
}


std::string union_map::to_str() const {
  char *Tmp = isl_union_map_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_map::get_ctx() const {
  return ctx(isl_union_map_get_ctx(ptr));
}

union_map union_map::add_map(map map) const
{
  if (!ptr || map.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_add_map(copy(), map.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::apply_domain(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_apply_domain(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::apply_range(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_apply_range(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_coalesce(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_compute_divs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::curry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_curry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_map::deltas() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_deltas(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_map::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::domain_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::domain_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_map::domain_map_union_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain_map_union_pw_multi_aff(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::domain_product(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_domain_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::eq_at(multi_union_pw_aff mupa) const
{
  if (!ptr || mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_eq_at_multi_union_pw_aff(copy(), mupa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map union_map::extract_map(space dim) const
{
  if (!ptr || dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_extract_map(get(), dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::fixed_power(val exp) const
{
  if (!ptr || exp.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_fixed_power_val(copy(), exp.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::flat_range_product(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_flat_range_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_map::foreach_map(const std::function<void(map)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(map)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_map *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_map_foreach_map(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

union_map union_map::from(union_pw_multi_aff upma)
{
  if (upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = upma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_union_pw_multi_aff(upma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::from(multi_union_pw_aff mupa)
{
  if (mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = mupa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_multi_union_pw_aff(mupa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::from_domain(union_set uset)
{
  if (uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = uset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_domain(uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::from_domain_and_range(union_set domain, union_set range)
{
  if (domain.is_null() || range.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_domain_and_range(domain.release(), range.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::from_range(union_set uset)
{
  if (uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = uset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_from_range(uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

map_list union_map::get_map_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_get_map_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space union_map::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::gist(union_map context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::gist_domain(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_gist_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::gist_params(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_gist_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::gist_range(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_gist_range(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::intersect(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_intersect(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::intersect_domain(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_intersect_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::intersect_params(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::intersect_range(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_intersect_range(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool union_map::is_bijective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_bijective(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_equal(const union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_equal(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_injective() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_injective(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_single_valued() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_single_valued(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_strict_subset(const union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_strict_subset(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_map::is_subset(const union_map &umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_is_subset(get(), umap2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

union_map union_map::lex_gt_at(multi_union_pw_aff mupa) const
{
  if (!ptr || mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_lex_gt_at_multi_union_pw_aff(copy(), mupa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::lex_lt_at(multi_union_pw_aff mupa) const
{
  if (!ptr || mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_lex_lt_at_multi_union_pw_aff(copy(), mupa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_map::n_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_n_map(get());
  return res;
}

union_map union_map::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::preimage_range_multi_aff(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_preimage_range_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::product(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::project_out_all_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_project_out_all_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_map::range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::range_factor_domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_range_factor_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::range_factor_range() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_range_factor_range(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::range_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_range_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::range_product(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_range_product(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::subtract(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_subtract(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::subtract_domain(union_set dom) const
{
  if (!ptr || dom.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_subtract_domain(copy(), dom.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::subtract_range(union_set dom) const
{
  if (!ptr || dom.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_subtract_range(copy(), dom.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::uncurry() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_uncurry(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::unite(union_map umap2) const
{
  if (!ptr || umap2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_union(copy(), umap2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_universe(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_map::wrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_wrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_map::zip() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_map_zip(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_pw_aff
union_pw_aff manage(__isl_take isl_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_aff(ptr);
}
union_pw_aff manage_copy(__isl_keep isl_union_pw_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_pw_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_pw_aff(ptr);
}

union_pw_aff::union_pw_aff()
    : ptr(nullptr) {}

union_pw_aff::union_pw_aff(const union_pw_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_pw_aff::union_pw_aff(__isl_take isl_union_pw_aff *ptr)
    : ptr(ptr) {}

union_pw_aff::union_pw_aff(pw_aff pa)
{
  if (pa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_from_pw_aff(pa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_aff::union_pw_aff(union_set domain, val v)
{
  if (domain.is_null() || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_val_on_domain(domain.release(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_aff::union_pw_aff(union_set domain, aff aff)
{
  if (domain.is_null() || aff.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_aff_on_domain(domain.release(), aff.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_aff::union_pw_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_pw_aff &union_pw_aff::operator=(union_pw_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_aff::~union_pw_aff() {
  if (ptr)
    isl_union_pw_aff_free(ptr);
}

__isl_give isl_union_pw_aff *union_pw_aff::copy() const & {
  return isl_union_pw_aff_copy(ptr);
}

__isl_keep isl_union_pw_aff *union_pw_aff::get() const {
  return ptr;
}

__isl_give isl_union_pw_aff *union_pw_aff::release() {
  isl_union_pw_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_aff::is_null() const {
  return ptr == nullptr;
}
union_pw_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_pw_aff& C) {
  os << C.to_str();
  return os;
}


std::string union_pw_aff::to_str() const {
  char *Tmp = isl_union_pw_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_pw_aff::get_ctx() const {
  return ctx(isl_union_pw_aff_get_ctx(ptr));
}

union_pw_aff union_pw_aff::add(union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_add(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_pw_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff union_pw_aff::extract_on_domain(space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_extract_on_domain_space(get(), space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_aff union_pw_aff::extract_pw_aff(space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_extract_pw_aff(get(), space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_pw_aff::foreach_pw_aff(const std::function<void(pw_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(pw_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_pw_aff_foreach_pw_aff(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

pw_aff_list union_pw_aff::get_pw_aff_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_get_pw_aff_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space union_pw_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::intersect_domain(union_set uset) const
{
  if (!ptr || uset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_intersect_domain(copy(), uset.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool union_pw_aff::involves_param(const id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_involves_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val union_pw_aff::max_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_max_val(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val union_pw_aff::min_val() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_min_val(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::mod(val f) const
{
  if (!ptr || f.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_mod_val(copy(), f.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_pw_aff::n_pw_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_n_pw_aff(get());
  return res;
}

union_pw_aff union_pw_aff::param_on_domain(union_set domain, id id)
{
  if (domain.is_null() || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_param_on_domain_id(domain.release(), id.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool union_pw_aff::plain_is_equal(const union_pw_aff &upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_plain_is_equal(get(), upa2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

union_pw_aff union_pw_aff::pullback(union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_pullback_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::scale(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_scale_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::scale_down(val v) const
{
  if (!ptr || v.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_scale_down_val(copy(), v.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::sub(union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_sub(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_aff::union_add(union_pw_aff upa2) const
{
  if (!ptr || upa2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_union_add(copy(), upa2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_pw_aff::zero_union_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_zero_union_set(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_pw_aff_list
union_pw_aff_list manage(__isl_take isl_union_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_aff_list(ptr);
}
union_pw_aff_list manage_copy(__isl_keep isl_union_pw_aff_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_aff_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_pw_aff_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_pw_aff_list(ptr);
}

union_pw_aff_list::union_pw_aff_list()
    : ptr(nullptr) {}

union_pw_aff_list::union_pw_aff_list(const union_pw_aff_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_aff_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_pw_aff_list::union_pw_aff_list(__isl_take isl_union_pw_aff_list *ptr)
    : ptr(ptr) {}

union_pw_aff_list::union_pw_aff_list(union_pw_aff el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_from_union_pw_aff(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_aff_list::union_pw_aff_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_pw_aff_list &union_pw_aff_list::operator=(union_pw_aff_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_aff_list::~union_pw_aff_list() {
  if (ptr)
    isl_union_pw_aff_list_free(ptr);
}

__isl_give isl_union_pw_aff_list *union_pw_aff_list::copy() const & {
  return isl_union_pw_aff_list_copy(ptr);
}

__isl_keep isl_union_pw_aff_list *union_pw_aff_list::get() const {
  return ptr;
}

__isl_give isl_union_pw_aff_list *union_pw_aff_list::release() {
  isl_union_pw_aff_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_aff_list::is_null() const {
  return ptr == nullptr;
}
union_pw_aff_list::operator bool() const
{
  return !is_null();
}



ctx union_pw_aff_list::get_ctx() const {
  return ctx(isl_union_pw_aff_list_get_ctx(ptr));
}

union_pw_aff_list union_pw_aff_list::add(union_pw_aff el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::concat(union_pw_aff_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_pw_aff_list::foreach(const std::function<void(union_pw_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(union_pw_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_pw_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_pw_aff_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

union_pw_aff union_pw_aff_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff_list union_pw_aff_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_pw_aff_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_aff_list_size(get());
  return res;
}

// implementations for isl::union_pw_multi_aff
union_pw_multi_aff manage(__isl_take isl_union_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_pw_multi_aff(ptr);
}
union_pw_multi_aff manage_copy(__isl_keep isl_union_pw_multi_aff *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_multi_aff_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_pw_multi_aff_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_pw_multi_aff(ptr);
}

union_pw_multi_aff::union_pw_multi_aff()
    : ptr(nullptr) {}

union_pw_multi_aff::union_pw_multi_aff(const union_pw_multi_aff &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_pw_multi_aff_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_pw_multi_aff::union_pw_multi_aff(__isl_take isl_union_pw_multi_aff *ptr)
    : ptr(ptr) {}

union_pw_multi_aff::union_pw_multi_aff(pw_multi_aff pma)
{
  if (pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pma.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_pw_multi_aff(pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(union_set domain, multi_val mv)
{
  if (domain.is_null() || mv.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = domain.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_multi_val_on_domain(domain.release(), mv.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_pw_multi_aff::union_pw_multi_aff(union_pw_aff upa)
{
  if (upa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = upa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_union_pw_aff(upa.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_pw_multi_aff &union_pw_multi_aff::operator=(union_pw_multi_aff obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_pw_multi_aff::~union_pw_multi_aff() {
  if (ptr)
    isl_union_pw_multi_aff_free(ptr);
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::copy() const & {
  return isl_union_pw_multi_aff_copy(ptr);
}

__isl_keep isl_union_pw_multi_aff *union_pw_multi_aff::get() const {
  return ptr;
}

__isl_give isl_union_pw_multi_aff *union_pw_multi_aff::release() {
  isl_union_pw_multi_aff *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_pw_multi_aff::is_null() const {
  return ptr == nullptr;
}
union_pw_multi_aff::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_pw_multi_aff& C) {
  os << C.to_str();
  return os;
}


std::string union_pw_multi_aff::to_str() const {
  char *Tmp = isl_union_pw_multi_aff_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_pw_multi_aff::get_ctx() const {
  return ctx(isl_union_pw_multi_aff_get_ctx(ptr));
}

union_pw_multi_aff union_pw_multi_aff::add(union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_add(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_pw_multi_aff::domain() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_domain(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

pw_multi_aff union_pw_multi_aff::extract_pw_multi_aff(space space) const
{
  if (!ptr || space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_extract_pw_multi_aff(get(), space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::flat_range_product(union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_flat_range_product(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_pw_multi_aff::foreach_pw_multi_aff(const std::function<void(pw_multi_aff)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(pw_multi_aff)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_pw_multi_aff *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_pw_multi_aff_foreach_pw_multi_aff(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

union_pw_multi_aff union_pw_multi_aff::from(union_map umap)
{
  if (umap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = umap.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_union_map(umap.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::from_multi_union_pw_aff(multi_union_pw_aff mupa)
{
  if (mupa.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = mupa.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_from_multi_union_pw_aff(mupa.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space union_pw_multi_aff::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_aff union_pw_multi_aff::get_union_pw_aff(int pos) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_get_union_pw_aff(get(), pos);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_pw_multi_aff::n_pw_multi_aff() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_n_pw_multi_aff(get());
  return res;
}

union_pw_multi_aff union_pw_multi_aff::pullback(union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_pullback_union_pw_multi_aff(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::scale(val val) const
{
  if (!ptr || val.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_scale_val(copy(), val.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::scale_down(val val) const
{
  if (!ptr || val.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_scale_down_val(copy(), val.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_pw_multi_aff union_pw_multi_aff::union_add(union_pw_multi_aff upma2) const
{
  if (!ptr || upma2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_pw_multi_aff_union_add(copy(), upma2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_set
union_set manage(__isl_take isl_union_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_set(ptr);
}
union_set manage_copy(__isl_keep isl_union_set *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_set_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_set_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_set(ptr);
}

union_set::union_set()
    : ptr(nullptr) {}

union_set::union_set(const union_set &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_set_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_set::union_set(__isl_take isl_union_set *ptr)
    : ptr(ptr) {}

union_set::union_set(basic_set bset)
{
  if (bset.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = bset.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_from_basic_set(bset.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_set::union_set(set set)
{
  if (set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = set.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_from_set(set.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_set::union_set(point pnt)
{
  if (pnt.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = pnt.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_from_point(pnt.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_set::union_set(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_set &union_set::operator=(union_set obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_set::~union_set() {
  if (ptr)
    isl_union_set_free(ptr);
}

__isl_give isl_union_set *union_set::copy() const & {
  return isl_union_set_copy(ptr);
}

__isl_keep isl_union_set *union_set::get() const {
  return ptr;
}

__isl_give isl_union_set *union_set::release() {
  isl_union_set *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_set::is_null() const {
  return ptr == nullptr;
}
union_set::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const union_set& C) {
  os << C.to_str();
  return os;
}

inline bool operator==(const union_set& C1, const union_set& C2) {
  return C1.is_equal(C2);
}


std::string union_set::to_str() const {
  char *Tmp = isl_union_set_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx union_set::get_ctx() const {
  return ctx(isl_union_set_get_ctx(ptr));
}

union_set union_set::add_set(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_add_set(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::affine_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_affine_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::apply(union_map umap) const
{
  if (!ptr || umap.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_apply(copy(), umap.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::coalesce() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_coalesce(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::compute_divs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_compute_divs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::detect_equalities() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_detect_equalities(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::empty(space space)
{
  if (space.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = space.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_empty(space.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool union_set::every_set(const std::function<bool(set)> &test) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct test_data {
    std::function<bool(set)> func;
    std::exception_ptr eptr;
  } test_data = { test };
  auto test_lambda = [](isl_set *arg_0, void *arg_1) -> isl_bool {
    auto *data = static_cast<struct test_data *>(arg_1);
    ISL_CPP_TRY {
      auto ret = (data->func)(manage_copy(arg_0));
      return ret ? isl_bool_true : isl_bool_false;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_bool_error;
    }
  };
  auto res = isl_union_set_every_set(get(), test_lambda, &test_data);
  if (test_data.eptr)
    std::rethrow_exception(test_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

set union_set::extract_set(space dim) const
{
  if (!ptr || dim.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_extract_set(get(), dim.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_set::foreach_point(const std::function<void(point)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(point)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_point *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_foreach_point(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

void union_set::foreach_set(const std::function<void(set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_foreach_set(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

set_list union_set::get_set_list() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_get_set_list(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

space union_set::get_space() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_get_space(get());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::gist(union_set context) const
{
  if (!ptr || context.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_gist(copy(), context.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::gist_params(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_gist_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_set::identity() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_identity(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::intersect(union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_intersect(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::intersect_params(set set) const
{
  if (!ptr || set.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_intersect_params(copy(), set.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool union_set::involves_param(const id &id) const
{
  if (!ptr || id.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_involves_param_id(get(), id.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_disjoint(const union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_disjoint(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_empty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_empty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_equal(const union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_equal(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_params(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_strict_subset(const union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_strict_subset(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool union_set::is_subset(const union_set &uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_is_subset(get(), uset2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

union_set union_set::lexmax() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_lexmax(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::lexmin() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_lexmin(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_set::n_set() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_n_set(get());
  return res;
}

set union_set::params() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_params(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::polyhedral_hull() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_polyhedral_hull(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::preimage(multi_aff ma) const
{
  if (!ptr || ma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_preimage_multi_aff(copy(), ma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::preimage(pw_multi_aff pma) const
{
  if (!ptr || pma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_preimage_pw_multi_aff(copy(), pma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::preimage(union_pw_multi_aff upma) const
{
  if (!ptr || upma.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_preimage_union_pw_multi_aff(copy(), upma.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

point union_set::sample_point() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_sample_point(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::subtract(union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_subtract(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::unite(union_set uset2) const
{
  if (!ptr || uset2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_union(copy(), uset2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set union_set::universe() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_universe(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_set::unwrap() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_unwrap(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_map union_set::wrapped_domain_map() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_wrapped_domain_map(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::union_set_list
union_set_list manage(__isl_take isl_union_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return union_set_list(ptr);
}
union_set_list manage_copy(__isl_keep isl_union_set_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_set_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_union_set_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return union_set_list(ptr);
}

union_set_list::union_set_list()
    : ptr(nullptr) {}

union_set_list::union_set_list(const union_set_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_union_set_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

union_set_list::union_set_list(__isl_take isl_union_set_list *ptr)
    : ptr(ptr) {}

union_set_list::union_set_list(union_set el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_from_union_set(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
union_set_list::union_set_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

union_set_list &union_set_list::operator=(union_set_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

union_set_list::~union_set_list() {
  if (ptr)
    isl_union_set_list_free(ptr);
}

__isl_give isl_union_set_list *union_set_list::copy() const & {
  return isl_union_set_list_copy(ptr);
}

__isl_keep isl_union_set_list *union_set_list::get() const {
  return ptr;
}

__isl_give isl_union_set_list *union_set_list::release() {
  isl_union_set_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool union_set_list::is_null() const {
  return ptr == nullptr;
}
union_set_list::operator bool() const
{
  return !is_null();
}



ctx union_set_list::get_ctx() const {
  return ctx(isl_union_set_list_get_ctx(ptr));
}

union_set_list union_set_list::add(union_set el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set_list union_set_list::concat(union_set_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set_list union_set_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void union_set_list::foreach(const std::function<void(union_set)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(union_set)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_union_set *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_union_set_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

union_set union_set_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

union_set_list union_set_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int union_set_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_union_set_list_size(get());
  return res;
}

// implementations for isl::val
val manage(__isl_take isl_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return val(ptr);
}
val manage_copy(__isl_keep isl_val *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_val_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_val_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return val(ptr);
}

val::val()
    : ptr(nullptr) {}

val::val(const val &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_val_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

val::val(__isl_take isl_val *ptr)
    : ptr(ptr) {}

val::val(ctx ctx, const std::string &str)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_read_from_str(ctx.release(), str.c_str());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
val::val(ctx ctx, long i)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_int_from_si(ctx.release(), i);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

val &val::operator=(val obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

val::~val() {
  if (ptr)
    isl_val_free(ptr);
}

__isl_give isl_val *val::copy() const & {
  return isl_val_copy(ptr);
}

__isl_keep isl_val *val::get() const {
  return ptr;
}

__isl_give isl_val *val::release() {
  isl_val *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool val::is_null() const {
  return ptr == nullptr;
}
val::operator bool() const
{
  return !is_null();
}

inline std::ostream& operator<<(std::ostream& os, const val& C) {
  os << C.to_str();
  return os;
}


std::string val::to_str() const {
  char *Tmp = isl_val_to_str(get());
  if (!Tmp)
    return "";
  std::string S(Tmp);
  free(Tmp);
  return S;
}


ctx val::get_ctx() const {
  return ctx(isl_val_get_ctx(ptr));
}

val val::abs() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_abs(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool val::abs_eq(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_abs_eq(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val val::add(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_add(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::ceil() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_ceil(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int val::cmp_si(long i) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_cmp_si(get(), i);
  return res;
}

val val::div(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_div(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool val::eq(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_eq(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val val::floor() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_floor(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::gcd(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_gcd(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool val::ge(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_ge(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

long val::get_den_si() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_get_den_si(get());
  return res;
}

long val::get_num_si() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_get_num_si(get());
  return res;
}

bool val::gt(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_gt(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val val::infty(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_infty(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::inv() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_inv(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool val::is_divisible_by(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_divisible_by(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_infty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_infty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_int() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_int(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_nan() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_nan(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_neg(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_neginfty() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_neginfty(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_negone() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_negone(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_nonneg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_nonneg(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_nonpos() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_nonpos(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_one() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_one(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_pos() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_pos(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_rat() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_rat(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::is_zero() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_is_zero(get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::le(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_le(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

bool val::lt(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_lt(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val val::max(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_max(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::min(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_min(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::mod(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_mod(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::mul(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_mul(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::nan(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_nan(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

bool val::ne(const val &v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_ne(get(), v2.get());
  if (res < 0)
    exception::throw_last_error(ctx);
  return bool(res);
}

val val::neg() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_neg(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::neginfty(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_neginfty(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::negone(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_negone(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::one(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_one(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int val::sgn() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_sgn(get());
  return res;
}

val val::sub(val v2) const
{
  if (!ptr || v2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_sub(copy(), v2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::trunc() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_trunc(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val val::zero(ctx ctx)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_zero(ctx.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

// implementations for isl::val_list
val_list manage(__isl_take isl_val_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  return val_list(ptr);
}
val_list manage_copy(__isl_keep isl_val_list *ptr) {
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_val_list_get_ctx(ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = isl_val_list_copy(ptr);
  if (!ptr)
    exception::throw_last_error(ctx);
  return val_list(ptr);
}

val_list::val_list()
    : ptr(nullptr) {}

val_list::val_list(const val_list &obj)
    : ptr(nullptr)
{
  if (!obj.ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = isl_val_list_get_ctx(obj.ptr);
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  ptr = obj.copy();
  if (obj.ptr && !ptr)
    exception::throw_last_error(ctx);
}

val_list::val_list(__isl_take isl_val_list *ptr)
    : ptr(ptr) {}

val_list::val_list(val el)
{
  if (el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = el.get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_from_val(el.release());
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}
val_list::val_list(ctx ctx, int n)
{
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_alloc(ctx.release(), n);
  if (!res)
    exception::throw_last_error(ctx);
  ptr = res;
}

val_list &val_list::operator=(val_list obj) {
  std::swap(this->ptr, obj.ptr);
  return *this;
}

val_list::~val_list() {
  if (ptr)
    isl_val_list_free(ptr);
}

__isl_give isl_val_list *val_list::copy() const & {
  return isl_val_list_copy(ptr);
}

__isl_keep isl_val_list *val_list::get() const {
  return ptr;
}

__isl_give isl_val_list *val_list::release() {
  isl_val_list *tmp = ptr;
  ptr = nullptr;
  return tmp;
}

bool val_list::is_null() const {
  return ptr == nullptr;
}
val_list::operator bool() const
{
  return !is_null();
}



ctx val_list::get_ctx() const {
  return ctx(isl_val_list_get_ctx(ptr));
}

val_list val_list::add(val el) const
{
  if (!ptr || el.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_add(copy(), el.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val_list val_list::concat(val_list list2) const
{
  if (!ptr || list2.is_null())
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_concat(copy(), list2.release());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val_list val_list::drop(unsigned int first, unsigned int n) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_drop(copy(), first, n);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

void val_list::foreach(const std::function<void(val)> &fn) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  struct fn_data {
    std::function<void(val)> func;
    std::exception_ptr eptr;
  } fn_data = { fn };
  auto fn_lambda = [](isl_val *arg_0, void *arg_1) -> isl_stat {
    auto *data = static_cast<struct fn_data *>(arg_1);
    ISL_CPP_TRY {
      (data->func)(manage(arg_0));
      return isl_stat_ok;
    } ISL_CPP_CATCH_ALL {
      data->eptr = std::current_exception();
      return isl_stat_error;
    }
  };
  auto res = isl_val_list_foreach(get(), fn_lambda, &fn_data);
  if (fn_data.eptr)
    std::rethrow_exception(fn_data.eptr);
  if (res < 0)
    exception::throw_last_error(ctx);
  return void(res);
}

val val_list::get_at(int index) const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_get_at(get(), index);
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

val_list val_list::reverse() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_reverse(copy());
  if (!res)
    exception::throw_last_error(ctx);
  return manage(res);
}

int val_list::size() const
{
  if (!ptr)
    exception::throw_invalid("NULL input", __FILE__, __LINE__);
  auto ctx = get_ctx();
  options_scoped_set_on_error saved_on_error(ctx, exception::on_error);
  auto res = isl_val_list_size(get());
  return res;
}
} // namespace isl

#endif /* ISL_CPP */
