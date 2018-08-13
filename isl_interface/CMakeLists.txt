# Location of the isl submodule.
SET(ISL_DIR "${PROJECT_SOURCE_DIR}/third-party/islpp")
# Location of extract_interface source files.
SET(ISL_INTERFACE_DIR "${ISL_DIR}/interface")

add_definitions(-DCLANG_PREFIX="${CLANG_PREFIX}")

# Hard-coded defines for LLVM/clang 5.0.0
add_definitions(-DADDPATH_TAKES_4_ARGUMENTS)
add_definitions(-DCREATETARGETINFO_TAKES_SHARED_PTR)
add_definitions(-DCREATEPREPROCESSOR_TAKES_TUKIND)
add_definitions(-DHAVE_BASIC_DIAGNOSTICOPTIONS_H)
add_definitions(-DHAVE_LEX_PREPROCESSOROPTIONS_H)
add_definitions(-DHAVE_SETMAINFILEID)
add_definitions(-DSETINVOCATION_TAKES_SHARED_PTR)
add_definitions(-DSETLANGDEFAULTS_TAKES_5_ARGUMENTS)
add_definitions(-DUSE_ARRAYREF)
add_definitions(-DgetNumArgs=getNumParams)
add_definitions(-DgetArgType=getParamType)
add_definitions(-DHandleTopLevelDeclReturn=bool)
add_definitions(-DHandleTopLevelDeclContinue=true)
add_definitions(-DIK_C=InputKind::C)

find_library(CLANG_FRONTEND_LIB REQUIRED clangFrontend
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_SERIALIZATION_LIB REQUIRED clangSerialization
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_PARSE_LIB REQUIRED clangParse
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_SEMA_LIB REQUIRED clangSema
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_EDIT_LIB REQUIRED clangEdit
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_ANALYSIS_LIB REQUIRED clangAnalysis
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_AST_LIB REQUIRED clangAST
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_LEX_LIB REQUIRED clangLex
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_BASIC_LIB REQUIRED clangBasic
  ${CLANG_PREFIX} PATH_SUFFIXES lib)
find_library(CLANG_DRIVER_LIB REQUIRED clangDriver
  ${CLANG_PREFIX} PATH_SUFFIXES lib)

SET(CLANG_LIBRARIES
  "${CLANG_FRONTEND_LIB}"
  "${CLANG_SERIALIZATION_LIB}"
  "${CLANG_PARSE_LIB}"
  "${CLANG_SEMA_LIB}"
  "${CLANG_EDIT_LIB}"
  "${CLANG_ANALYSIS_LIB}"
  "${CLANG_AST_LIB}"
  "${CLANG_LEX_LIB}"
  "${CLANG_BASIC_LIB}"
  "${CLANG_DRIVER_LIB}"
)

find_program(LLVM_CONFIG_BIN llvm-config PATHS ${CLANG_PREFIX}
  PATH_SUFFIXES bin
  NO_DEFAULT_PATH)

# The process of finding the right libraries to link against
# is based on isl's m4/ax_detect_clang.m4.
execute_process(COMMAND
  "${LLVM_CONFIG_BIN}" --targets-built
  OUTPUT_VARIABLE LLVM_TARGETS OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(LLVM_COMPONENTS ${LLVM_TARGETS} asmparser bitreader support mc option)
execute_process(COMMAND
  "${LLVM_CONFIG_BIN}" --libs ${LLVM_COMPONENTS}
  OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND
  "${LLVM_CONFIG_BIN}" --system-libs
  OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

# Add current directory for mock isl_config.h
include_directories(.)
add_executable(extract_isl_interface
  ${ISL_INTERFACE_DIR}/cpp.cc
  ${ISL_INTERFACE_DIR}/extract_interface.cc
  ${ISL_INTERFACE_DIR}/generator.cc
  ${ISL_INTERFACE_DIR}/python.cc
)
target_link_libraries(extract_isl_interface
  "${CLANG_LIBRARIES}" ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})

# Dummy library to ensure that C++ bindings depend on contents of header files.
add_library(isl_all_h_dep STATIC ${ISL_DIR}/all.c)

add_custom_command(
  OUTPUT ${ISL_CPP_H}
  DEPENDS isl_all_h_dep
  DEPENDS ${ISL_DIR}/cpp/cpp.h.top
  DEPENDS ${ISL_DIR}/cpp/cpp.h.pre
  DEPENDS ${ISL_DIR}/all.h
  DEPENDS ${ISL_DIR}/cpp/cpp.h.bot
  COMMAND mkdir -p ${CMAKE_CURRENT_LIST_DIR}/include/isl
  COMMAND cat ${ISL_DIR}/cpp/cpp.h.top > ${ISL_CPP_H} || exit 1
  # Turn off reformatting on this generated file.
  COMMAND echo "// clang-format off" >> ${ISL_CPP_H} || exit 1
  COMMAND cat ${ISL_DIR}/all.h >> ${ISL_CPP_H} || exit 1
  COMMAND cat ${ISL_DIR}/cpp/cpp.h.pre >> ${ISL_CPP_H} || exit 1
  COMMAND extract_isl_interface --language=cpp
    -I${ISL_DIR}/include ${ISL_DIR}/all.h
    -I${CMAKE_CURRENT_LIST_DIR}/include >> ${ISL_CPP_H} || exit 1
  COMMAND cat ${ISL_DIR}/cpp/cpp.h.bot >> ${ISL_CPP_H} || exit 1
  DEPENDS extract_isl_interface
)

add_custom_target(generate_isl_cpp_h_core DEPENDS ${ISL_CPP_H})
