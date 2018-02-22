IF(DEFINED CILK_PATH)
    message(STATUS "Using: "${CILK_PATH}) 
ELSE()
    message(FATAL_ERROR "Please specify the full path to libcilkrts.so with -DCILK_PATH=...") 
ENDIF()

SET(CILK_PATH ${CILK_PATH})
configure_file(libpaths.cc.in libpaths.cc)
