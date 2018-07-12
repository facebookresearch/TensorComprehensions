#Default protobuf function err'd with python,
#  borrowing caffe2 function to generate
function(tc_protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: tc_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
             "${output_dir}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()

tc_protobuf_generate_cpp_py(${CMAKE_CURRENT_BINARY_DIR} PROTO_SRCS PROTO_HDRS PROTO_PY mapping_options.proto compcache.proto)

add_library(tc_proto SHARED ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tc_proto ${PROTOBUF_LIBRARIES})
target_include_directories(tc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_HDRS})

install(TARGETS tc_proto DESTINATION lib)
