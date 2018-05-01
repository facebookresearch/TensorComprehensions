if (WITH_CUDA)
  add_library(
    tc_autotuner

    SHARED

    genetic_autotuner.cc
    genetic_autotuner_aten.cc
    genetic_search.cc
    genetic_tuning_harness.cc
    parameters.cc
    utils.cc)

  target_include_directories(tc_autotuner PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(tc_autotuner PUBLIC ${ATEN_LIBRARIES} tc_cuda tc_proto)

  install(TARGETS tc_autotuner DESTINATION lib)
endif()
