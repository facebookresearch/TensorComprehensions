set(AUTOTUNER_FILES

  genetic_search.cc
  parameters.cc
  utils.cc
  cpu/autotuner.cc
)

if (WITH_CUDA)
set(AUTOTUNER_FILES ${AUTOTUNER_FILES} cuda/autotuner.cc)
endif()

add_library(
  tc_autotuner

  SHARED

  ${AUTOTUNER_FILES}
)

target_include_directories(tc_autotuner PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (WITH_CUDA)
  target_link_libraries(tc_autotuner PUBLIC ${ATEN_LIBRARIES} tc_cuda tc_core_cpu)
else()
  target_link_libraries(tc_autotuner PUBLIC ${ATEN_LIBRARIES} tc_core_cpu)
endif()

install(TARGETS tc_autotuner DESTINATION lib)
